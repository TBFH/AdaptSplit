import argparse
import copy
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_std(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    std = array.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return std.astype(np.float32)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


# =========================
# Model
# =========================

class LatencyPowerPredictor(nn.Module):
    """
    Input: [num_layer, batch_size]
    Output: [batch_latency_ms, avg_power_W]
    """

    def __init__(self, input_dim: int = 2, hidden_dims: Tuple[int, ...] = (64, 64, 32), dropout: float = 0.05):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PredictorBundle:
    model: LatencyPowerPredictor
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    model_name: str
    device_name: str

    @torch.no_grad()
    def predict(self, num_layer: float, batch_size: float, device: str = "cuda") -> Tuple[float, float]:
        self.model.eval()
        x = np.array([[float(num_layer), float(batch_size)]], dtype=np.float32)
        x_norm = (x - self.x_mean) / self.x_std
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
        y_norm = self.model(x_tensor).cpu().numpy()
        y = y_norm * self.y_std + self.y_mean
        latency_ms, power_w = y[0].tolist()
        latency_ms = max(float(latency_ms), 0.0)
        power_w = max(float(power_w), 0.0)
        return latency_ms, power_w


# =========================
# Data processing
# =========================

def parse_json_records(json_path: str) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """
    将原始 json 解析为按 (model_name, device_name) 分组的数据:
    X -> [[num_layer, batch_size], ...]
    Y -> [[latency_ms, power_W], ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    grouped: Dict[Tuple[str, str], Dict[str, List[List[float]]]] = {}

    for record_idx, record in enumerate(records):
        model_name = record["model"]
        devices = record["devices"]
        num_layer = float(record["num_layer"])
        batch_sizes = record["batch_sizes"]
        batch_latencies = record["batch_latencys_ms"]
        powers = record["powers_W"]

        if len(devices) != len(batch_latencies) or len(devices) != len(powers):
            raise ValueError(
                f"记录 {record_idx} 中 devices / batch_latencys_ms / powers_W 的外层长度不一致。"
            )

        for dev_idx, device_name in enumerate(devices):
            latency_list = batch_latencies[dev_idx]
            power_list = powers[dev_idx]
            if len(batch_sizes) != len(latency_list) or len(batch_sizes) != len(power_list):
                raise ValueError(
                    f"记录 {record_idx}, 设备 {device_name} 中 batch_sizes / latency / power 长度不一致。"
                )

            key = (model_name, device_name)
            grouped.setdefault(key, {"X": [], "Y": []})

            for bs, latency_ms, power_w in zip(batch_sizes, latency_list, power_list):
                grouped[key]["X"].append([float(num_layer), float(bs)])
                grouped[key]["Y"].append([float(latency_ms), float(power_w)])

    grouped_np: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for key, value in grouped.items():
        X = np.asarray(value["X"], dtype=np.float32)
        Y = np.asarray(value["Y"], dtype=np.float32)
        grouped_np[key] = {"X": X, "Y": Y}

    return grouped_np


# =========================
# Training
# =========================

def build_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    val_ratio: float,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if n < 2:
        raise ValueError("每个预测器至少需要 2 条样本。")

    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if n >= 10 and val_ratio > 0:
        val_size = max(1, int(round(n * val_ratio)))
    else:
        val_size = 0

    if val_size > 0:
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
    else:
        train_idx = indices
        val_idx = np.array([], dtype=np.int64)

    X_train = X[train_idx]
    Y_train = Y[train_idx]

    x_mean = X_train.mean(axis=0).astype(np.float32)
    x_std = safe_std(X_train)
    y_mean = Y_train.mean(axis=0).astype(np.float32)
    y_std = safe_std(Y_train)

    X_train_norm = (X_train - x_mean) / x_std
    Y_train_norm = (Y_train - y_mean) / y_std

    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(Y_train_norm, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)

    val_loader = None
    if len(val_idx) > 0:
        X_val = X[val_idx]
        Y_val = Y[val_idx]
        X_val_norm = (X_val - x_mean) / x_std
        Y_val_norm = (Y_val - y_mean) / y_std
        val_dataset = TensorDataset(
            torch.tensor(X_val_norm, dtype=torch.float32),
            torch.tensor(Y_val_norm, dtype=torch.float32),
        )
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    return train_loader, val_loader, x_mean, x_std, y_mean, y_std


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: str,
) -> Dict[str, float]:
    model.eval()
    X_norm = (X - x_mean) / x_std
    x_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
    pred_norm = model(x_tensor).cpu().numpy()
    pred = pred_norm * y_std + y_mean

    abs_err = np.abs(pred - Y)
    mae_latency = float(abs_err[:, 0].mean())
    mae_power = float(abs_err[:, 1].mean())

    rmse_latency = float(np.sqrt(((pred[:, 0] - Y[:, 0]) ** 2).mean()))
    rmse_power = float(np.sqrt(((pred[:, 1] - Y[:, 1]) ** 2).mean()))

    mape_latency = float((np.abs((pred[:, 0] - Y[:, 0]) / np.clip(Y[:, 0], 1e-6, None))).mean() * 100.0)
    mape_power = float((np.abs((pred[:, 1] - Y[:, 1]) / np.clip(Y[:, 1], 1e-6, None))).mean() * 100.0)

    return {
        "mae_latency_ms": mae_latency,
        "mae_power_W": mae_power,
        "rmse_latency_ms": rmse_latency,
        "rmse_power_W": rmse_power,
        "mape_latency_percent": mape_latency,
        "mape_power_percent": mape_power,
    }


def train_one_predictor(
    X: np.ndarray,
    Y: np.ndarray,
    model_name: str,
    device_name: str,
    train_device: str = "cuda",
    epochs: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    hidden_dims: Tuple[int, ...] = (64, 64, 32),
    dropout: float = 0.05,
    patience: int = 200,
    seed: int = 42,
) -> Dict:
    train_loader, val_loader, x_mean, x_std, y_mean, y_std = build_dataloaders(
        X=X,
        Y=Y,
        val_ratio=val_ratio,
        batch_size=batch_size,
        seed=seed,
    )

    model = LatencyPowerPredictor(hidden_dims=hidden_dims, dropout=dropout).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss()

    best_loss = math.inf
    best_state = None
    best_epoch = -1
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(train_device)
            yb = yb.to(train_device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(train_device)
                    yb = yb.to(train_device)
                    pred = model(xb)
                    val_losses.append(criterion(pred, yb).item())
            current_score = float(np.mean(val_losses))
        else:
            current_score = train_loss

        scheduler.step(current_score)

        if current_score < best_loss:
            best_loss = current_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"Training Failed: {model_name} / {device_name} 未得到有效模型参数。")

    model.load_state_dict(best_state)

    metrics = evaluate_metrics(
        model=model,
        X=X,
        Y=Y,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=train_device,
    )

    checkpoint = {
        "model_name": model_name,
        "device_name": device_name,
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "state_dict": model.cpu().state_dict(),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "metrics": metrics,
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "num_samples": int(len(X)),
        "feature_names": ["num_layer", "batch_size"],
        "target_names": ["batch_latency_ms", "avg_power_W"],
    }
    return checkpoint


# =========================
# Save / Load / Predict
# =========================

def save_checkpoint(checkpoint: Dict, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    model_name = sanitize_filename(checkpoint["model_name"])
    device_name = sanitize_filename(checkpoint["device_name"])
    save_path = os.path.join(save_dir, f"{model_name}-{device_name}.pt")
    torch.save(checkpoint, save_path)
    return save_path


def load_predictor(checkpoint_path: str, device: str = "cuda") -> PredictorBundle:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = LatencyPowerPredictor(
        hidden_dims=tuple(ckpt["hidden_dims"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return PredictorBundle(
        model=model,
        x_mean=np.asarray(ckpt["x_mean"], dtype=np.float32),
        x_std=np.asarray(ckpt["x_std"], dtype=np.float32),
        y_mean=np.asarray(ckpt["y_mean"], dtype=np.float32),
        y_std=np.asarray(ckpt["y_std"], dtype=np.float32),
        model_name=ckpt["model_name"],
        device_name=ckpt["device_name"],
    )


# =========================
# High-level API
# =========================

def train_all_predictors(
    json_path: str,
    save_dir: str,
    train_device: Optional[str] = None,
    epochs: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    hidden_dims: Tuple[int, ...] = (64, 64, 32),
    dropout: float = 0.05,
    patience: int = 200,
    seed: int = 42,
    only_model: Optional[str] = None,
    only_device: Optional[str] = None,
) -> List[Dict]:
    set_seed(seed)
    
    if not save_dir:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    if train_device is None:
        train_device = "cuda" if torch.cuda.is_available() else "cpu"

    grouped = parse_json_records(json_path)
    summaries: List[Dict] = []

    modelname = None

    for (model_name, device_name), data in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        if only_model is not None and model_name != only_model:
            continue
        if only_device is not None and device_name != only_device:
            continue

        X = data["X"]
        Y = data["Y"]

        checkpoint = train_one_predictor(
            X=X,
            Y=Y,
            model_name=model_name,
            device_name=device_name,
            train_device=train_device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            val_ratio=val_ratio,
            hidden_dims=hidden_dims,
            dropout=dropout,
            patience=patience,
            seed=seed,
        )
        ckpt_path = save_checkpoint(checkpoint, save_dir)

        summary = {
            "model": model_name,
            "device": device_name,
            "checkpoint": ckpt_path,
            **checkpoint["metrics"],
            "best_epoch": checkpoint["best_epoch"],
            "best_loss": checkpoint["best_loss"],
            "num_samples": checkpoint["num_samples"],
        }
        summaries.append(summary)

        print(
            f"[Done] {model_name} | {device_name} | samples={checkpoint['num_samples']} | "
            f"MAE(latency)={summary['mae_latency_ms']:.4f} ms | MAE(power)={summary['mae_power_W']:.4f} W | "
            f"saved={ckpt_path}"
        )
        modelname = model_name

    summary_path = os.path.join(save_dir, f"{modelname}-training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\n训练汇总已保存到: {summary_path}")
    return summaries


# =========================
# CLI
# =========================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train latency/power predictors for edge pipeline devices.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="从 JSON 数据训练并保存所有预测器")
    train_parser.add_argument("--json_path", type=str, required=True, help="训练数据 json 文件路径")
    train_parser.add_argument("--save_dir", type=str, default=None, help="权重保存目录")
    train_parser.add_argument("--train_device", type=str, default="cuda", help="cpu 或 cuda；默认自动选择")
    train_parser.add_argument("--epochs", type=int, default=2000)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--val_ratio", type=float, default=0.2)
    train_parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64, 32])
    train_parser.add_argument("--dropout", type=float, default=0.05)
    train_parser.add_argument("--patience", type=int, default=200)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--only_model", type=str, default=None, help="只训练指定模型名")
    train_parser.add_argument("--only_device", type=str, default=None, help="只训练指定设备名")

    pred_parser = subparsers.add_parser("predict", help="加载单个预测器并做预测")
    pred_parser.add_argument("--checkpoint", type=str, required=True, help="某个模型-设备预测器的 .pt 权重路径")
    pred_parser.add_argument("--num_layer", type=float, required=True, help="部署层数")
    pred_parser.add_argument("--batch_size", type=float, required=True, help="批大小")
    pred_parser.add_argument("--device", type=str, default="cuda", help="推理设备: cpu / cuda")

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "train":
        train_all_predictors(
            json_path=args.json_path,
            save_dir=args.save_dir,
            train_device=args.train_device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            patience=args.patience,
            seed=args.seed,
            only_model=args.only_model,
            only_device=args.only_device,
        )

    elif args.command == "predict":
        bundle = load_predictor(args.checkpoint, device=args.device)
        latency_ms, power_w = bundle.predict(
            num_layer=args.num_layer,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"model   : {bundle.model_name}")
        print(f"device  : {bundle.device_name}")
        print(f"layers  : {args.num_layer}")
        print(f"batch   : {args.batch_size}")
        print(f"latency : {latency_ms:.6f} ms")
        print(f"power   : {power_w:.6f} W")


if __name__ == "__main__":
    main()
