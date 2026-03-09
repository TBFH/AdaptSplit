import requests
import json
import time

import sys
import logging
from datetime import datetime
import os
from typing import Dict, Any, List

# Depricated Logger
def setup_logging(log_file: str):
    """
    初始化日志控制器，可将彩色日志输出至终端，同时保存至本地log文件
    
    :param log_file: 本地log文件路径
    :type log_file: str
    """
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 创建文件handler（保存到文件，不带颜色代码）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台handler（带颜色输出到终端）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台formatter带颜色
    class ColorFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[36m',     # 青色
            'INFO': '\033[32m',      # 绿色
            'WARNING': '\033[33m',   # 黄色
            'ERROR': '\033[31m',     # 红色
            'CRITICAL': '\033[1;31m' # 粗体红色
        }
        RESET = '\033[0m'
        
        def format(self, record):
            log_message = super().format(record)
            if record.levelname in self.COLORS:
                return f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
            return log_message
    
    console_formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 应用formatter
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class RainbowLogger:
    """日志记录器 - 支持多种颜色"""
    
    def __init__(self, log_dir: str = None):
        # 设置logging
        self.logger = logging.getLogger('RainbowLogger')
        self.logger.setLevel(logging.DEBUG)
        
        # 文件handler（保存所有级别）
        # 只有初始化时传入了路径，才认为要将日志保存至本地log文件，否则不保存日志至本地
        if log_dir:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_path = os.path.join(log_dir, f"{timestamp}.log")
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            # file_formatter = logging.Formatter(
            #     '%(asctime)s - %(message)s',
            #     datefmt='%Y-%m-%d %H:%M:%S'
            # )
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # 控制台handler（不显示DEBUG）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 自定义控制台格式化器
        class ConsoleFormatter(logging.Formatter):
            COLOR_MAP = {
                'INFO': '\033[39m',
                'WARNING': '\033[33m',
                'ERROR': '\033[31m',
                'CRITICAL': '\033[1;31m',
                'GREEN': '\033[32m',
                'BLUE': '\033[34m',
            }
            
            def format(self, record):
                message = super().format(record)
                color = self.COLOR_MAP.get(record.levelname, '')
                reset = '\033[0m'
                if color:
                    return f"{color}{message}{reset}"
                return message
        
        console_handler.setFormatter(ConsoleFormatter('%(message)s'))
        self.logger.addHandler(console_handler)
        
        # 添加自定义日志级别
        for idx, level_name in enumerate(['GREEN', 'BLUE']):
            level_num = logging.INFO + idx + 1  # 自定义级别值
            logging.addLevelName(level_num, level_name)
            setattr(self.logger, level_name.lower(), 
                    lambda msg, ln=level_name, lv=level_num: self.logger.log(lv, msg))


class JsonHelper:
    """JSON文件辅助类，支持初始化创建文件和字典追加到数组中"""

    def __init__(self, dir_path: str, file_name: str = None):
        """
        初始化JSON辅助类
        :param dir_path: JSON文件所在的目录路径
        :param file_name: JSON文件名，默认为"当前时间戳.json"
        """
        # 确保目录存在
        self.dir_path = dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # 拼接完整的文件路径
        if not file_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"{timestamp}.json"
        self.file_path = os.path.join(dir_path, file_name)
        
        # 确保文件存在，初始写入空数组[]
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as f:
                # 初始写入空数组，保证JSON格式合法
                json.dump([], f, ensure_ascii=False, indent=4)

    def append_dict(self, new_data: Dict[str, Any]) -> bool:
        """
        将字典作为新元素追加到JSON数组中
        :param new_data: 要追加的字典数据
        :return: 写入成功返回True，失败返回False
        """
        try:
            # 读取现有数组数据
            with open(self.file_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    # 确保读取到的是数组类型，否则重置为空数组
                    if not isinstance(existing_data, list):
                        existing_data = []
                except json.JSONDecodeError:
                    # 如果文件内容损坏，重置为空数组
                    existing_data = []
            
            # 将新字典追加到数组末尾
            existing_data.append(new_data)
            
            # 写入更新后的数组数据
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
            
            return True
        
        except Exception as e:
            print(f"写入JSON文件失败: {e}")
            return False

    def get_all_data(self) -> List[Dict[str, Any]]:
        """
        获取JSON文件中的所有数组数据
        :return: 列表格式的数据，每个元素是字典
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 确保返回的是列表类型
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
            return []

    def clear_data(self) -> bool:
        """
        清空JSON文件中的所有数据（重置为空数组）
        :return: 清空成功返回True，失败返回False
        """
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"清空JSON文件失败: {e}")
            return False


def mnserver_query_instant(query):
    '''
    调用已部署至集群的后端应用，其中调用了Grafana API接口查找当前时刻的监控数据，根据传入的PromQL语法字符串查找
    
    :参数 query: PromQL查找串
    '''

    base_url = 'http://219.222.20.79:31362/admin-api/ai/k8s-monitor/grafana/query'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "query": query,
        "type": 0
    }
    # 发送http请求
    response = requests.post(
        base_url,
        headers=headers,
        json=data,
        timeout=5
    )
    # 处理响应
    if response.status_code == 200:
        res = response.json()
        return json.loads(res["data"])
    elif response.status_code == 404:
        print("资源未找到")
        return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")


def mnserver_query_range(query, start, end, step):
    '''
    调用已部署至集群的后端应用，其中调用了Grafana API接口查找过去一段时间内的所有监控数据，根据传入的PromQL语法字符串查找
    '''

    base_url = 'http://219.222.20.79:31362/admin-api/ai/k8s-monitor/grafana/query'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "query": query,
        "type": 1,
        "start": start,
        "end": end,
        "step": step
    }
    # 发送http请求
    response = requests.post(
        base_url,
        headers=headers,
        json=data,
        timeout=5
    )
    # 处理响应
    if response.status_code == 200:
        res = response.json()
        return json.loads(res["data"])
    elif response.status_code == 404:
        print("资源未找到")
        return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")


def grafana_query_instant(query):
    '''
    直接调用Grafana API接口查找当前时刻的监控数据，根据传入的PromQL语法字符串查找
    
    参数 query: PromQL查找串
    '''
    grafana_key = os.environ.get('GRAFANA_API_KEY')
    base_url = 'http://219.222.20.79:32411/api/datasources/proxy/1/api/v1/query'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Bearer {grafana_key}'
    }
    data = { "query": query }
    # 发送http请求
    response = requests.post(
        base_url,
        headers=headers,
        data=data,
        timeout=5
    )
    # 处理响应
    if response.status_code == 200:
        res = response.json()
        return res
    elif response.status_code == 404:
        print("资源未找到")
        return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")


def grafana_query_range(query, start, end, step):
    '''
    直接调用Grafana API接口查找过去一段时间内的所有监控数据，根据传入的PromQL语法字符串查找
    '''
    grafana_key = os.environ.get('GRAFANA_API_KEY')
    base_url = 'http://219.222.20.79:32411/api/datasources/proxy/1/api/v1/query_range'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Bearer {grafana_key}'
    }
    data = {
        "query": query,
        "type": 1,
        "start": start,
        "end": end,
        "step": step
    }
    # 发送http请求
    response = requests.post(
        base_url,
        headers=headers,
        data=data,
        timeout=5
    )
    # 处理响应
    if response.status_code == 200:
        res = response.json()
        return res
    elif response.status_code == 404:
        print("资源未找到")
        return None
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")


def sort_power_data(*fetched_list):
    res = {}
    for fetched in fetched_list:
        if fetched['status'] != 'success':
            print('Fetched Not Success')
            return None
        for node in fetched['data']['result']:
            node_name = node['metric']['instance']
            if 'jetson' in node_name:
                val = [[int(time), float(power)/1000] for time, power in node['values']]
                res[node_name] = val
            elif 'pc' in node_name:
                val = [[int(time), float(power)] for time, power in node['values']]
                res[f"{node_name}-{node['metric']['gpu']}"] = val
    
    return res


def sort_gpu_data(fetched):
    res = {}
    if fetched['status'] != 'success':
        print('Fetched Not Success')
        return None
    for node in fetched['data']['result']:
        _, ram_in_bytes = node['value']
        res[node['metric']['instance']] = int(ram_in_bytes)
    return res


def get_avg_power(power_data, avg):
    avgs = {}
    for device, data in power_data.items():
        powers = [float(power) for _, power in data]
        if avg:
            avgs[device] = sum(powers)/len(powers)
        else:
            avgs[device] = powers
    return avgs


def power_plot(devices, power_data, save_path):
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    data = power_data

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 四个不同的颜色
    # colors = plt.cm.tab20(np.linspace(0, 1, 20))
    # colors = plt.cm.Set3(np.linspace(0, 1, 16))

    # 2. 定义16种线型和标记组合
    line_styles = ['-', '--', '-.', ':'] * 4  # 重复使用以确保足够
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_', '1']  # 16种不同的标记

    # 创建图形
    plt.figure(figsize=(8, 5))

    # 3. 控制图例显示顺序（这里按设备名称排序，您可以自定义顺序）
    # 例如：按设备内存大小排序
    legend_order = {device:idx for idx, device in enumerate(devices)}

    # 按指定顺序排序设备
    sorted_devices = sorted(data.items(), key=lambda x: legend_order.get(x[0], 99))

    # 收集所有功耗值用于确定纵轴范围
    all_power_values = []

    # 绘制每条曲线
    for idx, (device_name, device_data) in enumerate(sorted_devices):
        # 提取时间戳和功耗值
        timestamps = [item[0] for item in device_data]
        power_values = [item[1] for item in device_data]
        
        # 收集所有功耗值
        all_power_values.extend(power_values)
        
        # 从0开始的时间（以第一个时间戳为基准）
        start_time = min(timestamps)
        relative_times = [ts - start_time for ts in timestamps]
        
        # 绘制曲线
        plt.plot(relative_times, power_values, 
                color=colors[idx % len(colors)],
                linestyle=line_styles[0 % len(line_styles)],
                marker=markers[0 % len(markers)],
                linewidth=2,
                markersize=4,
                label=device_name)

    # 设置图形属性
    plt.title('Power Curve', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Power (W)', fontsize=14)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. 自适应纵轴显示范围（基于数据范围，上下留10%的边距）
    if all_power_values:
        min_power = min(all_power_values)
        max_power = max(all_power_values)
        power_range = max_power - min_power
        
        # 计算纵轴范围，上下留出10%的边距
        y_margin = power_range * 0.1
        y_lower = max(0, min_power - y_margin)  # 确保不低于0
        y_upper = max_power + y_margin
        
        plt.ylim(y_lower, y_upper)
        
        # 在纵轴上添加一条参考线（0功率线，如果需要）
        if y_lower <= 0:
            plt.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # 设置x轴范围（从0开始，稍微留点边距）
    all_timestamps = [item[0] for sublist in data.values() for item in sublist]
    if all_timestamps:
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        time_range = end_time - start_time
        plt.xlim(-time_range*0.05, time_range*1.05)  # 左边留5%，右边留5%

    # 添加图例
    plt.legend(fontsize=10, loc='best', framealpha=0.95, ncol=2)  # ncol=2可以让图例分两列显示

    # 添加数值标签（可选）
    # for idx, (device_name, device_data) in enumerate(sorted_devices):
    #     timestamps = [item[0] for item in device_data]
    #     power_values = [item[1] for item in device_data]
    #     start_time = min(timestamps)
    #     relative_times = [ts - start_time for ts in timestamps]
        
    #     # 在每个数据点上添加数值标签
    #     for x, y in zip(relative_times, power_values):
    #         plt.text(x, y, f'{y}', fontsize=9, ha='center', va='bottom')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    # plt.show()

    # 保存图形（可选）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_path, f"power-plot_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"power plot saved at {save_path}")


def profile_power(devices, duration, step, plot_dir=None, avg=True):
    devices_jetson=[]
    devices_pc=[]
    for d in devices:
        if 'jetson' in d:
            devices_jetson.append(d)
        elif 'pc' in d:
            devices_pc.append(d)
        else:
            raise ValueError(f"Power Profile Error: Unknown Type of Device {d}")
    # 接口调用参数
    jetson_query = f'integrated_power_mW{"{"}instance=~"({"|".join(devices_jetson)})", statistic="power"{"}"}'
    pc_query = f'DCGM_FI_DEV_POWER_USAGE{"{"}instance=~"({"|".join(devices_pc)})", gpu=~"(0|1)"{"}"}'
    end = int(time.time())
    start = int(end - duration)
    step = step    # 表示每多少秒获取一次数据
    # 获取功耗数据
    jetson_fetched = grafana_query_range(jetson_query, start, end, step)
    pc_fetched = grafana_query_range(pc_query, start, end, step)
    # 格式化功耗数据
    power_data = sort_power_data(jetson_fetched, pc_fetched)
    # 保存功耗曲线图
    if plot_dir:
        power_plot(devices, power_data, plot_dir)
    # 计算平均功耗
    power_data = get_avg_power(power_data, avg)
    
    return power_data


def profile_vram(devices):
    # 接口调用参数
    instances = "|".join(devices)
    free_vram_query = f'ram_kB{"{"}instance=~"({instances})", statistic="free"{"}"}'
    cached_vram_query = f'ram_kB{"{"}instance=~"({instances})", statistic="cached"{"}"}'
    # 获取显存数据并格式化
    fetched = mnserver_query_instant(free_vram_query)
    free_vram = sort_gpu_data(fetched)
    fetched = mnserver_query_instant(cached_vram_query)
    cached_vram = sort_gpu_data(fetched)
    # 返回数据
    return {key: free_vram[key] + cached_vram[key] for key in free_vram}


if __name__ == '__main__':
    # Testing
    # devices = ['jetson-64g-4', 'jetson-16g-3', 'jetson-16g-7', 'jetson-8g-1']
    devices = ['jetson-64g-4', 'pc-4090', 'pc-3090', 'jetson-8g-1']
    duration = 300      # 要获取过去的多少秒的数据
    # profile_power(devices, duration, 1, "/home/austin/stats/uneven_partition/")
    # profile_power(devices, duration, 50, "/home/austin/stats/pre_benchmarking")
    # print(profile_power(devices, duration, 50))
