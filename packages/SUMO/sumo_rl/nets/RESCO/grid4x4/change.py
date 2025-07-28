#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML处理脚本：提取depart在1500-3000范围内的车辆，复制10倍并随机分布时间
"""

import xml.etree.ElementTree as ET
import random
import numpy as np
from typing import List, Tuple


def parse_xml_and_extract_vehicles(xml_file: str, min_depart: float = 1500.0, max_depart: float = 3000.0) -> List[Tuple[str, str, str]]:
    """
    解析XML文件并提取指定depart时间范围内的车辆
    
    Args:
        xml_file: XML文件路径
        min_depart: 最小depart时间
        max_depart: 最大depart时间
        
    Returns:
        包含(vehicle_id, depart_time, route_edges)的列表
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    vehicles = []
    
    for vehicle in root.findall('vehicle'):
        vehicle_id = vehicle.get('id')
        depart_attr = vehicle.get('depart')
        
        # 检查depart属性是否存在
        if depart_attr is None:
            continue
            
        depart_time = float(depart_attr)
        
        # 检查depart时间是否在指定范围内
        if min_depart <= depart_time <= max_depart:
            route = vehicle.find('route')
            if route is not None:
                route_edges = route.get('edges')
                if route_edges is not None:
                    vehicles.append((vehicle_id, str(depart_time), route_edges))
    
    return vehicles


def generate_uniform_random_times(num_vehicles: int, min_time: float = 1500.0, max_time: float = 3000.0) -> List[float]:
    """
    生成均匀分布的随机时间
    
    Args:
        num_vehicles: 车辆数量
        min_time: 最小时间
        max_time: 最大时间
        
    Returns:
        随机时间列表
    """
    # 使用numpy生成均匀分布的随机时间
    times = np.random.uniform(min_time, max_time, num_vehicles)
    # 排序以确保时间递增
    times.sort()
    return times.tolist()


def generate_extended_random_times(num_vehicles: int, min_time: float = 0.0, max_time: float = 15000.0) -> List[float]:
    """
    生成0-15000范围内的均匀分布随机时间
    
    Args:
        num_vehicles: 车辆数量
        min_time: 最小时间 (0)
        max_time: 最大时间 (15000)
        
    Returns:
        随机时间列表
    """
    # 使用numpy生成均匀分布的随机时间
    times = np.random.uniform(min_time, max_time, num_vehicles)
    # 排序以确保时间递增
    times.sort()
    return times.tolist()


def create_new_vehicle_elements(vehicles: List[Tuple[str, str, str]], 
                              new_times: List[float], 
                              base_id: int = 10000) -> List[ET.Element]:
    """
    创建新的车辆元素
    
    Args:
        vehicles: 原始车辆信息列表
        new_times: 新的depart时间列表
        base_id: 新车辆ID的基础值
        
    Returns:
        新的车辆元素列表
    """
    new_vehicles = []
    
    for i, ((original_id, original_depart, route_edges), new_time) in enumerate(zip(vehicles, new_times)):
        # 创建新的vehicle元素
        vehicle_elem = ET.Element('vehicle')
        # 使用更安全的ID生成方式，避免冲突
        new_id = f"{original_id}_copy_{i+1}"
        vehicle_elem.set('id', new_id)
        vehicle_elem.set('depart', f"{new_time:.2f}")
        
        # 创建route子元素
        route_elem = ET.SubElement(vehicle_elem, 'route')
        route_elem.set('edges', route_edges)
        
        new_vehicles.append(vehicle_elem)
    
    return new_vehicles


def process_xml_file(input_file: str, output_file: str, 
                    min_depart: float = 1500.0, max_depart: float = 3000.0,
                    multiplier: int = 10, base_id: int = 10000):
    """
    处理XML文件的主要函数
    
    Args:
        input_file: 输入XML文件路径
        output_file: 输出XML文件路径
        min_depart: 最小depart时间
        max_depart: 最大depart时间
        multiplier: 复制倍数
        base_id: 新车辆ID的基础值
    """
    print(f"正在处理文件: {input_file}")
    
    # 解析XML并提取车辆
    vehicles = parse_xml_and_extract_vehicles(input_file, min_depart, max_depart)
    print(f"找到 {len(vehicles)} 个车辆在时间范围 {min_depart}-{max_depart} 内")
    
    if not vehicles:
        print("没有找到符合条件的车辆")
        return
    
    # 复制车辆信息
    all_vehicles = vehicles * multiplier
    print(f"复制 {multiplier} 倍后，总共有 {len(all_vehicles)} 个车辆")
    
    # 生成新的随机时间（在0-15000范围内）
    new_times = generate_extended_random_times(len(all_vehicles), 0.0, 15000.0)
    print(f"生成了 {len(new_times)} 个随机时间，时间范围: 0-15000")
    
    # 创建新的车辆元素
    new_vehicle_elements = create_new_vehicle_elements(all_vehicles, new_times, base_id)
    print(f"创建了 {len(new_vehicle_elements)} 个新的车辆元素")
    
    # 读取原始XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # 移除原有的车辆元素（可选，这里我们保留原有车辆）
    # 或者我们可以只保留不在指定时间范围内的车辆
    
    # 添加新的车辆元素
    for vehicle_elem in new_vehicle_elements:
        root.append(vehicle_elem)
    
    # 创建格式化的XML字符串
    xml_str = ET.tostring(root, encoding='unicode')
    
    # 添加换行和缩进
    import re
    xml_str = re.sub(r'><', '>\n    <', xml_str)
    xml_str = re.sub(r'<vehicle', '    <vehicle', xml_str)
    xml_str = re.sub(r'<route', '        <route', xml_str)
    xml_str = re.sub(r'</route>', '        </route>', xml_str)
    xml_str = re.sub(r'</vehicle>', '    </vehicle>', xml_str)
    
    # 保存修改后的XML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml_str)
    print(f"已保存到: {output_file}")
    
    # 输出统计信息
    print(f"\n统计信息:")
    print(f"- 原始符合条件的车辆数量: {len(vehicles)}")
    print(f"- 复制倍数: {multiplier}")
    print(f"- 新生成的车辆数量: {len(new_vehicle_elements)}")
    print(f"- 时间范围: {min_depart} - {max_depart}")
    print(f"- 新车辆ID范围: {base_id} - {base_id + len(new_vehicle_elements) - 1}")


def main():
    """主函数"""
    input_file = "grid4x4_1.rou.xml"
    output_file = "grid4x4_1.rou.xml"  # 直接替换原文件
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    try:
        process_xml_file(
            input_file=input_file,
            output_file=output_file,
            min_depart=1500.0,
            max_depart=3000.0,
            multiplier=10,
            base_id=10000
        )
        print("\n处理完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    main()
