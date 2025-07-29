#!/usr/bin/env python3
"""
测试 ROS2 集成的脚本
生成模拟坐标数据来测试 ROS2 发布器
"""

import json
import time
import math
import os


def generate_test_coordinates():
    """生成测试用的三维坐标数据"""
    
    print("🧪 开始生成测试坐标数据...")
    print("📁 坐标文件: /tmp/tracking_coords.json")
    print("按 Ctrl+C 停止测试")
    
    t = 0
    try:
        while True:
            # 生成圆形轨迹的坐标
            x = 2.0 + 0.5 * math.cos(t * 0.1)  # 前方 2 米，来回摆动
            y = 0.3 * math.sin(t * 0.1)        # 左右摆动
            z = 1.5 + 0.2 * math.sin(t * 0.05) # 上下摆动
            
            # 创建坐标数据
            coord_data = {
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'timestamp': time.time()
            }
            
            # 写入文件
            with open('/tmp/tracking_coords.json', 'w') as f:
                json.dump(coord_data, f)
            
            print(f"📍 坐标: x={x:.2f}, y={y:.2f}, z={z:.2f}")
            
            t += 1
            time.sleep(0.1)  # 10 Hz 更新
            
    except KeyboardInterrupt:
        print("\n🛑 测试已停止")
        
        # 清理测试文件
        if os.path.exists('/tmp/tracking_coords.json'):
            os.remove('/tmp/tracking_coords.json')
            print("🧹 已清理测试文件")


if __name__ == '__main__':
    generate_test_coordinates()
