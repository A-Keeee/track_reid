#!/usr/bin/env python3
"""
坐标发布器 - 不依赖ROS2
通过多种方式发布三维坐标数据：UDP、TCP、文件、终端输出等
"""

import json
import os
import time
import socket
import threading
import argparse
from datetime import datetime


class CoordinatePublisher:
    """多种方式发布坐标数据的发布器"""
    
    def __init__(self, coord_file='/tmp/tracking_coords.json', 
                 udp_port=9999, tcp_port=9998, publish_rate=10.0):
        self.coord_file = coord_file
        self.udp_port = udp_port
        self.tcp_port = tcp_port
        self.publish_rate = publish_rate
        self.running = False
        
        # 最新的坐标数据
        self.latest_coords = None
        self.last_file_mtime = 0
        
        # 网络套接字
        self.udp_socket = None
        self.tcp_socket = None
        self.tcp_clients = []
        
        print(f"📡 坐标发布器已初始化")
        print(f"   - 坐标文件: {self.coord_file}")
        print(f"   - UDP端口: {self.udp_port}")
        print(f"   - TCP端口: {self.tcp_port}")
        print(f"   - 发布频率: {self.publish_rate} Hz")
    
    def setup_network(self):
        """设置网络套接字"""
        try:
            # UDP 套接字
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            print("✅ UDP 广播套接字已设置")
            
            # TCP 服务器套接字
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_socket.bind(('0.0.0.0', self.tcp_port))
            self.tcp_socket.listen(5)
            self.tcp_socket.settimeout(1.0)  # 非阻塞
            print("✅ TCP 服务器已启动")
            
            # 启动 TCP 客户端接受线程
            threading.Thread(target=self.accept_tcp_clients, daemon=True).start()
            
        except Exception as e:
            print(f"❌ 网络设置失败: {e}")
    
    def accept_tcp_clients(self):
        """接受 TCP 客户端连接"""
        while self.running:
            try:
                client_socket, addr = self.tcp_socket.accept()
                self.tcp_clients.append(client_socket)
                print(f"🔗 TCP 客户端已连接: {addr}")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ TCP 客户端接受失败: {e}")
                break
    
    def read_coordinates_from_file(self):
        """从文件中读取坐标数据"""
        try:
            if not os.path.exists(self.coord_file):
                return None
            
            # 检查文件是否有更新
            current_mtime = os.path.getmtime(self.coord_file)
            if current_mtime <= self.last_file_mtime:
                return self.latest_coords
            
            self.last_file_mtime = current_mtime
            
            with open(self.coord_file, 'r') as f:
                data = json.load(f)
                coords = {
                    'x': data.get('x', 0.0),
                    'y': data.get('y', 0.0),
                    'z': data.get('z', 0.0),
                    'timestamp': data.get('timestamp', 0)
                }
                
                # 检查数据是否太旧（超过2秒）
                if time.time() - coords['timestamp'] > 2.0:
                    return None
                
                return coords
                
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            return None
    
    def publish_coordinates(self, coords):
        """发布坐标到各种输出"""
        if coords is None:
            return
        
        x, y, z = coords['x'], coords['y'], coords['z']
        
        # 检查坐标有效性
        if x == 0.0 and y == 0.0 and z == 0.0:
            return
        
        # 创建消息
        timestamp = datetime.now().isoformat()
        message = {
            'type': 'pose',
            'frame_id': 'camera_link',
            'timestamp': timestamp,
            'position': {
                'x': float(x),
                'y': float(y),
                'z': float(z)
            },
            'orientation': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'w': 1.0
            }
        }
        
        message_json = json.dumps(message)
        
        # 1. UDP 广播
        if self.udp_socket:
            try:
                self.udp_socket.sendto(
                    message_json.encode('utf-8'), 
                    ('<broadcast>', self.udp_port)
                )
            except Exception as e:
                print(f"❌ UDP 发送失败: {e}")
        
        # 2. TCP 发送
        if self.tcp_clients:
            for client in self.tcp_clients[:]:  # 复制列表避免修改
                try:
                    client.send((message_json + '\n').encode('utf-8'))
                except Exception as e:
                    self.tcp_clients.remove(client)
                    client.close()
        
        # 3. 终端输出（每秒最多一次）
        if hasattr(self, '_last_log_time'):
            if time.time() - self._last_log_time > 1.0:
                print(f"📍 [{timestamp}] 坐标: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                self._last_log_time = time.time()
        else:
            self._last_log_time = time.time()
    
    def start(self):
        """启动发布器"""
        self.running = True
        self.setup_network()
        
        print("🚀 坐标发布器已启动")
        print("   - UDP 广播地址: <broadcast>:9999")
        print("   - TCP 服务器: 0.0.0.0:9998")
        print("   - 按 Ctrl+C 停止")
        
        try:
            while self.running:
                coords = self.read_coordinates_from_file()
                if coords:
                    self.publish_coordinates(coords)
                    self.latest_coords = coords
                
                time.sleep(1.0 / self.publish_rate)
                
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号")
        finally:
            self.stop()
    
    def stop(self):
        """停止发布器"""
        self.running = False
        
        # 关闭网络连接
        if self.udp_socket:
            self.udp_socket.close()
        
        if self.tcp_socket:
            self.tcp_socket.close()
        
        for client in self.tcp_clients:
            client.close()
        
        print("✅ 坐标发布器已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='坐标发布器 - 多种方式发布三维坐标')
    parser.add_argument('--coord-file', default='/tmp/tracking_coords.json', 
                       help='坐标文件路径')
    parser.add_argument('--udp-port', type=int, default=9999, 
                       help='UDP 广播端口')
    parser.add_argument('--tcp-port', type=int, default=9998, 
                       help='TCP 服务器端口')
    parser.add_argument('--rate', type=float, default=10.0, 
                       help='发布频率 (Hz)')
    
    args = parser.parse_args()
    
    # 创建并启动发布器
    publisher = CoordinatePublisher(
        coord_file=args.coord_file,
        udp_port=args.udp_port,
        tcp_port=args.tcp_port,
        publish_rate=args.rate
    )
    
    publisher.start()


if __name__ == '__main__':
    main()
