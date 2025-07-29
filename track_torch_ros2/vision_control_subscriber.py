#!/usr/bin/env python3
"""
ROS2 视觉控制订阅器
订阅 /start_vision 话题，接收到1时开启跟随，接收到0时关闭跟随
通过文件方式与 track_torch.py 通信，不影响 gRPC 逻辑
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import json
import os
import time
import threading


class VisionControlSubscriber(Node):
    """ROS2 视觉控制订阅器"""
    
    def __init__(self):
        super().__init__('vision_control_subscriber')
        
        # 创建订阅器
        self.subscription = self.create_subscription(
            Int32,
            '/start_vision',
            self.vision_command_callback,
            10
        )
        
        # 参数配置
        self.declare_parameter('control_file', '/tmp/vision_control.json')
        self.declare_parameter('status_file', '/tmp/vision_status.json')
        
        self.control_file = self.get_parameter('control_file').value
        self.status_file = self.get_parameter('status_file').value
        
        # 当前状态
        self.current_command = 0
        self.last_command_time = 0
        
        # 创建状态发布定时器
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info('🎯 视觉控制订阅器已启动')
        self.get_logger().info(f'   - 订阅话题: /start_vision')
        self.get_logger().info(f'   - 控制文件: {self.control_file}')
        self.get_logger().info(f'   - 状态文件: {self.status_file}')
        
        # 初始化控制文件
        self.write_control_command(0)
    
    def vision_command_callback(self, msg):
        """处理接收到的视觉控制命令"""
        command = msg.data
        
        if command in [0, 1]:
            self.current_command = command
            self.last_command_time = time.time()
            
            # 写入控制文件
            self.write_control_command(command)
            
            if command == 1:
                self.get_logger().info('🚀 收到开启跟随命令')
            else:
                self.get_logger().info('🛑 收到关闭跟随命令')
        else:
            self.get_logger().warn(f'⚠️  无效命令: {command} (仅支持 0 或 1)')
    
    def write_control_command(self, command):
        """写入控制命令到文件"""
        try:
            control_data = {
                'command': int(command),
                'timestamp': time.time(),
                'source': 'ros2_vision_control'
            }
            
            with open(self.control_file, 'w') as f:
                json.dump(control_data, f)
                
            self.get_logger().debug(f'已写入控制命令: {command}')
            
        except Exception as e:
            self.get_logger().error(f'❌ 写入控制文件失败: {e}')
    
    def publish_status(self):
        """发布当前状态到状态文件"""
        try:
            status_data = {
                'current_command': self.current_command,
                'last_command_time': self.last_command_time,
                'node_active': True,
                'timestamp': time.time()
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f)
                
        except Exception as e:
            self.get_logger().error(f'❌ 写入状态文件失败: {e}')
    
    def destroy_node(self):
        """节点销毁时清理资源"""
        # 清理控制文件
        try:
            if os.path.exists(self.control_file):
                os.remove(self.control_file)
            if os.path.exists(self.status_file):
                os.remove(self.status_file)
            self.get_logger().info('🧹 已清理控制文件')
        except Exception as e:
            self.get_logger().error(f'❌ 清理文件失败: {e}')
        
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = VisionControlSubscriber()
        
        print("🚀 开始监听视觉控制命令...")
        print("📡 订阅话题: /start_vision")
        print("🔄 消息类型: std_msgs/Int32")
        print("📝 命令说明:")
        print("   - 发送 1: 开启跟随")
        print("   - 发送 0: 关闭跟随")
        print("按 Ctrl+C 停止监听")
        
        # 运行节点
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n🛑 视觉控制订阅器已停止")
    except Exception as e:
        print(f"❌ 节点运行出错: {e}")
    finally:
        # 清理资源
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
