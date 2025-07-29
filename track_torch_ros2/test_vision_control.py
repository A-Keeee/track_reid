#!/usr/bin/env python3
"""
ROS2 视觉控制测试脚本
发布控制命令到 /start_vision 话题进行测试
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import time
import sys


class VisionControlTester(Node):
    """ROS2 视觉控制测试节点"""
    
    def __init__(self):
        super().__init__('vision_control_tester')
        
        # 创建发布器
        self.publisher = self.create_publisher(Int32, '/start_vision', 10)
        
        self.get_logger().info('🧪 视觉控制测试节点已启动')
        self.get_logger().info('📡 发布话题: /start_vision')
    
    def send_command(self, command):
        """发送控制命令"""
        msg = Int32()
        msg.data = command
        
        self.publisher.publish(msg)
        
        if command == 1:
            self.get_logger().info('🚀 已发送开启跟随命令')
        elif command == 0:
            self.get_logger().info('🛑 已发送关闭跟随命令')
        else:
            self.get_logger().warn(f'⚠️  发送了无效命令: {command}')


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python test_vision_control.py 1    # 开启跟随")
        print("  python test_vision_control.py 0    # 关闭跟随")
        print("  python test_vision_control.py auto # 自动测试模式")
        sys.exit(1)
    
    command_arg = sys.argv[1]
    
    try:
        node = VisionControlTester()
        
        if command_arg == "auto":
            # 自动测试模式
            print("🤖 自动测试模式")
            print("将每5秒切换一次开启/关闭状态，持续30秒")
            
            for i in range(6):  # 30秒，每5秒一次
                command = i % 2  # 0, 1, 0, 1, 0, 1
                node.send_command(command)
                
                if i < 5:  # 最后一次不等待
                    time.sleep(5)
        else:
            # 单次命令模式
            try:
                command = int(command_arg)
                if command not in [0, 1]:
                    raise ValueError("命令必须是 0 或 1")
                
                node.send_command(command)
                
            except ValueError as e:
                print(f"❌ 无效参数: {e}")
                sys.exit(1)
        
        print("✅ 命令发送完成")
        
    except KeyboardInterrupt:
        print("\n🛑 测试已停止")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
    finally:
        # 清理资源
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
