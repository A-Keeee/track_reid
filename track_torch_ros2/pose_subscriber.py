#!/usr/bin/env python3
"""
ROS2 Pose 订阅器，用于测试和验证坐标发布
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class PoseSubscriber(Node):
    """简单的 pose 订阅器，用于测试"""
    
    def __init__(self):
        super().__init__('pose_subscriber')
        
        # 创建订阅器
        self.subscription = self.create_subscription(
            PoseStamped,
            '/tracking/target_pose',
            self.pose_callback,
            10
        )
        
        self.get_logger().info('🎯 Pose 订阅器已启动，监听 /tracking/target_pose')
        self.msg_count = 0
    
    def pose_callback(self, msg):
        """处理接收到的 pose 消息"""
        self.msg_count += 1
        
        # 提取坐标
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # 提取时间戳
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        
        # 打印信息
        if self.msg_count % 10 == 0:  # 每10条消息打印一次
            self.get_logger().info(
                f'📍 收到坐标 #{self.msg_count}: '
                f'x={x:.2f}, y={y:.2f}, z={z:.2f} '
                f'[{frame_id}]'
            )


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = PoseSubscriber()
        
        print("🚀 开始监听 ROS2 pose 话题...")
        print("📡 话题: /tracking/target_pose")
        print("🔄 消息类型: geometry_msgs/PoseStamped")
        print("按 Ctrl+C 停止监听")
        
        # 运行节点
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n🛑 订阅器已停止")
    except Exception as e:
        print(f"❌ 节点运行出错: {e}")
    finally:
        # 清理资源
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
