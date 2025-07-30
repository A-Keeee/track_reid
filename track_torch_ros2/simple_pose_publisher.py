#!/usr/bin/env python3
"""
简化版 ROS2 Pose Publisher for Track Torch
通过共享文件或其他方式获取三维坐标并发布到 ROS2 pose 话题
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header
import json
import os
import time

class SimpleTrackingPosePublisher(Node):
    """简化的 ROS2 节点，用于发布跟踪目标的位姿信息"""
    
    def __init__(self):
        super().__init__('simple_tracking_pose_publisher')
        
        # 创建 pose 发布器
        self.pose_publisher = self.create_publisher(
            PoseStamped, 
            '/tracking/target_pose', 
            10
        )
        
        # 参数配置
        self.declare_parameter('publish_rate', 10.0)  # 发布频率 Hz
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('coord_file', '/tmp/tracking_coords.json')
        self.declare_parameter('data_timeout', 2.0) # 数据有效时间（秒）

        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.coord_file = self.get_parameter('coord_file').get_parameter_value().string_value
        self.data_timeout = self.get_parameter('data_timeout').get_parameter_value().double_value
        
        # 定时器用于定期发布
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_pose)
        
        # 用于调试日志
        self._last_log_time = 0
        
        self.get_logger().info(f'Simple Tracking Pose Publisher 已启动')
        self.get_logger().info(f'发布频率: {self.publish_rate} Hz')
        self.get_logger().info(f'坐标系: {self.frame_id}')
        self.get_logger().info(f'坐标文件: {self.coord_file}')
        self.get_logger().info(f'数据超时时间: {self.data_timeout} s')

    def read_coordinates_from_file(self):
        """
        从文件中读取坐标数据。
        只有当文件存在且内部时间戳未超时时，才返回坐标。
        """
        try:
            # 检查文件是否存在，不存在则直接返回
            if not os.path.exists(self.coord_file):
                return None
            
            with open(self.coord_file, 'r') as f:
                data = json.load(f)
                
                # 检查数据中的时间戳是否过旧
                timestamp = data.get('timestamp', 0)
                if time.time() - timestamp > self.data_timeout:
                    # self.get_logger().warn(f'坐标数据已过时。') # 可选的警告信息
                    return None
                
                coords = (data.get('x', 0.0), data.get('y', 0.0), data.get('z', 0.0))
                return coords
                
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # self.get_logger().warn(f'读取或解析坐标文件失败: {e}')
            return None

    def publish_pose(self):
        """发布当前的位姿信息"""
        # 读取最新坐标
        coords = self.read_coordinates_from_file()
        
        # 如果坐标无效（文件不存在、数据过时、格式错误等），则不发布
        if coords is None:
            return
        
        x, y, z = coords
        
        
        # 创建 PoseStamped 消息
        pose_msg = PoseStamped()
        
        # 设置时间戳和坐标系
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.frame_id
        
        # 设置位置信息
        pose_msg.pose.position = Point()
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)
        
        # 设置方向信息（暂时设为单位四元数，表示无旋转）
        pose_msg.pose.orientation = Quaternion()
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        # 发布消息
        self.pose_publisher.publish(pose_msg)
        
        # 可选：打印调试信息，使用节流阀避免刷屏
        current_time = time.time()
        if current_time - self._last_log_time > 1.0:
            self.get_logger().info(f'发布位姿: x={x:.2f}, y={y:.2f}, z={z:.2f}')
            self._last_log_time = current_time


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = None # 预先声明
    try:
        node = SimpleTrackingPosePublisher()
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("收到中断信号，正在退出...")
    except Exception as e:
        print(f"节点运行出错: {e}")
    finally:
        # 清理资源
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()