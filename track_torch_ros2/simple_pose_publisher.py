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
import threading
import pickle


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
        
        self.publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.coord_file = self.get_parameter('coord_file').value
        
        # 定时器用于定期发布
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_pose)
        
        # 最新的坐标数据
        self.latest_coords = None
        self.last_file_mtime = 0
        
        self.get_logger().info(f'Simple Tracking Pose Publisher 已启动')
        self.get_logger().info(f'发布频率: {self.publish_rate} Hz')
        self.get_logger().info(f'坐标系: {self.frame_id}')
        self.get_logger().info(f'坐标文件: {self.coord_file}')
    
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
                coords = (data.get('x', 0.0), data.get('y', 0.0), data.get('z', 0.0))
                timestamp = data.get('timestamp', 0)
                
                # 检查数据是否太旧（超过2秒）
                if time.time() - timestamp > 2.0:
                    return None
                
                return coords
                
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # self.get_logger().warn(f'读取坐标文件失败: {e}')
            return None
    
    def publish_pose(self):
        """发布当前的位姿信息"""
        # 读取最新坐标
        coords = self.read_coordinates_from_file()
        if coords is None:
            return
        
        x, y, z = coords
        
        # 检查坐标有效性
        if x == 0.0 and y == 0.0 and z == 0.0:
            return  # 跳过无效坐标
        
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
        
        # 更新缓存
        self.latest_coords = coords
        
        # 可选：打印调试信息
        if hasattr(self, '_last_log_time'):
            if time.time() - self._last_log_time > 1.0:  # 每秒最多打印一次
                self.get_logger().info(f'发布位姿: x={x:.2f}, y={y:.2f}, z={z:.2f}')
                self._last_log_time = time.time()
        else:
            self._last_log_time = time.time()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = SimpleTrackingPosePublisher()
        
        # 运行节点
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("收到中断信号，正在退出...")
    except Exception as e:
        print(f"节点运行出错: {e}")
    finally:
        # 清理资源
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
