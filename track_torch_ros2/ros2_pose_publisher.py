#!/usr/bin/env python3
"""
ROS2 Pose Publisher for Track Torch
发布从 track_torch.py 接收到的三维坐标到 ROS2 pose 话题
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header
import threading
import queue
import time
import grpc
import tracking_pb2
import tracking_pb2_grpc


class TrackingPosePublisher(Node):
    """ROS2 节点，用于发布跟踪目标的位姿信息"""
    
    def __init__(self):
        super().__init__('tracking_pose_publisher')
        
        # 创建 pose 发布器
        self.pose_publisher = self.create_publisher(
            PoseStamped, 
            '/tracking/target_pose', 
            10
        )
        
        # 参数配置
        self.declare_parameter('grpc_server', 'localhost:50051')
        self.declare_parameter('publish_rate', 30.0)  # 发布频率 Hz
        self.declare_parameter('frame_id', 'camera_link')
        
        self.grpc_server = self.get_parameter('grpc_server').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        
        # gRPC 客户端
        self.grpc_client = TrackingGRPCSubscriber(self.grpc_server)
        self.coordinate_queue = queue.Queue(maxsize=50)
        
        # 定时器用于定期发布
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_pose)
        
        # 最新的坐标数据
        self.latest_coords = None
        self.last_update_time = time.time()
        
        self.get_logger().info(f'Tracking Pose Publisher 已启动')
        self.get_logger().info(f'gRPC 服务器: {self.grpc_server}')
        self.get_logger().info(f'发布频率: {self.publish_rate} Hz')
        self.get_logger().info(f'坐标系: {self.frame_id}')
        
        # 启动 gRPC 连接
        self.start_grpc_client()
    
    def start_grpc_client(self):
        """启动 gRPC 客户端连接"""
        def grpc_worker():
            if self.grpc_client.connect():
                self.grpc_client.start_coordinate_subscription(self.coordinate_queue)
                # 监听坐标更新
                while rclpy.ok():
                    try:
                        coord_data = self.coordinate_queue.get(timeout=1.0)
                        self.latest_coords = (coord_data.x, coord_data.y, coord_data.z)
                        self.last_update_time = time.time()
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.get_logger().error(f'接收坐标数据时出错: {e}')
                        break
            else:
                self.get_logger().error('无法连接到 gRPC 服务器')
        
        self.grpc_thread = threading.Thread(target=grpc_worker, daemon=True)
        self.grpc_thread.start()
    
    def publish_pose(self):
        """发布当前的位姿信息"""
        current_time = time.time()
        
        # 检查是否有有效的坐标数据
        if self.latest_coords is None:
            return
        
        # 检查数据是否过期（超过2秒没有更新）
        if current_time - self.last_update_time > 2.0:
            # self.get_logger().warn('坐标数据过期，跳过发布')
            return
        
        x, y, z = self.latest_coords
        
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
        
        # 可选：打印调试信息
        if hasattr(self, '_last_log_time'):
            if current_time - self._last_log_time > 1.0:  # 每秒最多打印一次
                self.get_logger().info(f'发布位姿: x={x:.2f}, y={y:.2f}, z={z:.2f}')
                self._last_log_time = current_time
        else:
            self._last_log_time = current_time
    
    def destroy_node(self):
        """节点销毁时清理资源"""
        if hasattr(self, 'grpc_client'):
            self.grpc_client.disconnect()
        super().destroy_node()


class TrackingGRPCSubscriber:
    """gRPC 客户端，用于接收跟踪坐标"""
    
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.connected = False
    
    def connect(self):
        """连接到 gRPC 服务器"""
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            grpc.channel_ready_future(self.channel).result(timeout=5)
            self.stub = tracking_pb2_grpc.TrackingServiceStub(self.channel)
            self.connected = True
            print(f"✅ gRPC 订阅客户端连接成功: {self.server_address}")
            return True
        except Exception as e:
            print(f"❌ gRPC 连接失败: {e}")
            self.connected = False
            return False
    
    def start_coordinate_subscription(self, coordinate_queue):
        """开始订阅坐标流"""
        if not self.connected:
            return
        
        def coordinate_listener():
            try:
                # 发送空请求以开始接收坐标流
                empty_request = tracking_pb2.Empty()
                coordinate_stream = self.stub.SendCoordinates(iter([empty_request]))
                
                for coordinate_data in coordinate_stream:
                    if coordinate_queue.full():
                        coordinate_queue.get_nowait()  # 移除旧数据
                    coordinate_queue.put_nowait(coordinate_data)
            except grpc.RpcError as e:
                print(f"❌ 坐标订阅失败: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"❌ 坐标监听异常: {e}")
        
        self.listen_thread = threading.Thread(target=coordinate_listener, daemon=True)
        self.listen_thread.start()
    
    def disconnect(self):
        """断开 gRPC 连接"""
        if self.channel:
            self.channel.close()
        self.connected = False
        print("gRPC 订阅客户端已断开连接")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = TrackingPosePublisher()
        
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
