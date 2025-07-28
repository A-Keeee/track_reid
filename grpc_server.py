#!/usr/bin/env python3
"""
gRPC服务器模块 - 用于与APP进行局域网通信
支持接收跟随指令和发送跟踪状态
"""

import grpc
from concurrent import futures
import threading
import time
import queue
import tracking_pb2
import tracking_pb2_grpc
from typing import Optional


class TrackingServiceImpl(tracking_pb2_grpc.TrackingServiceServicer):
    """跟踪服务实现"""
    
    def __init__(self):
        self.follow_enabled = False
        self.target_id = None
        self.tracking_start_time = None
        self.current_coordinate = None
        self.coordinate_queue = queue.Queue(maxsize=100)
        self.status_lock = threading.Lock()
        
    def SendCoordinates(self, request_iterator, context):
        """接收坐标数据流"""
        try:
            for coordinate_data in request_iterator:
                # 更新当前坐标
                self.current_coordinate = coordinate_data
                
                # 添加到队列供订阅者获取
                try:
                    self.coordinate_queue.put_nowait(coordinate_data)
                except queue.Full:
                    # 队列满时移除最旧的数据
                    try:
                        self.coordinate_queue.get_nowait()
                        self.coordinate_queue.put_nowait(coordinate_data)
                    except queue.Empty:
                        pass
                
                # 打印接收到的坐标信息
                print(f"📍 接收坐标: ({coordinate_data.x:.2f}, {coordinate_data.y:.2f}, {coordinate_data.z:.2f})")
                    
            return tracking_pb2.Response(success=True, message="坐标数据接收完成")
            
        except Exception as e:
            print(f"接收坐标数据错误: {e}")
            return tracking_pb2.Response(success=False, message=f"错误: {e}")
    


    def Active(self, request, context):
        """接收APP的开启跟随指令"""
        with self.status_lock:
            self.follow_enabled = True
            self.target_id = request.target_id if request.target_id > 0 else None
            
            if self.tracking_start_time is None:
                self.tracking_start_time = time.time()
                print(f"🎯 收到开启跟随指令: 开始跟随目标 ID {self.target_id}")
                # 返回10秒倒计时，给用户准备时间
                countdown_time = 10
            else:
                # 已经在跟随状态，更新目标ID
                countdown_time = 0
                print(f"🔄 更新跟随目标 ID: {self.target_id}")
                
        return tracking_pb2.ActiveResponse(time=countdown_time)

    def Disactive(self, request, context):
        """接收APP的关闭跟随指令"""
        with self.status_lock:
            self.follow_enabled = False
            self.tracking_start_time = None
            print("⏹️ 收到关闭跟随指令: 停止跟随")
                
        return tracking_pb2.Empty()



    def SetFollowCommand(self, request, context):
        """接收APP的跟随指令"""
        with self.status_lock:
            self.follow_enabled = request.start_follow
            self.target_id = request.target_id if request.target_id > 0 else None
            
            if self.follow_enabled and self.tracking_start_time is None:
                self.tracking_start_time = time.time()
                print(f"收到跟随指令: 开始跟随目标 ID {self.target_id}")
            elif not self.follow_enabled:
                self.tracking_start_time = None
                print("收到跟随指令: 停止跟随")
                
        return tracking_pb2.Response(
            success=True,
            message=f"跟随指令已更新: {'启用' if self.follow_enabled else '禁用'}"
        )
    
    def GetTrackingStatus(self, request, context):
        """发送当前跟踪状态给APP"""
        with self.status_lock:
            tracking_time = 0.0
            if self.follow_enabled and self.tracking_start_time:
                tracking_time = time.time() - self.tracking_start_time
                
            return tracking_pb2.TrackingStatus(
                is_active=self.follow_enabled,
                tracking_time=tracking_time,
                target_id=self.target_id or 0,
                timestamp=time.time()
            )
    
    def GetCurrentCoordinates(self, request, context):
        """获取当前坐标"""
        if self.current_coordinate:
            return self.current_coordinate
        else:
            return tracking_pb2.CoordinateData(
                x=0.0, y=0.0, z=0.0
            )
    
    def SubscribeCoordinates(self, request, context):
        """订阅坐标更新流"""
        print("客户端订阅坐标更新流")
        try:
            while True:
                try:
                    # 等待新的坐标数据，超时5秒
                    coordinate = self.coordinate_queue.get(timeout=5.0)
                    yield coordinate
                except queue.Empty:
                    # 发送心跳数据
                    yield tracking_pb2.CoordinateData(
                        x=0.0, y=0.0, z=0.0
                    )
        except Exception as e:
            print(f"坐标流订阅错误: {e}")
    
    def update_coordinate(self, coordinate_data):
        """更新坐标数据（从主程序调用）"""
        self.current_coordinate = coordinate_data
        try:
            self.coordinate_queue.put_nowait(coordinate_data)
        except queue.Full:
            # 队列满时移除最旧的数据
            try:
                self.coordinate_queue.get_nowait()
                self.coordinate_queue.put_nowait(coordinate_data)
            except queue.Empty:
                pass
    
    def is_follow_enabled(self):
        """检查是否启用跟随"""
        with self.status_lock:
            return self.follow_enabled
    
    def get_target_id(self):
        """获取跟随目标ID"""
        with self.status_lock:
            return self.target_id
    
    def get_tracking_time(self):
        """获取跟踪时间"""
        with self.status_lock:
            if self.follow_enabled and self.tracking_start_time:
                return time.time() - self.tracking_start_time
            return 0.0
    
    def IsActived(self, request, context):
        """获取当前是否在跟随状态"""
        with self.status_lock:
            return tracking_pb2.IsActivedResponse(
                is_active=self.follow_enabled
            )
    
    def SetAutoTracking(self, request, context):
        """设置自动跟踪配置"""
        # 这里可以根据需要实现自动跟踪逻辑
        # 目前暂时返回成功响应
        return tracking_pb2.Response(
            success=True,
            message=f"自动跟踪配置已更新: {'启用' if request.enabled else '禁用'}"
        )
    
    def ToggleAutoTracking(self, request, context):
        """切换自动跟踪模式"""
        # 这里可以根据需要实现自动跟踪切换逻辑
        # 目前暂时返回成功响应
        return tracking_pb2.Response(
            success=True,
            message="自动跟踪模式已切换"
        )




class GRPCServer:
    """gRPC服务器管理类"""
    
    def __init__(self, port=50051):
        self.port = port
        self.server = None
        self.service_impl = TrackingServiceImpl()
        self.running = False
        
    def start(self):
        """启动gRPC服务器"""
        try:
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            tracking_pb2_grpc.add_TrackingServiceServicer_to_server(
                self.service_impl, self.server
            )
            
            listen_addr = f'[::]:{self.port}'
            self.server.add_insecure_port(listen_addr)
            self.server.start()
            self.running = True
            
            print(f"gRPC服务器已启动，监听端口: {self.port}")
            print(f"服务地址: {listen_addr}")
            return True
            
        except Exception as e:
            print(f"gRPC服务器启动失败: {e}")
            return False
    
    def stop(self):
        """停止gRPC服务器"""
        if self.server and self.running:
            print("正在停止gRPC服务器...")
            self.server.stop(grace=5)
            self.running = False
            print("gRPC服务器已停止")
    
    def update_target_coordinate(self, target_state):
        """更新目标坐标（从主程序调用）"""
        if target_state and target_state.active:
            coordinate_data = tracking_pb2.CoordinateData(
                x=target_state.world_position[0] if target_state.world_position else 0.0,
                y=target_state.world_position[1] if target_state.world_position else 0.0,
                z=target_state.world_position[2] if target_state.world_position else 0.0
            )
            self.service_impl.update_coordinate(coordinate_data)
    
    def is_follow_enabled(self):
        """检查是否启用跟随"""
        return self.service_impl.is_follow_enabled()
    
    def get_target_id(self):
        """获取跟随目标ID"""
        return self.service_impl.get_target_id()
    
    def get_tracking_time(self):
        """获取跟踪时间"""
        return self.service_impl.get_tracking_time()


if __name__ == "__main__":
    # 测试服务器
    server = GRPCServer()
    if server.start():
        try:
            print("按 Ctrl+C 停止服务器")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n收到停止信号")
        finally:
            server.stop()
