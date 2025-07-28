#!/usr/bin/env python3
"""
gRPC客户端测试脚本 - 模拟APP与跟踪系统的通信
支持跟随指令和状态查询
"""

import grpc
import time
import threading
import sys
import math

# 导入生成的gRPC模块
try:
    import tracking_pb2
    import tracking_pb2_grpc
except ImportError:
    print("错误: 未找到gRPC模块，请先运行: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. tracking.proto")
    sys.exit(1)


class TrackingClient:
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        
    def connect(self):
        """连接到gRPC服务器"""
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            # 测试连接
            grpc.channel_ready_future(self.channel).result(timeout=10)
            self.stub = tracking_pb2_grpc.TrackingServiceStub(self.channel)
            print(f"✅ 成功连接到服务器: {self.server_address}")
            return True
            
        except grpc.RpcError as e:
            print(f"❌ gRPC连接失败: {e}")
            return False
        except Exception as e:
            print(f"❌ 连接异常: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.channel:
            self.channel.close()
            print("已断开连接")
    
    def send_follow_command(self, start_follow=True, target_id=1):
        """发送跟随指令（旧版本兼容）"""
        try:
            request = tracking_pb2.FollowCommand(
                start_follow=start_follow,
                target_id=target_id
            )
            response = self.stub.SetFollowCommand(request)
            action = "开始跟随" if start_follow else "停止跟随"
            print(f"📡 {action}指令发送成功: {response.message}")
            return response.success
        except Exception as e:
            print(f"❌ 发送跟随指令失败: {e}")
            return False

    def send_active_command(self, target_id=1):
        """发送 Active 指令（开启跟随）"""
        try:
            request = tracking_pb2.ActiveRequest(
                target_id=target_id
            )
            response = self.stub.Active(request)
            if response.time > 0:
                print(f"📡 开始跟随指令发送成功! 倒计时: {response.time}秒")
            else:
                print(f"📡 开始跟随指令发送成功! 响应时间: {response.time}秒")
            return True
        except Exception as e:
            print(f"❌ 发送 Active 指令失败: {e}")
            return False
    
    def send_disactive_command(self):
        """发送 Disactive 指令（停止跟随）"""
        try:
            request = tracking_pb2.Empty()
            response = self.stub.Disactive(request)
            print(f"📡 停止跟随指令发送成功!")
            return True
        except Exception as e:
            print(f"❌ 发送 Disactive 指令失败: {e}")
            return False
    
    def get_tracking_status(self):
        """获取跟踪状态"""
        try:
            request = tracking_pb2.Empty()
            response = self.stub.GetTrackingStatus(request)
            status = "🟢 活跃" if response.is_active else "🔴 非活跃"
            print(f"📊 跟踪状态: {status}")
            print(f"   跟踪时间: {response.tracking_time:.1f}秒")
            print(f"   目标ID: {response.target_id}")
            print(f"   时间戳: {time.strftime('%H:%M:%S', time.localtime(response.timestamp))}")
            return response
        except Exception as e:
            print(f"❌ 获取跟踪状态失败: {e}")
            return None
    
    def get_current_coordinates(self):
        """获取当前坐标"""
        try:
            request = tracking_pb2.Empty()
            response = self.stub.GetCurrentCoordinates(request)
            print(f"📍 当前坐标:")
            print(f"   位置: X={response.x:.2f}m, Y={response.y:.2f}m, Z={response.z:.2f}m")
            return response
        except Exception as e:
            print(f"❌ 获取坐标失败: {e}")
            return None
    
    def subscribe_coordinates_stream(self, duration=30):
        """订阅坐标流"""
        def stream_worker():
            try:
                request = tracking_pb2.Empty()
                print(f"📡 开始订阅坐标流 ({duration}秒)...")
                response_stream = self.stub.SubscribeCoordinates(request)
                
                start_time = time.time()
                count = 0
                for response in response_stream:
                    if time.time() - start_time > duration:
                        break
                    
                    count += 1
                    current_time = time.strftime('%H:%M:%S', time.localtime())
                    
                    print(f"[{current_time}] 📍 坐标: "
                          f"({response.x:.2f}, {response.y:.2f}, {response.z:.2f})")
                        
                print(f"📡 坐标流订阅结束，共接收 {count} 条消息")
                
            except Exception as e:
                print(f"❌ 坐标流订阅失败: {e}")
        
        # 在后台线程中运行
        stream_thread = threading.Thread(target=stream_worker)
        stream_thread.daemon = True
        stream_thread.start()
        return stream_thread
    
    def check_is_actived(self):
        """检查是否在跟随状态"""
        try:
            request = tracking_pb2.IsActivedRequest()
            response = self.stub.IsActived(request)
            status = "🟢 活跃跟随中" if response.is_active else "🔴 未在跟随"
            print(f"🔍 当前跟随状态: {status}")
            return response.is_active
        except Exception as e:
            print(f"❌ 检查跟随状态失败: {e}")
            return None


def main():
    print("=== gRPC客户端测试程序 ===")
    print("🔗 模拟APP与跟踪系统的通信")
    
    # 创建客户端
    server_address = input("输入服务器地址 (默认 localhost:50051): ").strip() or "localhost:50051"
    client = TrackingClient(server_address)
    
    if not client.connect():
        print("❌ 无法连接到服务器，请确保跟踪程序正在运行")
        return
    
    try:
        while True:
            print("\n📋 请选择操作:")
            print("1. 📍 开始跟随目标")
            print("2.  选择目标")
            print("3. ⏹️  停止跟随")
            # print("4. 🎯 开始跟随目标 (Active新版)")
            # print("5. ⏸️  停止跟随 (Active新版)")
            print("4. 🔍 检查跟随状态 (IsActived)")
            print("5. 📊 获取详细跟踪状态")
            print("6. 🗺️  获取当前坐标")
            print("7. 📡 订阅坐标流 (30秒)")
            print("8. 🚪 退出")

            choice = input("\n输入选择 (1-8): ").strip()

            # if choice == '1':
            #     target_id = int(input("输入目标ID (默认1): ") or "1")
            #     client.send_follow_command(start_follow=True, target_id=target_id)
                
            # elif choice == '2':
            #     client.send_follow_command(start_follow=False, target_id=0)
                
            if choice == '1':
                # target_id = int(input("输入目标ID (默认1): ") or "1")
                client.send_active_command(target_id=1)
            
            elif choice == '2':
                target_id = int(input("输入目标ID (默认1): ") or "1")
                client.send_active_command(target_id=target_id)

            elif choice == '3':
                client.send_disactive_command()

            elif choice == '4':
                client.check_is_actived()

            elif choice == '5':
                client.get_tracking_status()

            elif choice == '6':
                client.get_current_coordinates()

            elif choice == '7':
                stream_thread = client.subscribe_coordinates_stream(30)
                print("📡 坐标流已在后台运行，请等待...")
                time.sleep(2)  # 让用户看到一些输出

            elif choice == '8':
                break
                
            else:
                print("❌ 无效选择，请重试")
                
    except KeyboardInterrupt:
        print("\n⏹️ 程序被中断")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
