# REID+姿态+前10s无重复特征+gRPC通信
# 
# 性能优化版本 - 单目标跟踪系统
# 
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import time
import math
import depthai as dai
import random
import heapq
import sys
import os
import pickle
import threading
import queue
import grpc
import argparse

# --- [修改] ReID 相关导入 ---
try:
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import ToTensor
    from PIL import Image
    # 导入新的ReID库组件
    from reid.config import cfg as reidCfg
    from reid.modeling import build_model
    from reid.data.transforms import build_transforms
    print("PyTorch 和 ReID 库导入成功")
except ImportError as e:
    print(f"警告: 导入失败 ({e})，ReID功能将受限")
    print("请确保已正确安装 PyTorch 和 fast-reid 库")
    torch = None

# 导入生成的gRPC模块
try:
    import tracking_pb2
    import tracking_pb2_grpc
except ImportError:
    print("警告: 未找到gRPC模块，gRPC通信功能将被禁用")
    print("请运行: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. tracking.proto")
    tracking_pb2 = None
    tracking_pb2_grpc = None

# 全局变量：特征存储路径
FEATURE_STORAGE_PATH = "target_features.pkl"

# 全局变量：ReID处理器
reid_handler = None

# 确保清理旧特征文件
if os.path.exists(FEATURE_STORAGE_PATH):
    os.remove(FEATURE_STORAGE_PATH)
    print(f"已清理旧特征文件: {FEATURE_STORAGE_PATH}")

# --- [新增] 基于 reid 库的特征处理器 ---
class ReIDHandler:
    def __init__(self, model_path): # <-- 修改：移除 config_file 参数
        if torch is None:
            self.model = None
            print("错误: PyTorch未安装，无法初始化ReID处理器")
            return

        try:
            # --- [修改] 使用导入的 reidCfg, 不再从文件加载 ---
            reidCfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            # DEVICE_ID在defaults.py中是字符串类型，保持一致
            reidCfg.MODEL.DEVICE_ID = '0'
            
            # 使用 reidCfg 中的配置构建模型, num_classes 参考 reid_depthai_oak.py
            self.model = build_model(reidCfg, num_classes=1501) 
            self.model.load_param(model_path) # 直接使用传入的权重路径
            self.model.to(reidCfg.MODEL.DEVICE).eval()
            
            self.transforms = build_transforms(reidCfg) # <-- 修改：移除 is_train=False 参数
            self.device = reidCfg.MODEL.DEVICE
            self.dist_thresh = 1.15 # 欧氏距离阈值
            print(f"ReID模型已成功加载到 {self.device}")

        except Exception as e:
            self.model = None
            print(f"初始化ReID处理器时出错: {e}")

    def extract_features(self, image_bgr):
        """ 使用新的ReID模型提取特征 """
        if self.model:
            try:
                # 预处理: BGR -> RGB -> PIL -> Tensor
                pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                input_tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)

                # 推理
                with torch.no_grad():
                    features = self.model(input_tensor)
                
                # 后处理：L2归一化
                features = F.normalize(features, dim=1, p=2)
                return features.cpu() # 返回Tensor，便于后续计算
            except Exception as e:
                print(f"ReID特征提取失败: {e}")
                return self.fallback_feature_extractor(image_bgr)
        else:
            return self.fallback_feature_extractor(image_bgr)
    
    def compute_distance(self, feature1_tensor, feature2_tensor):
        """ 计算两个特征向量之间的欧氏距离 """
        distmat = torch.pow(feature1_tensor, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(feature2_tensor, 2).sum(dim=1, keepdim=True).t()
        # 使用新的、基于关键字参数的签名来调用 addmm_
        distmat.addmm_(feature1_tensor, feature2_tensor.t(), beta=1, alpha=-2)
        return distmat.squeeze().item()

    def fallback_feature_extractor(self, image):
        """ 备用特征提取器（颜色直方图） """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        # 返回一个伪Tensor以便后续处理
        return torch.from_numpy(hist.flatten()).unsqueeze(0)


# 姿态特征提取与相似度计算（来自track_pose.py）
def extract_pose_features(keypoints):
    """提取详细的姿态特征用于匹配"""
    if keypoints is None or len(keypoints) < 17:
        return None
    
    kpts = keypoints.copy()
    valid_mask = (kpts[:, 0] > 0) & (kpts[:, 1] > 0)
    valid_kpts = kpts[valid_mask]
    
    if len(valid_kpts) < 5:
        return None
    
    left_shoulder, right_shoulder = 5, 6
    left_hip, right_hip = 11, 12
    
    center_points = []
    if valid_mask[left_shoulder] and valid_mask[right_shoulder]:
        center_points.extend([kpts[left_shoulder], kpts[right_shoulder]])
    if valid_mask[left_hip] and valid_mask[right_hip]:
        center_points.extend([kpts[left_hip], kpts[right_hip]])
    
    center = np.mean(center_points, axis=0) if len(center_points) > 0 else np.mean(valid_kpts, axis=0)
    
    distances = np.linalg.norm(valid_kpts - center, axis=1)
    scale = np.max(distances) if len(distances) > 0 and np.max(distances) > 0 else 1.0
    
    features = []
    for i in range(17):
        if valid_mask[i]:
            features.extend(((kpts[i] - center) / scale).tolist())
        else:
            features.extend([0.0, 0.0])
            
    return np.array(features)

def pose_similarity(feat1, feat2):
    """计算姿态特征相似度"""
    if feat1 is None or feat2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    cos_sim = np.dot(feat1, feat2) / (norm1 * norm2)
    return max(0.0, min(1.0, cos_sim))

def draw_pose_keypoints(img, keypoints, color=(0, 255, 255)):
    """绘制姿态关键点和骨架连接"""
    if keypoints is None or len(keypoints) < 17:
        return img
    
    # COCO格式17个关键点的连接关系
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # 绘制关键点
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)
    
    # 绘制骨架连接
    for connection in skeleton:
        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 转换为0-based索引
        if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and 
            keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and
            keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0):
            pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
            pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
            cv2.line(img, pt1, pt2, color, 2)
    
    return img

# gRPC客户端类
class TrackingGRPCClient:
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.connected = False
        self.last_coordinate_time = 0
        self.coordinate_interval = 0.1  # 坐标发送间隔（秒）
        self.coordinate_queue = queue.Queue(maxsize=100)
        self.stream_thread = None
        self.streaming = False
        
    def connect(self):
        """连接到gRPC服务器"""
        if tracking_pb2 is None or tracking_pb2_grpc is None:
            print("gRPC模块未导入，跳过连接")
            return False
            
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            # 测试连接
            grpc.channel_ready_future(self.channel).result(timeout=5)
            self.stub = tracking_pb2_grpc.TrackingServiceStub(self.channel)
            self.connected = True
            print(f"✅ gRPC客户端连接成功: {self.server_address}")
            
            # 启动坐标流传输
            self.start_coordinate_stream()
            return True
            
        except grpc.RpcError as e:
            print(f"❌ gRPC RPC错误: {e.code()}: {e.details()}")
            self.connected = False
            return False
        except Exception as e:
            print(f"❌ gRPC连接异常: {type(e).__name__}: {str(e)}")
            self.connected = False
            return False
    
    def start_coordinate_stream(self):
        """启动坐标流传输"""
        if not self.connected:
            return
            
        def coordinate_generator():
            """坐标数据生成器"""
            while self.streaming:
                try:
                    coordinate = self.coordinate_queue.get(timeout=1.0)
                    yield coordinate
                except queue.Empty:
                    # 发送心跳（只包含x, y, z）
                    yield tracking_pb2.CoordinateData(
                        x=0.0, y=0.0, z=0.0
                    )
        
        def stream_worker():
            """流传输工作线程"""
            try:
                self.streaming = True
                response = self.stub.SendCoordinates(coordinate_generator())
                print(f"坐标流传输结果: {response.message}")
            except Exception as e:
                print(f"❌ 坐标流传输失败: {e}")
            finally:
                self.streaming = False
        
        self.stream_thread = threading.Thread(target=stream_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        print("📡 坐标流传输已启动")
    
    def disconnect(self):
        """断开连接"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
            
        if self.channel:
            try:
                self.channel.close()
                self.connected = False
                print("gRPC客户端已断开连接")
            except:
                pass
    
    def send_target_coordinates(self, target_state):
        """发送目标坐标到服务器"""
        if not self.connected or tracking_pb2 is None or not self.streaming:
            return False
            
        current_time = time.time()
        if current_time - self.last_coordinate_time < self.coordinate_interval:
            return True  # 跳过发送，避免过于频繁
            
        try:
            if target_state and target_state.active and target_state.world_position:
                X, Y, Z = target_state.world_position
                
                # 创建坐标消息（只包含x, y, z）
                coordinate_msg = tracking_pb2.CoordinateData(
                    x=float(X),
                    y=float(Y), 
                    z=float(Z)
                )
                
                # 将坐标添加到队列
                try:
                    self.coordinate_queue.put_nowait(coordinate_msg)
                    self.last_coordinate_time = current_time
                    return True
                except queue.Full:
                    # 队列满时移除最旧的数据
                    try:
                        self.coordinate_queue.get_nowait()
                        self.coordinate_queue.put_nowait(coordinate_msg)
                        self.last_coordinate_time = current_time
                        return True
                    except queue.Empty:
                        pass
                
            else:
                # 发送无活跃目标的心跳（只包含x, y, z）
                coordinate_msg = tracking_pb2.CoordinateData(
                    x=0.0, y=0.0, z=0.0
                )
                
                try:
                    self.coordinate_queue.put_nowait(coordinate_msg)
                    self.last_coordinate_time = current_time
                    return True
                except queue.Full:
                    return True  # 心跳数据可以丢弃
                
        except Exception as e:
            print(f"❌ 发送坐标失败: {e}")
            return False
    
    def send_tracking_status(self, is_active, target_id=0, tracking_time=0.0):
        """发送跟踪状态"""
        if not self.connected or tracking_pb2 is None:
            return False
            
        try:
            status_msg = tracking_pb2.TrackingStatus(
                is_active=is_active,
                target_id=target_id,
                tracking_time=tracking_time,
                timestamp=time.time()
            )
            
            # 这里可以发送状态到服务器
            # response = self.stub.SendTrackingStatus(status_msg)
            return True
            
        except Exception as e:
            print(f"❌ 发送跟踪状态失败: {e}")
            return False
    
    def get_follow_commands(self):
        """获取跟随指令（如果有的话）"""
        if not self.connected or tracking_pb2 is None:
            return None
            
        try:
            # 从服务器获取跟踪状态指令
            request = tracking_pb2.Empty()
            response = self.stub.GetTrackingStatus(request)
            
            # 转换为命令格式
            if response.is_active and response.target_id > 0:
                return [{'action': 'follow', 'target_id': response.target_id}]
            elif not response.is_active:
                return [{'action': 'stop_follow'}]
            else:
                return None
            
        except Exception as e:
            # 静默处理错误，避免过多日志
            return None


# 初始化gRPC客户端
grpc_client = TrackingGRPCClient()


# 特征存储管理器
class FeatureStorage:
    def __init__(self, file_path=FEATURE_STORAGE_PATH):
        self.file_path = file_path
        self.features = {}
        # 不加载旧特征，确保每次运行从零开始

    def save_features(self):
        """保存特征到文件"""
        try:
            with open(self.file_path, 'wb') as f:
                pickle.dump(self.features, f)
            print(f"已保存{len(self.features)}个目标的特征到{self.file_path}")
        except Exception as e:
            print(f"保存特征失败: {e}")

    def add_feature(self, target_id, feature):
        """极简单目标模式：只保留唯一目标的特征"""
        self.features.clear()
        self.features[target_id] = [feature]

    def get_features(self, target_id):
        """获取目标的所有特征"""
        return self.features.get(target_id, [])

    def find_best_match(self, candidate_feature, threshold=0.6):
        """在所有保存的特征中寻找最佳匹配"""
        best_match_id = None
        max_similarity = 0.0

        for target_id, features in self.features.items():
            for feature in features:
                try:
                    similarity = np.dot(feature.flatten(), candidate_feature.flatten()) / (
                            np.linalg.norm(feature) * np.linalg.norm(candidate_feature) + 1e-10)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_id = target_id
                except:
                    continue

        if max_similarity > threshold:
            return best_match_id, max_similarity
        return None, max_similarity

    def clear_all_features(self):
        """清理所有特征"""
        self.features.clear()
        print("🧹 已清理所有特征数据")
        
        # 同时删除特征文件
        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
                print(f"🗑️ 已删除特征文件: {self.file_path}")
            except Exception as e:
                print(f"删除特征文件失败: {e}")

    def reset_features(self):
        """重置特征存储（别名方法）"""
        self.clear_all_features()


# 初始化特征存储器
feature_storage = FeatureStorage()


# 初始化OAK相机管道
def create_pipeline():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    # 使用新的API替换已弃用的设置
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(45)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    # 使用新的API替换已弃用的设置
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    # 使用新的API替换已弃用的设置
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    return pipeline


# 相机连接管理器
class CameraManager:
    def __init__(self, max_retries=5, retry_delay=3):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.device = None
        self.pipeline = None

    def connect_camera(self):
        """尝试连接相机，失败时重试"""
        for attempt in range(self.max_retries):
            try:
                print(f"尝试连接OAK相机... (第 {attempt + 1}/{self.max_retries} 次)")
                
                # 创建管道
                self.pipeline = create_pipeline()
                
                # 尝试连接设备
                self.device = dai.Device(self.pipeline)
                print("OAK相机连接成功！")
                return True
                
            except Exception as e:
                print(f"相机连接失败 (第 {attempt + 1} 次): {e}")
                
                # 清理资源
                if self.device:
                    try:
                        self.device.close()
                    except:
                        pass
                    self.device = None
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries - 1:
                    print(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    print("所有重试尝试都失败了")
                    return False
        
        return False

    def get_device(self):
        """获取设备实例"""
        return self.device

    def close(self):
        """关闭设备连接"""
        if self.device:
            try:
                self.device.close()
            except:
                pass
            self.device = None


# 目标状态类（增强鲁棒性和预测能力）
class TargetState:
    def __init__(self, target_id):
        self.target_id = target_id
        self.position = None
        self.velocity = (0, 0)
        self.size = None
        self.confidence = 0.0
        self.timestamp = time.time()
        self.last_seen = time.time()
        self.trajectory = deque(maxlen=50)
        self.world_position = None
        self.distance = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.reid_features = deque(maxlen=20)  # 增加特征保留数量
        self.stable_features = deque(maxlen=5)  # 稳定特征集合
        self.initial_feature = None  # 初始特征
        self.lock_strength = 1.0
        self.stability = 0.0
        self.consecutive_frames = 0
        self.last_update_time = time.time()
        self.last_output_time = time.time()
        self.lost_frame_count = 0
        self.kalman = self.init_kalman_filter()
        self.active = True
        self.color = (0, 0, 255)  # 目标颜色
        self.pose_landmarks = None  # 存储姿态关键点
        self.pose_score = 0.0  # 姿态置信度
        self.pose_visibility = 0.0  # 姿态可见度
        self.full_body_feature = None  # 全身特征
        self.last_feature_save_time = 0  # 上次保存特征的时间
        # 添加姿态特征相关属性（来自track_pose.py）
        self.pose_features = deque(maxlen=10)  # 姿态特征历史
        self.stable_pose_features = deque(maxlen=3)  # 稳定姿态特征
        self.initial_pose_feature = None  # 初始姿态特征

    def init_kalman_filter(self):
        kalman = cv2.KalmanFilter(6, 2)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)
        kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        kalman.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
        return kalman

    def update_reid_features(self, frame, box, is_full_body=False):
        # 限制特征更新频率
        current_time = time.time()
        if current_time - self.last_feature_save_time < 0.5:  # 每0.5秒更新一次
            return
            
        x1, y1, x2, y2 = map(int, box)
        if y1 >= y2 or x1 >= x2 or y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            return

        expand = 10
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(frame.shape[1], x2 + expand)
        y2 = min(frame.shape[0], y2 + expand)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if x1 >= x2 or y1 >= y2:
            return

        target_roi = frame[y1:y2, x1:x2]
        if target_roi.size == 0:
            return

        # --- [修改] 使用新的ReID处理器提取特征 ---
        try:
            # reid_handler.extract_features 现在返回一个Tensor
            feature_tensor = reid_handler.extract_features(target_roi)
            if feature_tensor is None: return

            # 保存初始特征
            if self.initial_feature is None:
                self.initial_feature = feature_tensor
                self.stable_features.append(feature_tensor)
                if is_full_body:
                    self.full_body_feature = feature_tensor
                    # 注意：feature_storage现在存储Tensor
                    feature_storage.add_feature(self.target_id, feature_tensor)

            # 只在目标稳定时更新特征
            if self.stability > 0.6:
                self.reid_features.append(feature_tensor)

                # 当稳定性高时更新稳定特征
                if self.stability > 0.8 and len(self.stable_features) < 5:
                    self.stable_features.append(feature_tensor)

                    if is_full_body and self.full_body_feature is None:
                        self.full_body_feature = feature_tensor
                        feature_storage.add_feature(self.target_id, feature_tensor)
                        
            self.last_feature_save_time = time.time()
        except Exception as e:
            print(f"ReID特征更新失败 (TargetState): {e}")

    def compare_signature(self, frame, box):
        if not self.reid_features and not self.stable_features and self.full_body_feature is None:
            return 0.0

        x1, y1, x2, y2 = map(int, box)
        if y1 >= y2 or x1 >= x2 or y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            return 0.0

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        candidate_roi = frame[y1:y2, x1:x2]
        if candidate_roi.size == 0:
            return 0.0

        # --- [修改] 使用欧氏距离进行比对，并转换为相似度分数 ---
        try:
            candidate_feature = reid_handler.extract_features(candidate_roi)
            if candidate_feature is None: return 0.0

            min_dist = float('inf')

            # 收集所有待比较的特征
            features_to_compare = []
            if self.full_body_feature is not None:
                features_to_compare.append(self.full_body_feature)
            if self.initial_feature is not None:
                features_to_compare.append(self.initial_feature)
            features_to_compare.extend(self.stable_features)
            features_to_compare.extend(self.reid_features)

            if not features_to_compare:
                return 0.0
            
            # 将所有库特征合并为一个Tensor进行批量计算
            gallery_features = torch.cat(features_to_compare, dim=0)

            # 计算候选者与库中所有特征的距离
            distmat = torch.pow(candidate_feature, 2).sum(dim=1, keepdim=True).expand(1, len(gallery_features)) + \
                      torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(len(gallery_features), 1).t()
            
            # 使用新的、基于关键字参数的签名来调用 addmm_
            distmat.addmm_(candidate_feature, gallery_features.t(), beta=1, alpha=-2)
            
            # 找到最小距离
            min_dist = torch.min(distmat).item()

        except Exception as e:
            print(f"特征比较失败: {e}")
            return 0.0

        # 将距离转换为相似度分数 (0到1之间，越高越好)
        similarity = max(0.0, 1.0 - (min_dist / reid_handler.dist_thresh))
        
        return similarity

    def update_pose(self, keypoints, visibility, score):
        """更新目标的姿态信息（基于YOLO关键点）"""
        self.pose_landmarks = keypoints  # 现在存储YOLO关键点格式 (17, 2)
        self.pose_score = score
        self.pose_visibility = visibility

        # 如果姿态置信度高，增加锁定强度
        if score > 0.7:
            self.lock_strength = min(1.0, self.lock_strength + 0.05)
        elif score < 0.3:
            self.lock_strength = max(0.3, self.lock_strength - 0.05)

    def update_pose_features(self, keypoints):
        """更新姿态特征（来自track_pose.py）"""
        if keypoints is None or len(keypoints) < 17:
            return
            
        pose_feat = extract_pose_features(keypoints)
        if pose_feat is not None:
            self.pose_features.append(pose_feat)
            if self.initial_pose_feature is None:
                self.initial_pose_feature = pose_feat
            if self.stability > 0.8 and len(self.stable_pose_features) < 3:
                self.stable_pose_features.append(pose_feat)
            
            # 计算姿态可见度和置信度
            valid_kpts = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
            visibility = len(valid_kpts) / 17.0 if len(keypoints) >= 17 else 0.0
            score = visibility  # 简单的置信度估计
            
            # 更新姿态信息
            self.update_pose(keypoints, visibility, score)

    def compare_pose(self, keypoints):
        """比较姿态相似度（来自track_pose.py）"""
        if not any([self.initial_pose_feature is not None, self.stable_pose_features]):
             return 0.0
        
        candidate_feature = extract_pose_features(keypoints)
        if candidate_feature is None:
            return 0.0

        max_similarity = 0.0
        
        # 与初始姿态比较
        if self.initial_pose_feature is not None:
            max_similarity = max(max_similarity, pose_similarity(self.initial_pose_feature, candidate_feature))
            
        # 与稳定姿态比较
        for feature in self.stable_pose_features:
            max_similarity = max(max_similarity, pose_similarity(feature, candidate_feature))
            
        return max_similarity

    def update(self, x, y, w, h, conf, depth_map=None):
        current_time = time.time()
        dt = max(0.01, current_time - self.last_update_time)
        self.last_update_time = current_time

        self.consecutive_frames += 1
        self.stability = min(1.0, self.consecutive_frames / 30.0)
        self.lock_strength = min(1.0, self.lock_strength + 0.1 * (0.5 + self.stability * 0.5))

        if self.position:
            prev_x, prev_y = self.position
            vx = (x - prev_x) / dt
            vy = (y - prev_y) / dt
            smooth_factor = 0.7
            self.velocity = (
                smooth_factor * vx + (1 - smooth_factor) * self.velocity[0],
                smooth_factor * vy + (1 - smooth_factor) * self.velocity[1]
            )

        self.position = (x, y)
        self.size = (w, h)
        self.confidence = conf
        self.timestamp = current_time
        self.last_seen = current_time
        self.lost_frame_count = 0
        self.trajectory.append((x, y, self.timestamp))

        if not np.isnan(x) and not np.isnan(y):
            measurement = np.array([[x], [y]], dtype=np.float32)
            self.kalman.correct(measurement)

        if depth_map is not None:
            self.world_position = calculate_3d_coordinates(depth_map, (x, y), (w, h))
            if self.world_position != (0, 0, 0) and not any(math.isnan(val) for val in self.world_position):
                X, Y, Z = self.world_position
                if Z > 0.001:
                    self.distance = math.sqrt(X**2 + Y**2 + Z**2)
                    self.yaw = math.atan2(X, Z)
                    distance_xy = math.sqrt(X**2 + Z**2)
                    self.pitch = math.atan2(Y, distance_xy) if distance_xy > 0 else 0

    def predict_next_position(self):
        prediction = self.kalman.predict()
        return prediction[0][0], prediction[1][0]

    def mark_lost(self):
        self.lost_frame_count += 1
        self.lock_strength = max(0.3, self.lock_strength - 0.05)
        self.consecutive_frames = 0
        self.stability = 0.0
        self.pose_landmarks = None  # 清除姿态信息

        if self.lost_frame_count > 50:
            self.active = False

    def get_state(self):
        return {
            'id': self.target_id,
            'distance': self.distance,
            'yaw': self.yaw,
            'pitch': self.pitch,
            'position': self.position,
            'active': self.active,
            'lock_strength': self.lock_strength,
            'pose_score': self.pose_score
        }

    def output_state(self):
        current_time = time.time()
        if current_time - self.last_output_time > 0.2 and self.active:
            status = "活动" if self.active else f"丢失({self.lost_frame_count}帧)"
            if self.world_position and not any(math.isnan(val) for val in self.world_position):
                X, Y, Z = self.world_position
                print(f"目标ID {self.target_id} [{status}]: "
                      f"位置=({X:.2f}m, {Y:.2f}m, {Z:.2f}m), "
                      f"距离={self.distance:.2f}m, "
                      f"方位角={math.degrees(self.yaw):.2f}°, "
                      f"俯仰角={math.degrees(self.pitch):.2f}°, "
                      f"姿态置信度={self.pose_score:.2f}, "
                      f"锁定强度={self.lock_strength:.2f}")
            else:
                print(f"目标ID {self.target_id} [{status}]: "
                      f"位置=({self.position[0]:.1f}, {self.position[1]:.1f}), "
                      f"姿态置信度={self.pose_score:.2f}, "
                      f"锁定强度={self.lock_strength:.2f}")
            self.last_output_time = current_time


# 三维坐标计算函数
def calculate_3d_coordinates(depth_map, center_point, size=None):
    u, v = int(center_point[0]), int(center_point[1])
    height, width = depth_map.shape

    if size is None:
        w, h = 1, 1
    else:
        w, h = size

    roi_size = max(10, min(50, w // 2, h // 2))
    x1 = max(0, u - roi_size)
    y1 = max(0, v - roi_size)
    x2 = min(width - 1, u + roi_size)
    y2 = min(height - 1, v + roi_size)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x1 >= x2 or y1 >= y2:
        return (0, 0, 0)

    depth_roi = depth_map[y1:y2, x1:x2]
    valid_mask = (depth_roi > 300) & (depth_roi < 8000)

    if not np.any(valid_mask):
        return (0, 0, 0)

    valid_depths = depth_roi[valid_mask]
    median_depth = np.median(valid_depths)
    weights = 1.0 / (np.abs(valid_depths - median_depth) + 1e-5)
    weights /= np.sum(weights)
    weighted_depth = np.sum(valid_depths * weights)

    Z_cam = weighted_depth / 1000.0

    if Z_cam <= 0.3 or Z_cam > 15.0:
        return (0, 0, 0)

    fx = 860.0
    fy = 860.0
    cx = width / 2
    cy = height / 2

    try:
        # 相机坐标系坐标
        X_cam = (u - cx) * Z_cam / fx
        Y_cam = (v - cy) * Z_cam / fy
        
        # 转换为世界坐标系
        # 相机坐标系：X轴向右，Y轴向下，Z轴向前
        # 世界坐标系：X轴向前，Y轴向左，Z轴向上
        X_world = Z_cam      # X轴向前（原相机坐标系的Z轴）
        Y_world = -X_cam     # Y轴向左（原相机坐标系的X轴取反）
        Z_world = -Y_cam     # Z轴向上（原相机坐标系的Y轴取反）
        
    except ZeroDivisionError:
        return (0, 0, 0)

    if any(math.isnan(val) for val in (X_world, Y_world, Z_world)):
        return (0, 0, 0)

    return (X_world, Y_world, Z_world)


# 单目标跟踪管理器（优化版本）
class MultiTargetManager:
    def __init__(self):
        self.targets = {}
        self.next_id = 0
        self.active_target_id = None
        self.target_class = "person"
        self.last_detection_time = time.time()
        self.follow_target_id = None  # 客户端指定的跟随目标ID
        self.follow_only_mode = False  # 是否只跟踪指定目标
        self.reappear_threshold = 0.6  # 重新出现匹配阈值
        self.initial_feature_start = None  # 初始特征收集开始时间
        self.initial_feature_duration = 10  # 减少特征收集时间

    def get_new_target_id(self):
        self.next_id += 1
        return self.next_id

    def set_follow_target(self, target_id):
        """设置客户端指定的跟随目标，启用单目标跟踪模式"""
        if target_id is None or target_id == 0:
            self.follow_target_id = None
            self.follow_only_mode = False
            print(f"🛑 停止跟随目标，恢复多目标检测")
        else:
            self.follow_target_id = target_id
            self.follow_only_mode = True  # 启用单目标模式
            if target_id in self.targets:
                self.set_active_target(target_id)
                print(f"🎯 单目标跟踪模式：只跟随目标 ID: {target_id}")
                # 清理其他目标以节省资源
                self.cleanup_non_follow_targets()
                # 开始特征收集
                if self.initial_feature_start is None:
                    self.initial_feature_start = time.time()
            else:
                print(f"⚠️ 目标 ID {target_id} 当前不存在，将在检测到时开始跟随")

    def cleanup_non_follow_targets(self):
        """清理非跟随目标以节省资源"""
        if self.follow_target_id is None:
            return
        
        targets_to_remove = []
        for target_id, target in self.targets.items():
            if target_id != self.follow_target_id:
                targets_to_remove.append(target_id)
        
        for target_id in targets_to_remove:
            del self.targets[target_id]
            print(f"🗑️ 清理非跟随目标 ID: {target_id}")
        
        print(f"💡 单目标模式：保留目标 ID {self.follow_target_id}，清理了 {len(targets_to_remove)} 个其他目标")

    def create_target(self, frame, box, cls_id, conf, depth_map):
        """极简单目标模式：只允许创建一个目标，且只保留该目标特征"""
        if self.follow_only_mode:
            if self.follow_target_id and self.follow_target_id in self.targets:
                return None
        # 清空所有旧目标和特征（极简）
        self.targets.clear()
        feature_storage.clear_all_features()
        target_id = self.get_new_target_id()
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        bbox_width, bbox_height = abs(x2 - x1), abs(y2 - y1)
        new_target = TargetState(target_id)
        new_target.update(center_x, center_y, bbox_width, bbox_height, conf, depth_map)
        new_target.update_reid_features(frame, (x1, y1, x2, y2), is_full_body=True)
        self.targets[target_id] = new_target
        self.set_active_target(target_id)
        self.follow_target_id = target_id
        self.follow_only_mode = True
        self.initial_feature_start = time.time()
        print(f"🎯 极简单目标模式：创建并跟随目标 ID: {target_id}")
        return new_target

    def update_target(self, target, x, y, w, h, conf, depth_map):
        target.update(x, y, w, h, conf, depth_map)

    def find_best_match(self, frame, boxes, cls_ids, confs, depth_map, keypoints_data=None):
        """优化的匹配算法：单目标模式下只匹配跟随目标，集成姿态匹配逻辑"""
        matches = []
        unmatched_detections = list(range(len(boxes)))
        
        # 如果处于单目标跟踪模式，只匹配跟随目标
        if self.follow_only_mode and self.follow_target_id:
            target = self.targets.get(self.follow_target_id)
            if target and target.active:
                best_match_score = 0.0
                best_detection_idx = -1
                
                for i in range(len(boxes)):
                    cls_id = cls_ids[i]
                    if model.names[cls_id] != self.target_class:
                        continue

                    x1, y1, x2, y2 = boxes[i]
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                    # 集成姿态匹配逻辑（来自track_pose.py）
                    match_score = 0.0
                    
                    # ReID特征相似度 (权重0.7)
                    reid_sim = target.compare_signature(frame, (x1, y1, x2, y2))
                    match_score += 0.6 * reid_sim
                    
                    # 姿态相似度 (权重0.2)
                    pose_sim = 0.0
                    if keypoints_data is not None and i < len(keypoints_data):
                        keypoints = keypoints_data[i]
                        pose_sim = target.compare_pose(keypoints)
                    match_score += 0.3 * pose_sim

                    # 位置距离 (权重0.1)
                    if target.position:
                        px, py = target.position
                        distance = math.sqrt((center_x - px)**2 + (center_y - py)**2)
                        position_score = max(0, 1 - distance / 300)  # 增加搜索范围
                        match_score += 0.1 * position_score

                    if match_score > best_match_score and match_score > 0.65:  # 稍微降低匹配阈值
                        best_match_score = match_score
                        best_detection_idx = i
                
                if best_detection_idx >= 0:
                    matches.append((self.follow_target_id, best_detection_idx, best_match_score))
                    if best_detection_idx in unmatched_detections:
                        unmatched_detections.remove(best_detection_idx)
                
                # 单目标模式下，移除所有其他未匹配的检测以避免创建新目标
                if self.follow_only_mode:
                    unmatched_detections = []
        else:
            # 多目标模式的原始逻辑（保留用于非单目标场景）
            for target_id, target in self.targets.items():
                if not target.active:
                    continue
                    
                best_match_score = 0.0
                best_detection_idx = -1
                
                for i in range(len(boxes)):
                    cls_id = cls_ids[i]
                    if model.names[cls_id] != self.target_class:
                        continue

                    x1, y1, x2, y2 = boxes[i]
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                    match_score = 0.0
                    # ReID特征相似度 (权重0.6)
                    reid_sim = target.compare_signature(frame, (x1, y1, x2, y2))
                    match_score += 0.6 * reid_sim
                    
                    # 姿态相似度 (权重0.2)
                    pose_sim = 0.0
                    if keypoints_data is not None and i < len(keypoints_data):
                        keypoints = keypoints_data[i]
                        pose_sim = target.compare_pose(keypoints)
                    match_score += 0.2 * pose_sim

                    # 位置相似度 (权重0.2)
                    if target.position:
                        px, py = target.position
                        distance = math.sqrt((center_x - px)**2 + (center_y - py)**2)
                        position_score = max(0, 1 - distance / 200)
                        match_score += 0.2 * position_score

                    if match_score > best_match_score and match_score > 0.5:
                        best_match_score = match_score
                        best_detection_idx = i
                
                if best_detection_idx >= 0:
                    matches.append((target_id, best_detection_idx, best_match_score))
        
        # 解决冲突
        final_matches = {}
        used_detections = set()
        matches.sort(key=lambda x: x[2], reverse=True)
        
        for target_id, detection_idx, score in matches:
            if detection_idx not in used_detections:
                final_matches[target_id] = detection_idx
                used_detections.add(detection_idx)
                if detection_idx in unmatched_detections:
                    unmatched_detections.remove(detection_idx)
        
        return final_matches, unmatched_detections

    def update_inactive_targets(self):
        """极简单目标模式：丢失后彻底清理目标和特征"""
        current_time = time.time()
        if self.follow_only_mode and self.follow_target_id:
            target = self.targets.get(self.follow_target_id)
            if target:
                if not target.active:
                    target.mark_lost()
                    # 丢失超过一定时间或锁定强度过低，彻底清理
                    if target.lock_strength < 0.3:
                        print(f"🗑️ 丢失目标，彻底清理 ID: {self.follow_target_id}")
                        del self.targets[self.follow_target_id]
                        self.follow_target_id = None
                        self.follow_only_mode = False
                        feature_storage.clear_all_features()
                elif target.active and target.lost_frame_count > 0:
                    pred_x, pred_y = target.predict_next_position()
                    target.position = (pred_x, pred_y)
                    target.last_seen = current_time
        else:
            # 多目标模式：更新所有目标
            for target_id, target in list(self.targets.items()):
                if not target.active:
                    target.mark_lost()
                    if target.lock_strength < 0.3 or current_time - target.last_seen > 10.0:
                        print(f"移除非活动目标 ID: {target_id}")
                        del self.targets[target_id]
                elif target.active and target.lost_frame_count > 0:
                    pred_x, pred_y = target.predict_next_position()
                    target.position = (pred_x, pred_y)
                    target.last_seen = current_time

    def set_active_target(self, target_id):
        if target_id in self.targets:
            self.active_target_id = target_id
            print(f"设置活动目标 ID: {target_id}")

    def get_active_target(self):
        return self.targets.get(self.active_target_id)

    def get_follow_target(self):
        """获取当前跟随的目标"""
        if self.follow_target_id and self.follow_target_id in self.targets:
            return self.targets[self.follow_target_id]
        return None
    
    def get_all_targets(self):
        """获取所有目标的列表"""
        return list(self.targets.values())
        
    def process_detections(self, frame, boxes, cls_ids, confs, depth_map, keypoints_data=None):
        """优化的检测处理：单目标模式下仅处理跟随目标，集成姿态匹配"""
        final_matches, unmatched_detections = self.find_best_match(frame, boxes, cls_ids, confs, depth_map, keypoints_data)
        
        # 更新匹配的目标
        for target_id, detection_idx in final_matches.items():
            target = self.targets[target_id]
            x1, y1, x2, y2 = boxes[detection_idx]
            cls_id = cls_ids[detection_idx]
            conf = confs[detection_idx]
            
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            bbox_width, bbox_height = abs(x2 - x1), abs(y2 - y1)
            
            self.update_target(target, center_x, center_y, bbox_width, bbox_height, conf, depth_map)
            
            # 更新姿态特征（基于YOLO关键点，来自track_pose.py的逻辑）
            if keypoints_data is not None and detection_idx < len(keypoints_data):
                keypoints = keypoints_data[detection_idx]
                target.update_pose_features(keypoints)
            
            # 如果是跟随目标，进行特征收集
            if target_id == self.follow_target_id:
                # [修改] 持续更新特征，而不仅仅是在前10秒
                if target.stability > 0.7: # 只在目标稳定时更新
                    current_time = time.time()
                    if current_time - target.last_feature_save_time >= 1.0:  # 降低特征保存频率
                        print(f"🔄 为目标 {target_id} 持续更新ReID特征 (稳定性: {target.stability:.2f})")
                        target.update_reid_features(frame, (x1, y1, x2, y2), is_full_body=False)
                        target.last_feature_save_time = current_time
        
        # 创建新目标（单目标模式下限制创建）
        for detection_idx in unmatched_detections:
            cls_id = cls_ids[detection_idx]
            if model.names[cls_id] == self.target_class and confs[detection_idx] > 0.6:
                # 单目标模式下，只有在没有跟随目标或指定ID时才创建
                if not self.follow_only_mode or (self.follow_target_id and len(self.targets) == 0):
                    x1, y1, x2, y2 = boxes[detection_idx]
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # 过滤边缘检测
                    if center_x < 50 or center_x > frame.shape[1] - 50 or center_y < 50 or center_y > frame.shape[0] - 50:
                        continue
                    
                    new_target = self.create_target(frame, [x1, y1, x2, y2], cls_id, confs[detection_idx], depth_map)
                    if new_target:
                        # 为新目标初始化姿态特征（基于YOLO关键点）
                        if keypoints_data is not None and detection_idx < len(keypoints_data):
                            keypoints = keypoints_data[detection_idx]
                            new_target.update_pose_features(keypoints)
        
        return final_matches, unmatched_detections

    def output_all_states(self):
        """优化：只输出跟随目标的状态"""
        if self.follow_only_mode and self.follow_target_id:
            target = self.targets.get(self.follow_target_id)
            if target:
                target.output_state()
        else:
            # 多目标模式才输出所有状态
            for target in self.targets.values():
                target.output_state()

    def has_active_targets(self):
        """检查是否有活跃目标"""
        if self.follow_only_mode and self.follow_target_id:
            target = self.targets.get(self.follow_target_id)
            return target and target.active
        else:
            return any(target.active for target in self.targets.values())

    def draw_target_list(self, vis_frame, all_detected_targets):
        """在屏幕上绘制所有目标ID列表"""
        if not all_detected_targets:
            return
        
        # 在右侧绘制目标列表
        list_x = vis_frame.shape[1] - 250
        list_y = 100
        
        # 绘制背景
        cv2.rectangle(vis_frame, (list_x - 10, list_y - 30), 
                     (vis_frame.shape[1] - 10, list_y + len(all_detected_targets) * 25 + 50), 
                     (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (list_x - 10, list_y - 30), 
                     (vis_frame.shape[1] - 10, list_y + len(all_detected_targets) * 25 + 50), 
                     (255, 255, 255), 2)
        
        # 标题
        cv2.putText(vis_frame, "DETECTED TARGETS", (list_x, list_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示跟随目标信息
        follow_target = self.get_follow_target()
        if follow_target:
            follow_text = f"FOLLOW: ID-{follow_target.target_id}"
            cv2.putText(vis_frame, follow_text, (list_x, list_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(vis_frame, "FOLLOW: NONE", (list_x, list_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # 绘制目标列表
        current_y = list_y + 40
        for i, target_info in enumerate(all_detected_targets):
            target_id = target_info['id']
            conf = target_info['conf']
            status = target_info['status']
            is_follow = target_info['is_follow_target']
            
            # 设置颜色
            if is_follow:
                color = (0, 0, 255)  # 红色 - 跟随目标
                prefix = "► "
            elif status == 'NEW':
                color = (0, 255, 0)  # 绿色 - 新检测
                prefix = "● "
            else:
                color = (255, 255, 0)  # 青色 - 已跟踪目标
                prefix = "○ "
            
            # 绘制目标信息
            if target_id == 'NEW':
                text = f"{prefix}NEW ({conf:.2f})"
            else:
                text = f"{prefix}ID-{target_id} ({conf:.2f})"
            
            cv2.putText(vis_frame, text, (list_x, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            current_y += 25
        
        # 显示说明
        instructions_y = current_y + 10
        cv2.putText(vis_frame, "► = Following", (list_x, instructions_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(vis_frame, "○ = Tracked", (list_x, instructions_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(vis_frame, "● = New", (list_x, instructions_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# 全局模型变量
model = None

# 帧捕获线程
class FrameCaptureThread(threading.Thread):
    def __init__(self, device, frame_queue):
        super().__init__()
        self.device = device
        self.frame_queue = frame_queue
        self.running = True
        self.q_rgb = None
        self.q_depth = None

    def initialize_queues(self):
        """初始化设备队列"""
        try:
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=True)
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=True)
            return True
        except Exception as e:
            print(f"队列初始化失败: {e}")
            return False

    def run(self):
        if not self.initialize_queues():
            print("帧捕获线程启动失败：无法初始化队列")
            return

        while self.running:
            try:
                in_rgb = self.q_rgb.get()
                in_depth = self.q_depth.get()
                if in_rgb is not None and in_depth is not None:
                    frame = in_rgb.getCvFrame()
                    depth_frame = in_depth.getFrame()
                    
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait() # 丢弃旧帧
                    self.frame_queue.put((frame, depth_frame))
            except Exception as e:
                print(f"相机线程错误: {e}")
                self.running = False
        print("相机线程已停止。")

    def stop(self):
        self.running = False

# 处理线程（支持可视化开关）
class ProcessingThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, stop_event, grpc_server_address='localhost:50051', enable_visualization=True):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.target_manager = MultiTargetManager()
        self.special_target_missing_start = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 可视化开关
        self.enable_visualization = enable_visualization
        
        # 初始化gRPC客户端
        self.grpc_client = TrackingGRPCClient(grpc_server_address)
        self.grpc_enabled = False
        self.tracking_start_time = None
        
        # 第一次跟踪指令标志
        self.first_tracking_command = True

    def run(self):
        global model
        
        # 尝试连接gRPC服务器
        print("尝试连接gRPC服务器...")
        if self.grpc_client.connect():
            self.grpc_enabled = True
            print("gRPC通信已启用")
        else:
            self.grpc_enabled = False
            print("gRPC通信未启用，继续运行本地模式")
        
        try:
            model = YOLO('yolo11n-pose.pt')
            print("YOLO模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.stop_event.set()
            return

        while not self.stop_event.is_set():
            try:
                frame, depth_frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            if depth_frame.shape != (480, 640):
                depth_frame = cv2.resize(depth_frame, (640, 480), interpolation=cv2.INTER_NEAREST)
            depth_frame = cv2.medianBlur(depth_frame.astype(np.float32), 5).astype(np.uint16)

            # --- FPS 计算 ---
            self.frame_count += 1
            current_time = time.time()
            if self.frame_count % 30 == 0:  # 减少FPS打印频率
                elapsed = current_time - self.start_time
                self.fps = 30 / elapsed if elapsed > 0 else 0
                self.start_time = current_time
                self.frame_count = 0
                print(f"FPS: {self.fps:.1f}")

            # --- 目标检测（优化参数）---
            try:
                # 优化检测参数以提高速度
                results = model.track(
                    frame,
                    imgsz=640,  # 减小输入尺寸提高速度
                    tracker='botsort.yaml',
                    conf=0.6,  # 提高置信度阈值减少误检
                    iou=0.6,   # 提高IOU阈值减少重叠框
                    persist=True,
                    half=True,
                    verbose=False,
                    max_det=10,  # 限制最大检测数量
                    agnostic_nms=True  # 启用类别无关的NMS
                )
            except Exception as e:
                print(f"目标检测失败: {e}")
                results = []

            # --- 检查gRPC跟随指令 ---
            if self.grpc_enabled:
                try:
                    follow_commands = self.grpc_client.get_follow_commands()
                    if follow_commands:
                        for cmd in follow_commands:
                            if cmd['action'] == 'follow' and 'target_id' in cmd:
                                target_id = cmd['target_id']
                                
                                # 如果是第一次收到跟踪指令，重置所有特征并启用单目标模式
                                if self.first_tracking_command:
                                    print("🔄 首次跟踪指令，重置特征库并启用单目标模式...")
                                    feature_storage.clear_all_features()
                                    self.target_manager.targets.clear()
                                    self.target_manager.next_id = 0
                                    self.target_manager.active_target_id = None
                                    self.target_manager.set_follow_target(target_id)
                                    self.tracking_start_time = time.time()
                                    self.first_tracking_command = False
                                    print(f"✅ 启用单目标跟踪模式，目标ID: {target_id}")
                                else:
                                    self.target_manager.set_follow_target(target_id)
                                
                            elif cmd['action'] == 'stop_follow':
                                self.first_tracking_command = True
                                self.target_manager.set_follow_target(None)
                                self.tracking_start_time = None
                                print("🛑 停止跟踪，退出单目标模式")
                except Exception as e:
                    pass  # 静默处理gRPC错误

            # --- 结果处理和跟踪（多/单目标可视化）---
            vis_frame = frame.copy() if self.enable_visualization else None
            all_detected_targets = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                
                # 提取关键点数据（如果有的话）
                keypoints_data = None
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    keypoints_tensor = results[0].keypoints.xy.cpu().numpy()  # shape: (n_detections, 17, 2)
                    keypoints_data = []
                    for i in range(len(keypoints_tensor)):
                        # 转换为 (17, 2) 格式，每个关键点 [x, y]
                        kpts = keypoints_tensor[i]  # (17, 2)
                        keypoints_data.append(kpts)

                final_matches, unmatched_detections = self.target_manager.process_detections(
                    frame, boxes_xyxy, cls_ids, confs, depth_frame, keypoints_data
                )

                # 发送跟随目标数据到gRPC（只有在单目标模式下且跟随目标被成功匹配时才发送）
                follow_target = self.target_manager.get_follow_target()
                if (follow_target and follow_target.active and self.grpc_enabled and 
                    self.target_manager.follow_only_mode and 
                    follow_target.target_id in final_matches):
                    try:
                        self.grpc_client.send_target_coordinates(follow_target)
                        self.grpc_client.send_tracking_status(
                            is_active=True,
                            target_id=follow_target.target_id,
                            tracking_time=time.time() - self.tracking_start_time if self.tracking_start_time else 0.0
                        )
                    except Exception:
                        pass
                elif self.grpc_enabled and self.target_manager.follow_only_mode:
                    # 单目标模式下，如果跟随目标未被匹配到，发送停止坐标
                    try:
                        dummy_state = type('DummyState', (), {'active': False, 'target_id': 0, 'world_position': None, 'distance': 0, 'yaw': 0, 'pitch': 0, 'confidence': 0})()
                        self.grpc_client.send_target_coordinates(dummy_state)
                        self.grpc_client.send_tracking_status(is_active=False, target_id=0, tracking_time=0.0)
                    except Exception:
                        pass

                # --------- 可视化 ---------
                if self.enable_visualization and vis_frame is not None:
                    # 单目标模式：只画跟随目标
                    if self.target_manager.follow_only_mode and follow_target:
                        if follow_target.target_id in final_matches:
                            detection_idx = final_matches[follow_target.target_id]
                            bbox = boxes_xyxy[detection_idx]
                            conf = confs[detection_idx]
                            self.draw_simple_box(vis_frame, bbox, follow_target.target_id, conf, (0, 0, 255), 'FOLLOW')
                    else:
                        # 多目标模式：画所有目标框和右侧ID列表
                        # 1. 画所有已匹配目标
                        for target_id, detection_idx in final_matches.items():
                            target = self.target_manager.targets[target_id]
                            bbox = boxes_xyxy[detection_idx]
                            conf = confs[detection_idx]
                            is_follow = (target_id == self.target_manager.follow_target_id)
                            color = (0, 0, 255) if is_follow else (255, 255, 0)
                            status = 'FOLLOW' if is_follow else 'TRACKED'
                            self.draw_simple_box(vis_frame, bbox, target_id, conf, color, status)
                            all_detected_targets.append({
                                'id': target_id,
                                'bbox': bbox,
                                'conf': conf,
                                'is_follow_target': is_follow,
                                'status': status
                            })
                        # 2. 画未匹配的新目标
                        for detection_idx in unmatched_detections:
                            cls_id = cls_ids[detection_idx]
                            if model.names[cls_id] == self.target_manager.target_class:
                                bbox = boxes_xyxy[detection_idx]
                                conf = confs[detection_idx]
                                self.draw_simple_box(vis_frame, bbox, 'NEW', conf, (0, 255, 0), 'NEW')
                                all_detected_targets.append({
                                    'id': 'NEW',
                                    'bbox': bbox,
                                    'conf': conf,
                                    'is_follow_target': False,
                                    'status': 'NEW'
                                })
                        # 3. 右侧目标ID列表
                        # self.target_manager.draw_target_list(vis_frame, all_detected_targets)
                    # 系统信息
                    self.draw_info(vis_frame, depth_frame)
            else:
                # 没有检测到目标，确保停止发送gRPC数据（适用于所有模式）
                if self.grpc_enabled:
                    try:
                        dummy_state = type('DummyState', (), {'active': False, 'target_id': 0, 'world_position': None, 'distance': 0, 'yaw': 0, 'pitch': 0, 'confidence': 0})()
                        self.grpc_client.send_target_coordinates(dummy_state)
                        self.grpc_client.send_tracking_status(is_active=False, target_id=0, tracking_time=0.0)
                    except Exception:
                        pass
                # 显示无目标状态
                if self.enable_visualization and vis_frame is not None:
                    self.draw_info(vis_frame, depth_frame)

            # 更新目标状态（优化：仅更新跟随目标）
            prev_follow_target_id = self.target_manager.follow_target_id
            self.target_manager.update_inactive_targets()
            
            # 检查跟随目标是否被清理，如果是则发送停止坐标
            if (prev_follow_target_id is not None and 
                self.target_manager.follow_target_id is None and 
                self.grpc_enabled):
                try:
                    dummy_state = type('DummyState', (), {'active': False, 'target_id': 0, 'world_position': None, 'distance': 0, 'yaw': 0, 'pitch': 0, 'confidence': 0})()
                    self.grpc_client.send_target_coordinates(dummy_state)
                    self.grpc_client.send_tracking_status(is_active=False, target_id=0, tracking_time=0.0)
                    print("📡 跟随目标已丢失，发送停止坐标")
                except Exception:
                    pass
            
            self.target_manager.output_all_states()
            
            # 输出可视化结果
            if self.enable_visualization and vis_frame is not None and self.result_queue is not None:
                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()  # 丢弃旧帧
                    self.result_queue.put(vis_frame)
                except queue.Full:
                    pass  # 静默处理队列满的情况
        
        # 断开gRPC连接
        if self.grpc_enabled:
            self.grpc_client.disconnect()
        
        print("处理线程已停止。")

    def reset_tracker(self):
        self.target_manager = MultiTargetManager()
        self.special_target_missing_start = None
        print("手动重置目标选择")
    
    def draw_info(self, vis_frame, depth_frame):
        """优化的系统信息绘制（单目标模式）"""
        # 显示FPS
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示单目标模式状态
        # mode_text = "单目标跟踪模式" if self.target_manager.follow_only_mode else "多目标检测模式"
        # cv2.putText(vis_frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 显示gRPC状态
        grpc_status = "gRPC: ON" if self.grpc_enabled else "gRPC: OFF"
        grpc_color = (0, 255, 0) if self.grpc_enabled else (0, 0, 255)
        cv2.putText(vis_frame, grpc_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grpc_color, 2)
        
        # 显示跟随目标信息
        follow_target = self.target_manager.get_follow_target()
        if follow_target:
            status = "ACTIVE" if follow_target.active else f"LOST({follow_target.lost_frame_count}fps)"
            cv2.putText(vis_frame, f"Follow ID:{follow_target.target_id} [{status}]", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if follow_target.world_position and follow_target.active:
                cv2.putText(vis_frame, f"Distance: {follow_target.distance:.2f}m", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Yaw: {math.degrees(follow_target.yaw):.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Pitch: {math.degrees(follow_target.pitch):.1f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "No target selected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # 显示特征收集状态（仅在单目标模式下）
        if self.target_manager.follow_only_mode:
            current_time = time.time()
            if self.target_manager.initial_feature_start is not None:
                elapsed = current_time - self.target_manager.initial_feature_start
                if elapsed <= self.target_manager.initial_feature_duration:
                    remaining = self.target_manager.initial_feature_duration - elapsed
                #     cv2.putText(vis_frame, f"特征收集中: {remaining:.1f}s", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # elif elapsed <= self.target_manager.initial_feature_duration + 2:
                #     cv2.putText(vis_frame, "特征收集完成!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 简化的控制说明
        cv2.putText(vis_frame, "Q=Quit | R=Reset", (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def toggle_visualization(self):
        """切换可视化开关"""
        self.enable_visualization = not self.enable_visualization
        status = "启用" if self.enable_visualization else "禁用"
        print(f"可视化已{status}")
        return self.enable_visualization

    def draw_simple_box(self, vis_frame, bbox, target_id, conf, color, status):
        """简化的目标框绘制，包含姿态信息（提高性能）"""
        x1, y1, x2, y2 = map(int, bbox)
        thickness = 3 if status == 'FOLLOW' else 2
        
        # 绘制边界框
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
        
        # 获取目标的姿态信息
        pose_score = 0.0
        lock_strength = 0.0
        if target_id != 'NEW' and target_id in self.target_manager.targets:
            target = self.target_manager.targets[target_id]
            pose_score = target.pose_score
            lock_strength = target.lock_strength
            
            # 绘制姿态关键点（仅在置信度高时）- 适配YOLO关键点格式
            if target.pose_landmarks is not None and target.pose_score > 0.5:
                keypoints = target.pose_landmarks  # YOLO格式: (17, 2)
                if len(keypoints) >= 17:
                    for i, (kx, ky) in enumerate(keypoints):
                        if kx > 0 and ky > 0:  # 有效关键点
                            # YOLO关键点已经是绝对坐标，直接使用
                            cx, cy = int(kx), int(ky)
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                cv2.circle(vis_frame, (cx, cy), 2, (0, 255, 0), -1)
        
        # 绘制标签
        label = f"ID{target_id} {conf:.2f}"
        if status == 'FOLLOW':
            label = f"{label} [FOLLOW]"
        
        # 绘制标签背景
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 绘制姿态和锁定强度信息
        if target_id != 'NEW' and status == 'FOLLOW':
            info_text = f"Pose: {pose_score:.2f} | Lock: {lock_strength:.2f}"
            cv2.putText(vis_frame, info_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        
        # 绘制中心点
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)


# 主程序（集成gRPC通信和可视化）
def main(grpc_server='localhost:50051', enable_visualization=True):
    global reid_handler  # 声明全局变量
    
    print(f"=== OAK单目标跟踪系统 (性能优化版, fast-reid 库) ===")
    print(f"gRPC服务器地址: {grpc_server}")
    print(f"可视化状态: {'启用' if enable_visualization else '禁用'}")
    
    # --- [新增] 初始化 ReID 处理器 ---
    print("正在初始化ReID处理器...")
    reid_handler = ReIDHandler(model_path="weights/ReID_resnet50_ibn_a.pth")
    if reid_handler.model is None:
        print("错误: ReID模型初始化失败，程序退出。")
        return False
    
    # 创建相机管理器
    camera_manager = CameraManager(max_retries=5, retry_delay=3)
    
    # 尝试连接相机
    if not camera_manager.connect_camera():
        print("❌ 无法连接到OAK相机，程序退出")
        return False
    
    device = camera_manager.get_device()
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1) if enable_visualization else None
    stop_event = threading.Event()
    
    capture_thread = None
    processing_thread = None
    
    try:
        print("✅ OAK相机已连接，启动优化的处理流程...")
        print("📡 gRPC通信功能已集成")
        if enable_visualization:
            print("🎮 控制: Q=退出, R=重置, V=切换可视化")
        else:
            print("⚡ 无可视化模式 - 最大化性能运行")
            print("🎮 按 Ctrl+C 停止程序")
        
        capture_thread = FrameCaptureThread(device, frame_queue)
        processing_thread = ProcessingThread(frame_queue, result_queue, stop_event, grpc_server, enable_visualization)

        capture_thread.start()
        processing_thread.start()
        print("🧵 处理线程已启动")

        if enable_visualization and result_queue is not None:
            # 创建可视化窗口
            window_name = 'OAK单目标跟踪系统 - 性能优化版'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)

            # 减少预热时间以提高启动速度
            print("📸 相机预热中...")
            for i in range(15):  # 减少到15帧
                try:
                    frame, _ = frame_queue.get(timeout=2)
                    text = f"WARMING UP... {i+1}/15"
                    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_event.set()
                        break
                except queue.Empty:
                    print("⚠️ 预热期间未能从相机获取帧。")
                    break
            
            if not stop_event.is_set():
                print("✅ 预热完成，进入单目标跟踪模式")
                print("💡 等待gRPC指令选择跟踪目标...")

            # 主可视化循环
            while not stop_event.is_set():
                try:
                    display_frame = result_queue.get(timeout=1)
                    cv2.imshow(window_name, display_frame)
                except queue.Empty:
                    # 如果处理线程卡住，检查它是否还活着
                    if not processing_thread.is_alive():
                        print("❌ 处理线程已意外终止。")
                        break
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                elif key == ord('r'):
                    processing_thread.reset_tracker()
                    print("🔄 跟踪器已重置")
                elif key == ord('v'):
                    visualization_status = processing_thread.toggle_visualization()
                    if not visualization_status:
                        cv2.destroyAllWindows()
                        print("📺 可视化已禁用，切换到性能模式")
                        break  # 退出可视化循环
                elif key == ord(' '):
                    print("⏸️ 已暂停，按任意键继续...")
                    cv2.waitKey(0)  # 暂停直到按任意键
            
            cv2.destroyAllWindows()
        else:
            # 无可视化模式 - 最大化性能
            print("⚡ 进入高性能无可视化模式")
            try:
                while not stop_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 收到中断信号，正在停止...")
                stop_event.set()

    except Exception as e:
        print(f"❌ 主程序运行出错: {e}")
        stop_event.set()
    
    finally:
        print("正在停止线程...")
        stop_event.set()
        
        if capture_thread:
            capture_thread.stop()
            capture_thread.join(timeout=2)
        
        if processing_thread:
            processing_thread.join(timeout=5)

        # 关闭相机连接
        camera_manager.close()
        
        print("\n===== 最终跟踪报告 =====")
        if processing_thread and processing_thread.target_manager:
            tm = processing_thread.target_manager
            print(f"总跟踪目标数: {len(tm.targets)}")
            for target_id, target in tm.targets.items():
                status = "活动" if target.active else f"丢失({target.lost_frame_count}帧)"
                if target.position:
                    print(f"目标 ID {target_id}: {status}, 最后位置: ({target.position[0]:.1f}, {target.position[1]:.1f}), "
                          f"姿态置信度: {target.pose_score:.2f}, "
                          f"锁定强度: {target.lock_strength:.2f}")
                else:
                    print(f"目标 ID {target_id}: {status}, 无位置信息")
            # 保存所有特征
            feature_storage.save_features()
        else:
            print("未能生成报告，处理线程未正常初始化。")
    
    return True

if __name__ == "__main__":
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='OAK单目标跟踪系统 - 性能优化版 (集成gRPC通信和fast-reid库)')
    parser.add_argument('--grpc-server', default='localhost:50051', 
                       help='gRPC服务器地址 (默认: localhost:50051)')
    parser.add_argument('--retries', type=int, default=3, 
                       help='程序启动重试次数 (默认: 3)')
    parser.add_argument('--no-viz', '--no-visualization', action='store_true',
                       help='启用高性能模式：禁用可视化界面，仅运行后台跟踪和gRPC通信')
    parser.add_argument('--headless', action='store_true',
                       help='无头模式运行（等同于 --no-viz，最大化性能）')
    
    args = parser.parse_args()
    
    # 确定可视化状态
    enable_visualization = not (args.no_viz or args.headless)
    
    print(f"=== OAK单目标跟踪系统启动 - 性能优化版 (fast-reid) ===")
    print(f"gRPC服务器: {args.grpc_server}")
    print(f"最大重试次数: {args.retries}")
    print(f"可视化模式: {'启用' if enable_visualization else '禁用（高性能模式）'}")

    
    # 添加重试机制到主程序
    max_program_retries = args.retries
    for attempt in range(max_program_retries):
        print(f"\n========== 程序启动尝试 {attempt + 1}/{max_program_retries} ==========")
        
        try:
            if main(args.grpc_server, enable_visualization):
                print("程序正常退出")
                break
            else:
                print(f"程序启动失败 (第 {attempt + 1} 次)")
        except Exception as e:
            print(f"程序运行异常 (第 {attempt + 1} 次): {e}")
        
        if attempt < max_program_retries - 1:
            print(f"等待5秒后重试...")
            time.sleep(5)
        else:
            print("所有启动尝试都失败了")
    
    sys.exit(0)

# echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
# sudo udevadm control --reload-rules && sudo udevadm trigger