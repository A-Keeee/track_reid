# 文件名: reid_depthai_oak_with_depth.py

from pathlib import Path
import random
import argparse
import numpy as np
import sys
import os
import track_torch_pose
import cv2
import time
import torch.nn.functional as F
import depthai as dai
from ultralytics import YOLO
from PIL import Image
import math  # <-- 新增导入

# ReID 相关导入
from reid.data.transforms import build_transforms
from utils.ops import Profile
from reid.config import cfg as reidCfg
from reid.data import make_data_loader
from reid.modeling import build_model
from utils.checks import check_file
from utils.files import increment_path
from utils.plotting import plot_one_box

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# OAK 相机全局变量
selected_bbox = None
detected_boxes = []
display_frame = None
original_frame = None  # 存储原始干净帧
selected_bboxes = []  # 存储多次选择的结果
selected_frames = []  # 存储对应的帧
current_selection_count = 0

# YOLO v8 标签映射
labelMap = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# ==========================================================================================
# /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/--/-/-/- 新增/修改部分开始 /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
# ==========================================================================================

def create_camera_pipeline():
    """
    创建 DepthAI 相机管道（输出RGB和深度图像）
    参考 track_grpc_pose.py 的实现
    """
    pipeline = dai.Pipeline()
    
    # 定义源和输出
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    
    # 相机属性
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # 深度设置
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # 关键：将深度图与RGB摄像头对齐
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    # 连接
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)
    
    return pipeline

def calculate_3d_coordinates(depth_map, center_point, size=None):
    """
    从深度图和2D点计算3D坐标（相机坐标系）
    """
    u, v = int(center_point[0]), int(center_point[1])
    height, width = depth_map.shape

    if size is None:
        w, h = 10, 10
    else:
        w, h = size

    # 在中心点周围取一个小的ROI来提高深度估算的鲁棒性
    roi_size = max(5, int(min(w, h) * 0.1)) # ROI大小为边界框短边的10%，最小为5像素
    x1 = max(0, u - roi_size)
    y1 = max(0, v - roi_size)
    x2 = min(width - 1, u + roi_size)
    y2 = min(height - 1, v + roi_size)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x1 >= x2 or y1 >= y2:
        return (0, 0, 0)

    depth_roi = depth_map[y1:y2, x1:x2]
    # 过滤掉无效的深度值（0表示无数据）和在合理范围外的值
    valid_mask = (depth_roi > 300) & (depth_roi < 8000) # 有效范围：0.3米到8米

    if not np.any(valid_mask):
        return (0, 0, 0)

    # 使用有效深度值的中位数来抵抗异常值
    valid_depths = depth_roi[valid_mask]
    median_depth = np.median(valid_depths)

    Z_cam = median_depth / 1000.0  # 将单位从毫米(mm)转换为米(m)

    if Z_cam <= 0.3 or Z_cam > 15.0: # 再次进行范围检查
        return (0, 0, 0)

    # 相机内参 (近似值，适用于640x480预览)
    # 为了更高精度，建议使用相机标定得到的精确值
    fx = 860.0
    fy = 860.0
    cx = width / 2.0
    cy = height / 2.0

    try:
        # 反投影计算相机坐标系下的X和Y
        X_cam = (u - cx) * Z_cam / fx
        Y_cam = (v - cy) * Z_cam / fy
    except ZeroDivisionError:
        return (0, 0, 0)

    if any(math.isnan(val) for val in (X_cam, Y_cam, Z_cam)):
        return (0, 0, 0)
    

    

    # 返回相机坐标系下的坐标 (X: 右, Y: 下, Z: 前)
    return (X_cam, Y_cam, Z_cam)


# ========================================================================================
# /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/--/-/-/- 新增/修改部分结束 /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
# ========================================================================================


def mouse_callback(event, x, y, flags, param):
    """
    鼠标回调函数，用于点击选择检测到的目标（支持多次选择）
    """
    global selected_bbox, detected_boxes, selected_bboxes, current_selection_count, original_frame, selected_frames
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 检查点击的位置是否在某个检测框内
        for i, (xmin, ymin, xmax, ymax) in enumerate(detected_boxes):
            if xmin <= x <= xmax and ymin <= y <= ymax:
                selected_bbox = (xmin, ymin, xmax, ymax)
                selected_bboxes.append((xmin, ymin, xmax, ymax))
                current_selection_count += 1
                
                # 保存原始干净帧（不含UI元素）
                if original_frame is not None:
                    selected_frames.append(original_frame.copy())
                    print(f"✓ 已选择目标 {current_selection_count}: bbox=({xmin}, {ymin}, {xmax}, {ymax})")
                    print(f"✓ 已保存第 {current_selection_count} 次选择的干净帧")
                else:
                    print(f"✓ 已选择目标 {current_selection_count}: bbox=({xmin}, {ymin}, {xmax}, {ymax})")
                    print(f"⚠️ 原始帧为空，无法保存")
                return
        print("⚠️ 请点击检测到的人员框内")

def detect_all_persons(frame, model, conf_thres=0.5):
    """
    检测画面中的所有人员（使用 ultralytics YOLO）
    
    Args:
        frame: 输入图像
        model: YOLO模型
        conf_thres: 置信度阈值
    
    Returns:
        list: 检测到的人员bounding boxes列表
    """
    results = model.predict(source=frame, show=False, classes=[0], 
                          conf=conf_thres, verbose=False)
    det = results[0]
    
    boxes = []
    if len(det.boxes) > 0:
        for box in det.boxes:
            if box.conf[0] > conf_thres:
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                
                # 过滤太小的目标
                box_area = (xmax - xmin) * (ymax - ymin)
                if box_area > 2000:
                    boxes.append((xmin, ymin, xmax, ymax))
    
    return boxes



def interactive_target_selection_oak(device, qRgb, model, conf_thres=0.5, max_selections=3):
    """
    使用OAK相机进行实时交互式目标选择
    
    Args:
        device: DepthAI设备
        qRgb: RGB输出队列
        model: YOLO模型
        conf_thres: 检测置信度阈值
        max_selections: 最大选择次数
    
    Returns:
        tuple: (选中的bounding box列表, 选择时的帧列表) 或 (None, None)
    """
    global selected_bbox, detected_boxes, display_frame, original_frame, selected_bboxes, selected_frames, current_selection_count
    
    # 重置选择状态
    selected_bbox = None
    detected_boxes = []
    selected_bboxes = []
    selected_frames = []
    current_selection_count = 0
    original_frame = None
    
    print("OAK相机实时视频流已启动，请点击想要跟踪的目标...")
    print("操作说明:")
    print(f"- 点击检测到的人员框选择目标（可选择{max_selections}次）")
    print("- 按 'q' 键退出选择")
    print("- 按 'r' 键刷新检测")
    print("- 按 'c' 键确认当前选择并继续")
    print("- 按 'u' 键撤销上一次选择")
    
    # 设置鼠标回调
    cv2.namedWindow('OAK Target Selection', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('OAK Target Selection', mouse_callback)
    
    frame_count = 0
    fps_counter = time.time()
    
    while current_selection_count < max_selections:
        # 获取RGB帧
        inRgb = qRgb.tryGet()
        
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            original_frame = frame.copy()  # 保存原始干净帧
            display_frame = frame.copy()   # 创建用于显示的帧副本
            frame_count += 1
            
            # 每3帧检测一次，提高实时性
            if frame_count % 3 == 0:
                # 使用YOLO检测当前帧中的所有人员
                detected_boxes = detect_all_persons(original_frame, model, conf_thres)
        
        # 重置选择状态
        if selected_bbox is not None:
            selected_bbox = None  # 重置等待下次选择
        
        if display_frame is not None:
            # 绘制所有检测框和编号
            for i, (xmin, ymin, xmax, ymax) in enumerate(detected_boxes):
                # 绘制检测框
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # 添加编号标签
                label = f"Person {i+1}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (xmin, ymin-label_size[1]-10), 
                             (xmin+label_size[0], ymin), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (xmin, ymin-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # 添加点击提示
                cv2.putText(display_frame, "Click to select", (xmin, ymax+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 添加状态信息
            status_text = f"OAK Detected: {len(detected_boxes)} persons | Selected: {current_selection_count}/{max_selections}"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 添加操作提示
            instruction_text = "'q':quit | 'c':confirm | 'u':undo | 'r':refresh"
            cv2.putText(display_frame, instruction_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示FPS
            if frame_count % 30 == 0 and frame_count > 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_counter)
                fps_counter = current_time
            
            if frame_count > 30:
                fps_text = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --"
                cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 如果没有检测到目标，显示提示
            if len(detected_boxes) == 0:
                no_detection_text = "No persons detected by OAK camera"
                cv2.putText(display_frame, no_detection_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 绘制已选择的目标（用不同颜色标记）
            for i, (xmin, ymin, xmax, ymax) in enumerate(selected_bboxes):
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                cv2.putText(display_frame, f"Selected {i+1}", (xmin, ymin-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('OAK Target Selection', display_frame)
        
        # 检查键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户取消选择")
            break
        elif key == ord('r'):
            print("刷新检测...")
            detected_boxes = []  # 强制重新检测
        elif key == ord('c'):
            if current_selection_count > 0:
                print(f"确认选择，已选择 {current_selection_count} 次")
                break
            else:
                print("请先选择至少一个目标")
        elif key == ord('u'):
            if current_selection_count > 0:
                # 撤销上一次选择
                selected_bboxes.pop()
                if selected_frames:
                    selected_frames.pop()
                current_selection_count -= 1
                print(f"已撤销上一次选择，当前选择次数: {current_selection_count}")
            else:
                print("没有可撤销的选择")
    
    # 检查是否有有效选择
    if current_selection_count == 0:
        cv2.destroyAllWindows()
        return None, None
    
    # 显示最终选择结果
    if current_selection_count > 0:
        print(f"✓ 选择完成，共选择了 {current_selection_count} 次")
        
        # 显示最终选择结果
        if display_frame is not None:
            final_frame = display_frame.copy()
            for i, (xmin, ymin, xmax, ymax) in enumerate(selected_bboxes):
                cv2.rectangle(final_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
                cv2.putText(final_frame, f"TARGET {i+1} SELECTED!", (xmin, ymin-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(final_frame, f"Processing {current_selection_count} targets...", (10, final_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('OAK Target Selection', final_frame)
            cv2.waitKey(2000)  # 显示2秒
    
    cv2.destroyAllWindows()
    return selected_bboxes, selected_frames

def save_query_images(frames, bboxes, base_path="query", target_size=(128, 256)):
    """
    保存多个查询图像
    
    Args:
        frames: 原始帧列表
        bboxes: bounding box列表 [(xmin, ymin, xmax, ymax), ...]
        base_path: 保存目录
        target_size: 目标尺寸 (width, height)
    
    Returns:
        bool: 是否成功保存所有图像
    """
    print(f"调试信息: 帧数量={len(frames)}, 边界框数量={len(bboxes)}")
    if len(frames) != len(bboxes):
        print(f"错误：帧数量({len(frames)})与边界框数量({len(bboxes)})不匹配")
        return False
    
    try:
        # 确保目录存在
        os.makedirs(base_path, exist_ok=True)
        
        success_count = 0
        for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
            xmin, ymin, xmax, ymax = bbox
            crop_img = frame[ymin:ymax, xmin:xmax]
            
            if crop_img.size == 0:
                print(f"警告：第{i+1}个图像区域为空")
                continue
                
            # 调整图像大小
            img_resized = cv2.resize(crop_img, target_size)
            
            # 保存图像
            save_path = os.path.join(base_path, f"0001_c1s1_{i}_center.jpg")
            cv2.imwrite(save_path, img_resized)
            print(f"✓ 已保存查询图像 {i+1}: {save_path}")
            success_count += 1
        
        print(f"✓ 成功保存 {success_count}/{len(frames)} 个查询图像")
        return success_count > 0
        
    except Exception as e:
        print(f"保存查询图像时出错: {e}")
        return False

def initialize_reid_model(device):
    """
    初始化Reid模型并加载查询特征
    
    Args:
        device: 计算设备
        
    Returns:
        tuple: (reidModel, query_feats) 或 (None, None)
    """
    try:
        print("正在初始化Reid模型...")
        query_loader, num_query = make_data_loader(reidCfg)
        
        if num_query == 0:
            print("错误：未找到查询图像")
            return None, None
            
        reidModel = build_model(reidCfg, num_classes=1501)
        reidModel.load_param(reidCfg.TEST.WEIGHT)
        reidModel.to(device).eval()
        
        # 提取查询特征
        query_feats = []
        query_pids = []
        
        with track_torch_pose.no_grad():
            for i, batch in enumerate(query_loader):
                img, pid, camid = batch
                img = img.to(device)
                feat = reidModel(img)
                query_feats.append(feat)
                query_pids.extend(np.asarray(pid))
        
        query_feats = track_torch_pose.cat(query_feats, dim=0)
        query_feats = F.normalize(query_feats, dim=1, p=2)
        
        print(f"✓ Reid模型初始化完成，查询图像数量: {len(query_feats)}")
        return reidModel, query_feats
        
    except Exception as e:
        print(f"Reid模型初始化失败: {e}")
        return None, None

def detect_with_oak(
        model_path='yolov8s.pt',
        dist_thres=1.5,  # ReID距离阈值
        show=False,
        save=True,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        conf_thres=0.5,  # 检测置信度阈值
        device=None  # 自动选择设备
        ):
    """
    使用OAK相机进行人员检测和ReID搜索（YOLO在主机运行）
    """
    import track_torch_pose  # 确保torch在函数开始就被导入
    
    # 设备选择
    if device is None:
        device = track_torch_pose.device('cuda:0' if track_torch_pose.cuda.is_available() else 'cpu')
    else:
        device = track_torch_pose.device(device)
    
    print(f"使用计算设备: {device}")
    
    # 创建保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载YOLO模型
    print("正在加载YOLO模型...")
    try:
        model = YOLO(model_path)
        model.to(device)
        print("✓ YOLO模型加载完成")
    except Exception as e:
        print(f"YOLO模型加载失败: {e}")
        return

    print("正在创建OAK相机管道...")
    try:
        # 创建相机管道 (现在包含深度)
        pipeline = create_camera_pipeline()
        
        # 连接设备并启动管道
        with dai.Device(pipeline) as oak_device:
            print(f"✓ 已连接OAK设备: {oak_device.getDeviceName()}")
            print(f"连接的相机: {oak_device.getConnectedCameraFeatures()}")
            print(f"USB速度: {oak_device.getUsbSpeed().name}")
            
            # 获取输出队列
            qRgb = oak_device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qDepth = oak_device.getOutputQueue(name="depth", maxSize=4, blocking=False) # <-- 修改点: 获取深度队列
            
            print("启动OAK相机目标选择...")
            
            # 使用OAK相机进行实时目标选择
            selected_bboxes, selected_frames = interactive_target_selection_oak(oak_device, qRgb, model, conf_thres)
            
            if not selected_bboxes or not selected_frames:
                print("未选择任何目标，程序退出")
                return
            
            print(f"✓ 已选择 {len(selected_bboxes)} 个目标")
            
            # 保存查询图像
            if not save_query_images(selected_frames, selected_bboxes):
                print("错误：查询图像保存失败")
                return
            
            # 初始化Reid模型
            reidModel, query_feats = initialize_reid_model(device)
            if reidModel is None or query_feats is None:
                return
            
            # 开始实时检索
            print("开始使用OAK相机进行实时人员检索...")
            
            frame_count = 0
            fps_counter = time.time()
            depth_frame = None  # 初始化深度帧变量
            
            while True:
                # 获取RGB和深度帧
                inRgb = qRgb.tryGet()
                inDepth = qDepth.tryGet()

                # /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/--/-/-/- 新增/修改点 /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
                if inDepth is not None:
                    # 获取原始深度帧并进行处理以匹配RGB帧
                    raw_depth = inDepth.getFrame()
                    # 调整大小以匹配RGB预览 (640x480)
                    if raw_depth.shape != (480, 640):
                         raw_depth = cv2.resize(raw_depth, (640, 480), interpolation=cv2.INTER_NEAREST)
                    # 应用中值滤波来减少噪声
                    depth_frame = cv2.medianBlur(raw_depth.astype(np.float32), 5).astype(np.uint16)
                # /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/--/-/-/- 新增/修改点结束 /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                    frame_count += 1
                    
                    # 使用YOLO检测人员
                    person_boxes = detect_all_persons(frame, model, conf_thres)
                    
                    if person_boxes and len(person_boxes) > 0:
                        gallery_img = []
                        gallery_loc = []
                        
                        # 提取所有检测到的人员特征
                        for xmin, ymin, xmax, ymax in person_boxes:
                            w, h = xmax - xmin, ymax - ymin
                            if w * h > 1000:  # 过滤太小的目标
                                gallery_loc.append((xmin, ymin, xmax, ymax))
                                crop_img = frame[ymin:ymax, xmin:xmax]
                                crop_img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                                crop_img_tensor = build_transforms(reidCfg)(crop_img_pil).unsqueeze(0)
                                gallery_img.append(crop_img_tensor)
                        
                        if gallery_img and len(gallery_loc) > 0:
                            # 计算ReID特征
                            gallery_img = track_torch_pose.cat(gallery_img, dim=0).to(device)
                            
                            with track_torch_pose.no_grad():
                                gallery_feats = reidModel(gallery_img)
                                gallery_feats = F.normalize(gallery_feats, dim=1, p=2)
                                
                                # 计算距离矩阵
                                m, n = query_feats.shape[0], gallery_feats.shape[0]
                                distmat = track_torch_pose.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                                          track_torch_pose.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                                
                                distmat.addmm_(1, -2, query_feats, gallery_feats.t())
                                distmat = distmat.cpu().numpy()
                                
                                # /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/--/-/-/- 修改点 /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
                                # 对每个query找到最佳匹配，并记录下来
                                all_matches = []
                                for q_idx in range(m):
                                    best_gallery_idx = np.argmin(distmat[q_idx])
                                    all_matches.append({
                                        'dist': distmat[q_idx][best_gallery_idx],
                                        'gallery_idx': best_gallery_idx
                                    })
                                
                                # 找到所有query中距离最近的那个作为最终匹配结果
                                best_match = min(all_matches, key=lambda x: x['dist'])
                                
                                # 如果最佳匹配小于阈值，则认为是同一个人
                                if best_match['dist'] < dist_thres:
                                    bbox = gallery_loc[best_match['gallery_idx']]
                                    label = f'Target Found! Dist: {best_match["dist"]:.3f}'
                                    
                                    # 计算并添加深度信息到标签
                                    if depth_frame is not None:
                                        center_x = (bbox[0] + bbox[2]) / 2
                                        center_y = (bbox[1] + bbox[3]) / 2
                                        w = bbox[2] - bbox[0]
                                        h = bbox[3] - bbox[1]
                                        
                                        coords = calculate_3d_coordinates(depth_frame, (center_x, center_y), (w, h))
                                        
                                        if coords != (0, 0, 0):
                                            x_cam, y_cam, z_cam = coords
                                            # z_cam 是到相机的直线距离
                                            distance_m = z_cam
                                            label += f" Depth: {distance_m:.2f}m"

                                    plot_one_box(bbox, frame, label=label, color=(0, 255, 0), line_thickness=3)
                                # /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/--/-/-/- 修改点结束 /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/

                                # 显示距离信息
                                best_distance = best_match['dist']
                                cv2.rectangle(frame, (10, 60), (300, 90), (0, 0, 0), -1)
                                distance_text = f"Min Distance: {best_distance:.3f}"
                                cv2.putText(frame, distance_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                
                                # 显示阈值对比
                                threshold_text = f"Threshold: {dist_thres:.3f}"
                                cv2.putText(frame, threshold_text, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                # 匹配状态指示
                                if best_distance < dist_thres:
                                    match_text = "MATCH!"
                                    color = (0, 255, 0)
                                else:
                                    match_text = "NO MATCH"
                                    color = (0, 0, 255)
                                cv2.putText(frame, match_text, (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if frame is not None:
                    # 显示状态信息
                    status_text = f"OAK Frame: {frame_count} | Searching for target..."
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    
                    # 显示FPS
                    if frame_count % 30 == 0 and frame_count > 0:
                        current_time = time.time()
                        fps = 30 / (current_time - fps_counter)
                        fps_counter = current_time
                        print(f"OAK处理速度: {fps:.1f} FPS")
                    
                    # 显示结果
                    cv2.imshow('OAK Person ReID Search', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户退出")
                        break
                    elif key == ord('s'):
                        # 保存当前帧
                        save_frame_path = f"oak_frame_{frame_count}.jpg"
                        cv2.imwrite(save_frame_path, frame)
                        print(f"已保存当前帧: {save_frame_path}")
    
    except Exception as e:
        import traceback
        print(f"OAK相机检测过程中出错: {e}")
        traceback.print_exc()
        return
    finally:
        cv2.destroyAllWindows()
        print(f"结果已保存到: {save_dir}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Person ReID Search with OAK Camera')
    parser.add_argument('--model-path', type=str, default='yolov8s.pt', 
                       help='YOLOv8 model path (.pt format)')
    parser.add_argument('--dist-thres', type=float, default=1.15, help='distance threshold')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--device', type=str, default=None, help='device (cpu/cuda:0)')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--save', action='store_true', default=True, help='save results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    detect_with_oak(
        model_path=args.model_path,
        dist_thres=args.dist_thres,
        conf_thres=args.conf_thres,
        device=args.device,
        show=args.show,
        save=args.save
    )