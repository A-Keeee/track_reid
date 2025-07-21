import os
import glob
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

def preprocess_image(img_path, input_size=(256, 128)):
    """预处理图片，与ReID模型推理时的预处理保持一致"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size).astype(np.float32)
        
        # ReID标准归一化参数
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img = (img / 255.0 - mean) / std
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
        return img.astype(np.float32)
    except Exception as e:
        print(f"预处理图片失败 {img_path}: {e}")
        return None

def get_model_input_info(model_path):
    """获取ONNX模型的输入信息"""
    model = onnx.load(model_path)
    input_info = model.graph.input[0]
    input_name = input_info.name
    input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
    print(f"模型输入名称: {input_name}")
    print(f"模型输入形状: {input_shape}")
    return input_name, input_shape

class ImageDataReader(CalibrationDataReader):
    def __init__(self, dataset_path, input_name, input_size=(256, 128), max_samples=5000):
        """
        dataset_path: Market-1501数据集路径
        input_name: 模型输入层名称
        input_size: 模型输入尺寸 (width, height)
        max_samples: 最大校准样本数量
        """
        self.input_name = input_name
        self.input_size = input_size
        self.enum_data = None
        
        # 收集训练集图片路径
        train_path = os.path.join(dataset_path, "bounding_box_train", "*.jpg")
        image_paths = glob.glob(train_path)
        
        # 限制样本数量，避免校准时间过长
        if len(image_paths) > max_samples:
            image_paths = image_paths[:max_samples]
        
        print(f"找到 {len(image_paths)} 张校准图片")
        
        # 预处理所有图片
        self.calibration_data = []
        for i, img_path in enumerate(image_paths):
            if i % 50 == 0:
                print(f"预处理进度: {i}/{len(image_paths)}")
            
            processed_img = preprocess_image(img_path, input_size)
            if processed_img is not None:
                self.calibration_data.append(processed_img)
        
        print(f"成功预处理 {len(self.calibration_data)} 张校准图片")

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.calibration_data)
        try:
            input_data = next(self.enum_data)
            return {self.input_name: input_data}
        except StopIteration:
            return None

def main():
    model_path = "reidmodel.onnx"
    output_path = "reidmodel_int8_static.onnx"
    dataset_path = "Market-1501-v15.09.15"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集 {dataset_path}")
        return
    
    print("=== ReID模型INT8静态量化 ===")
    
    # 获取模型输入信息
    input_name, input_shape = get_model_input_info(model_path)
    
    # 根据模型输入形状确定图片尺寸 (假设输入为NCHW格式)
    if len(input_shape) == 4:
        input_size = (input_shape[3], input_shape[2])  # (width, height)
    else:
        input_size = (256, 128)  # 默认ReID尺寸
    
    print(f"使用输入尺寸: {input_size}")
    
    # 创建校准数据读取器
    print("准备校准数据...")
    calibration_reader = ImageDataReader(dataset_path, input_name, input_size, max_samples=5000)
    
    # 执行静态量化
    print("开始INT8静态量化...")
    try:
        quantize_static(
            model_path,
            output_path,
            calibration_reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8
        )
        print(f"量化完成! 输出文件: {output_path}")
        
        # 检查文件大小对比
        original_size = os.path.getsize(model_path) / (1024*1024)
        quantized_size = os.path.getsize(output_path) / (1024*1024)
        compression_ratio = original_size / quantized_size
        
        print(f"原始模型大小: {original_size:.2f} MB")
        print(f"量化模型大小: {quantized_size:.2f} MB")
        print(f"压缩比例: {compression_ratio:.2f}x")
        
    except Exception as e:
        print(f"量化失败: {e}")
        print("尝试降级到动态量化...")
        
        # 降级到动态量化
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic(
            model_path,
            "reidmodel_int8_dynamic.onnx",
            weight_type=QuantType.QInt8
        )
        print("动态量化完成: reidmodel_int8_dynamic.onnx")

if __name__ == "__main__":
    main()