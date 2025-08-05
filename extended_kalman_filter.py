import numpy as np
import time
from typing import Tuple, Optional


class ExtendedKalmanFilter3D:
    """
    三维坐标的扩展卡尔曼滤波器 - 匀加速运动模型
    状态向量: [x, y, z, vx, vy, vz, ax, ay, az]
    观测向量: [x, y, z]
    """
    
    def __init__(self, 
                 process_noise_std: float = 0.1,
                 measurement_noise_std: float = 0.2,
                 initial_velocity_std: float = 0.5,
                 initial_acceleration_std: float = 0.2):
        """
        初始化扩展卡尔曼滤波器
        
        Args:
            process_noise_std: 过程噪声标准差
            measurement_noise_std: 测量噪声标准差
            initial_velocity_std: 初始速度不确定性标准差
            initial_acceleration_std: 初始加速度不确定性标准差
        """
        # 状态维度: 9 (x, y, z, vx, vy, vz, ax, ay, az)
        self.state_dim = 9
        # 观测维度: 3 (x, y, z)
        self.obs_dim = 3
        
        # 状态向量 [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros((self.state_dim, 1))
        
        # 状态协方差矩阵
        self.P = np.eye(self.state_dim)
        self.P[3:6, 3:6] *= initial_velocity_std**2  # 速度的初始不确定性
        self.P[6:9, 6:9] *= initial_acceleration_std**2  # 加速度的初始不确定性
        
        # 过程噪声协方差矩阵 Q
        self.Q = np.eye(self.state_dim) * process_noise_std**2
        self.Q[3:6, 3:6] *= 2  # 速度的过程噪声稍大
        self.Q[6:9, 6:9] *= 5  # 加速度的过程噪声更大
        
        # 测量噪声协方差矩阵 R
        self.R = np.eye(self.obs_dim) * measurement_noise_std**2
        
        # 观测矩阵 H (线性的，只观测位置)
        self.H = np.zeros((self.obs_dim, self.state_dim))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # z
        
        # 时间相关
        self.last_time = None
        self.initialized = False
        
        # 预测相关
        self.prediction_horizon = 0.5  # 预测未来0.5秒的位置
        
        # 丢失目标处理
        self.lost_count = 0
        self.max_lost_count = 15
        
    def initialize(self, measurement: np.ndarray, timestamp: float):
        """
        使用第一个测量值初始化滤波器
        
        Args:
            measurement: 测量值 [x, y, z]
            timestamp: 时间戳
        """
        if len(measurement) != 3:
            raise ValueError("测量值必须是3维坐标 [x, y, z]")
            
        # 初始化位置，速度和加速度为0
        self.x[0:3, 0] = measurement.flatten()
        self.x[3:6, 0] = 0  # 初始速度为0
        self.x[6:9, 0] = 0  # 初始加速度为0
        
        self.last_time = timestamp
        self.initialized = True
        self.lost_count = 0
        
        print(f"EKF初始化完成: 位置 [{measurement[0]:.2f}, {measurement[1]:.2f}, {measurement[2]:.2f}]")
        
    def predict(self, timestamp: float) -> np.ndarray:
        """
        预测步骤 - 匀加速运动模型
        
        Args:
            timestamp: 当前时间戳
            
        Returns:
            预测的状态向量
        """
        if not self.initialized:
            return self.x.flatten()
            
        # 计算时间差
        dt = timestamp - self.last_time if self.last_time is not None else 0.033  # 默认30fps
        dt = max(0.001, min(dt, 0.2))  # 限制dt在合理范围内
        
        # 状态转移矩阵 F (匀加速度模型)
        F = np.eye(self.state_dim)
        # 位置更新: x = x + vx*dt + 0.5*ax*dt^2
        F[0, 3] = dt     # x <- vx
        F[1, 4] = dt     # y <- vy  
        F[2, 5] = dt     # z <- vz
        F[0, 6] = 0.5 * dt**2  # x <- ax
        F[1, 7] = 0.5 * dt**2  # y <- ay
        F[2, 8] = 0.5 * dt**2  # z <- az
        
        # 速度更新: vx = vx + ax*dt
        F[3, 6] = dt     # vx <- ax
        F[4, 7] = dt     # vy <- ay
        F[5, 8] = dt     # vz <- az
        
        # 加速度保持不变 (随机游走模型)
        # F[6:9, 6:9] 已经是单位矩阵
        
        # 预测状态
        self.x = F @ self.x
        
        # 预测协方差
        self.P = F @ self.P @ F.T + self.Q
        
        self.last_time = timestamp
        
        return self.x.flatten()
        
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        更新步骤
        
        Args:
            measurement: 测量值 [x, y, z]
            
        Returns:
            更新后的状态向量
        """
        if not self.initialized:
            raise RuntimeError("滤波器未初始化，请先调用initialize()")
            
        if len(measurement) != 3:
            raise ValueError("测量值必须是3维坐标 [x, y, z]")
            
        z = measurement.reshape(-1, 1)
        
        # 计算新息 (innovation)
        y = z - self.H @ self.x
        
        # 新息协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.x = self.x + K @ y
        
        # 更新协方差
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        # 重置丢失计数
        self.lost_count = 0
        
        return self.x.flatten()
        
    def predict_future_position(self, time_ahead: Optional[float] = None) -> Tuple[float, float, float]:
        """
        预测未来某个时刻的位置 - 基于匀加速运动模型
        
        Args:
            time_ahead: 预测多少秒后的位置，如果为None则使用默认值
            
        Returns:
            预测的(x, y, z)坐标
        """
        if not self.initialized:
            return (0.0, 0.0, 0.0)
            
        if time_ahead is None:
            time_ahead = self.prediction_horizon
            
        # 使用当前状态、速度和加速度预测未来位置
        # x_future = x + v*t + 0.5*a*t^2
        dt = time_ahead
        future_x = self.x[0, 0] + self.x[3, 0] * dt + 0.5 * self.x[6, 0] * dt**2
        future_y = self.x[1, 0] + self.x[4, 0] * dt + 0.5 * self.x[7, 0] * dt**2
        future_z = self.x[2, 0] + self.x[5, 0] * dt + 0.5 * self.x[8, 0] * dt**2
        
        return (float(future_x), float(future_y), float(future_z))
        
    def get_current_position(self) -> Tuple[float, float, float]:
        """
        获取当前滤波后的位置
        
        Returns:
            当前的(x, y, z)坐标
        """
        if not self.initialized:
            return (0.0, 0.0, 0.0)
            
        return (float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]))
        
    def get_current_velocity(self) -> Tuple[float, float, float]:
        """
        获取当前估计的速度
        
        Returns:
            当前的(vx, vy, vz)速度
        """
        if not self.initialized:
            return (0.0, 0.0, 0.0)
            
        return (float(self.x[3, 0]), float(self.x[4, 0]), float(self.x[5, 0]))
        
    def get_current_acceleration(self) -> Tuple[float, float, float]:
        """
        获取当前估计的加速度
        
        Returns:
            当前的(ax, ay, az)加速度
        """
        if not self.initialized:
            return (0.0, 0.0, 0.0)
            
        return (float(self.x[6, 0]), float(self.x[7, 0]), float(self.x[8, 0]))
        
    def handle_lost_target(self, timestamp: float) -> Optional[Tuple[float, float, float]]:
        """
        处理目标丢失的情况，基于预测继续跟踪
        
        Args:
            timestamp: 当前时间戳
            
        Returns:
            预测的位置，如果超过最大丢失次数则返回None
        """
        if not self.initialized:
            return None
            
        self.lost_count += 1
        
        if self.lost_count > self.max_lost_count:
            print("目标丢失时间过长，停止预测")
            return None
            
        # 仅进行预测步骤，不进行更新
        predicted_state = self.predict(timestamp)
        
        # 增加过程噪声以反映不确定性的增加
        self.P *= 1.1
        
        return self.get_current_position()
        
    def reset(self):
        """
        重置滤波器状态
        """
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim)
        self.P[3:6, 3:6] *= 0.5**2  # 重置速度不确定性
        self.P[6:9, 6:9] *= 0.2**2  # 重置加速度不确定性
        self.last_time = None
        self.initialized = False
        self.lost_count = 0
        print("EKF已重置")
        
    def is_initialized(self) -> bool:
        """
        检查滤波器是否已初始化
        
        Returns:
            是否已初始化
        """
        return self.initialized
        
    def get_position_uncertainty(self) -> float:
        """
        获取位置估计的不确定性（协方差矩阵的迹）
        
        Returns:
            位置不确定性
        """
        if not self.initialized:
            return float('inf')
            
        # 返回位置协方差的迹
        return np.trace(self.P[0:3, 0:3])
        
    def set_process_noise(self, std: float):
        """
        动态调整过程噪声
        
        Args:
            std: 新的过程噪声标准差
        """
        self.Q = np.eye(self.state_dim) * std**2
        self.Q[3:6, 3:6] *= 2  # 速度的过程噪声稍大
        self.Q[6:9, 6:9] *= 5  # 加速度的过程噪声更大
        
    def set_measurement_noise(self, std: float):
        """
        动态调整测量噪声
        
        Args:
            std: 新的测量噪声标准差
        """
        self.R = np.eye(self.obs_dim) * std**2


class AdaptiveEKF3D(ExtendedKalmanFilter3D):
    """
    自适应扩展卡尔曼滤波器，能够根据跟踪质量动态调整参数
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 自适应参数
        self.innovation_history = []
        self.history_length = 10
        self.base_process_noise = kwargs.get('process_noise_std', 0.1)
        self.base_measurement_noise = kwargs.get('measurement_noise_std', 0.2)
        
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        自适应更新步骤
        """
        if not self.initialized:
            raise RuntimeError("滤波器未初始化，请先调用initialize()")
            
        # 计算新息
        z = measurement.reshape(-1, 1)
        innovation = z - self.H @ self.x
        innovation_magnitude = np.linalg.norm(innovation)
        
        # 记录新息历史
        self.innovation_history.append(innovation_magnitude)
        if len(self.innovation_history) > self.history_length:
            self.innovation_history.pop(0)
            
        # 根据新息调整噪声参数
        if len(self.innovation_history) >= 3:
            avg_innovation = np.mean(self.innovation_history)
            
            # 如果新息较大，增加测量噪声
            if avg_innovation > 0.5:
                noise_factor = min(2.0, avg_innovation)
                self.set_measurement_noise(self.base_measurement_noise * noise_factor)
                self.set_process_noise(self.base_process_noise * noise_factor)
            else:
                # 新息较小时，逐渐恢复默认参数
                self.set_measurement_noise(self.base_measurement_noise)
                self.set_process_noise(self.base_process_noise)
                
        return super().update(measurement)
