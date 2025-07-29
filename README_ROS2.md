# Track Torch ROS2 集成

这个项目提供了将 track_torch.py 计算出的三维坐标发布到 ROS2 pose 话题的功能。

## 文件结构

```
track_torch/
├── track_torch.py              # 主跟踪程序 (已修改，支持坐标导出)
├── start_ros2.sh              # ROS2 集成启动脚本
├── track_torch_ros2/          # ROS2 包
│   ├── __init__.py
│   ├── ros2_pose_publisher.py     # 完整版 ROS2 发布器 (需要 gRPC)
│   └── simple_pose_publisher.py   # 简化版 ROS2 发布器 (通过文件通信)
├── package.xml                # ROS2 包配置
├── setup.py                  # Python 包配置
└── resource/                 # ROS2 资源文件
```

## 功能说明

### 1. 坐标导出机制
- `track_torch.py` 在跟踪到目标时，会将三维坐标导出到 `/tmp/tracking_coords.json`
- 坐标格式: `{"x": 1.23, "y": 0.45, "z": 2.67, "timestamp": 1690123456.789}`

### 2. ROS2 发布器
- **简化版**: `simple_pose_publisher.py` - 通过读取文件获取坐标
- **完整版**: `ros2_pose_publisher.py` - 通过 gRPC 直接获取坐标

### 3. ROS2 话题
- **话题名称**: `/tracking/target_pose`
- **消息类型**: `geometry_msgs/PoseStamped`
- **坐标系**: `camera_link` (可配置)
- **发布频率**: 30 Hz (可配置)

## 使用方法

### 方法一: 使用启动脚本 (推荐)

```bash
# 确保 ROS2 环境已设置
source /opt/ros/humble/setup.bash

# 运行启动脚本
./start_ros2.sh
```

### 方法二: 手动启动

```bash
# 终端1: 启动跟踪程序
conda activate yolo
python track_torch.py --no-viz

# 终端2: 启动 ROS2 发布器
source /opt/ros/humble/setup.bash
python track_torch_ros2/simple_pose_publisher.py
```

### 方法三: 作为 ROS2 包安装 (可选)

```bash
# 构建包
colcon build --packages-select track_torch_ros2

# 运行节点
ros2 run track_torch_ros2 simple_pose_publisher
```

## 订阅 ROS2 话题

```bash
# 查看话题信息
ros2 topic info /tracking/target_pose

# 实时查看坐标数据
ros2 topic echo /tracking/target_pose

# 查看话题频率
ros2 topic hz /tracking/target_pose
```

## 参数配置

### track_torch.py 参数
- `--no-ros-export`: 禁用 ROS2 坐标导出
- `--no-viz`: 禁用可视化界面 (推荐用于 ROS2 集成)
- `--no-grpc`: 禁用 gRPC 通信

### simple_pose_publisher.py 参数
- `publish_rate`: 发布频率 (默认: 30.0 Hz)
- `frame_id`: 坐标系名称 (默认: 'camera_link')
- `coord_file`: 坐标文件路径 (默认: '/tmp/tracking_coords.json')

```bash
# 自定义参数示例
ros2 run track_torch_ros2 simple_pose_publisher --ros-args -p publish_rate:=20.0 -p frame_id:=base_link
```

## 坐标系说明

track_torch.py 输出的坐标系约定：
- **X轴**: 相机前方 (深度方向)
- **Y轴**: 相机左侧
- **Z轴**: 相机上方

如需转换到其他坐标系，可以在 ROS2 中使用 `tf2` 进行坐标变换。

## 故障排除

### 1. 无法找到 ROS2
确保已安装并设置 ROS2 环境：
```bash
source /opt/ros/humble/setup.bash
```

### 2. 坐标文件不存在
确保 track_torch.py 正在运行且已检测到目标。检查文件：
```bash
cat /tmp/tracking_coords.json
```

### 3. 话题无数据
检查坐标文件的时间戳，确保数据是最新的（2秒内）。

### 4. 权限问题
确保启动脚本有执行权限：
```bash
chmod +x start_ros2.sh
```

## 依赖安装

### Python 依赖
```bash
pip install rclpy geometry-msgs std-msgs
```

### 系统依赖
```bash
sudo apt install ros-humble-geometry-msgs ros-humble-std-msgs
```
