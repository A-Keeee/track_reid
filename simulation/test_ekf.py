#!/usr/bin/env python3
# 文件名: test_enhanced_ekf.py
# 描述: 测试增强版EKF (包含角速度) 跟踪复杂行人轨迹

import numpy as np
import time
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("📊 matplotlib未安装，将跳过绘图功能")

# 导入新的增强版EKF
# 假设 extended_kalman_filter.py 文件与此脚本在同一目录下
from extended_kalman_filter import EnhancedEKF3D

def generate_person_trajectory(num_steps, dt):
    """生成一段非常复杂的、包含S弯/U型弯/停顿的行人轨迹"""
    print("生成非常复杂的行人轨迹...")
    trajectory = []
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 1.4, 0.0])
    base_speed = 1.4

    maneuvers = [
        ('straight', 30, None), ('sharp_turn_right', 10, None),
        ('straight', 15, 0.8), ('sharp_turn_left', 10, None),
        ('fast_straight', 40, 1.6), ('slow_straight', 20, 0.7),
        ('turn_left', 40, None), ('hesitate', 20, 0.1),
        ('straight', 10, 0.9), ('u_turn_right', 20, None),
        ('fast_straight', 50, 1.5), ('sharp_turn_left', 5, None),
        ('straight', 10, None), ('sharp_turn_right', 5, None),
        ('straight', 40, None),
    ]

    total_steps = sum(duration for _, duration, _ in maneuvers)
    if num_steps > total_steps:
        maneuvers.append(('straight', num_steps - total_steps, None))

    for maneuver, duration, param in maneuvers:
        for _ in range(duration):
            if 'turn' in maneuver:
                angle_deg = 9.0 if ('sharp' in maneuver or 'u_turn' in maneuver) else 2.5
                angle = np.deg2rad(angle_deg)
                if 'left' in maneuver:
                    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                else:
                    angle = -angle
                    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                vel[:2] = rot_matrix @ vel[:2]
            
            if any(keyword in maneuver for keyword in ['straight', 'hesitate']):
                speed_multiplier = param if param is not None else 1.0
                current_speed = np.linalg.norm(vel[:2])
                if current_speed > 0:
                    vel[:2] = vel[:2] / current_speed * base_speed * speed_multiplier
            
            pos += vel * dt
            trajectory.append(pos.copy())
            
    return np.array(trajectory), maneuvers

def plot_enhanced_simulation(person_true_path, person_measurements, filtered_person_path, dog_path, lost_periods, angular_velocities):
    """可视化增强版EKF模拟结果"""
    if not MATPLOTLIB_AVAILABLE:
        print("📊 matplotlib未安装，无法绘图")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle('Enhanced EKF Simulation: Robot Dog Following Person', fontsize=16)

    # --- 第一个子图：轨迹图 ---
    ax1.plot(person_true_path[:, 0], person_true_path[:, 1], 'g-', linewidth=3, label='Person True Path')
    ax1.plot(person_measurements[:, 0], person_measurements[:, 1], 'rx', markersize=4, alpha=0.5, label='Person Measurements')
    ax1.plot(filtered_person_path[:, 0], filtered_person_path[:, 1], 'b--', linewidth=2.5, label='Person Filtered Path (Enhanced EKF)')
    ax1.plot(dog_path[:, 0], dog_path[:, 1], 'c-', linewidth=2.5, label='Robot Dog Path (Smoothed)')

    ax1.plot(person_true_path[0, 0], person_true_path[0, 1], 'go', markersize=10, label='Person Start')
    ax1.plot(person_true_path[-1, 0], person_true_path[-1, 1], 'gs', markersize=10, label='Person End')
    ax1.plot(dog_path[0, 0], dog_path[0, 1], 'co', markersize=10, label='Dog Start')
    ax1.plot(dog_path[-1, 0], dog_path[-1, 1], 'cs', markersize=10, label='Dog End')

    lost_legend_added = False
    for start, end in lost_periods:
        end_idx = min(end, len(person_true_path) - 1)
        start_idx = min(start, end_idx)
        label = 'Target Lost Period' if not lost_legend_added else ""
        x_coords = person_true_path[start_idx:end_idx+1, 0]
        y_coords = person_true_path[start_idx:end_idx+1, 1]
        
        if len(x_coords) > 1:
            ax1.plot(x_coords, y_coords, color='orange', alpha=0.7, linewidth=8, label=label, solid_capstyle='round')
            lost_legend_added = True
        
        text_pos_x = np.mean(x_coords)
        text_pos_y = np.mean(y_coords) + 1.5
        ax1.text(text_pos_x, text_pos_y, 'LOST', fontsize=10, color='red', ha='center', weight='bold')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory Overview')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.axis('equal')
    
    # --- 第二个子图：角速度图 ---
    time_steps = np.arange(len(angular_velocities)) * 0.1
    ax2.plot(time_steps, np.rad2deg(angular_velocities), 'purple', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (deg/s)')
    ax2.set_title('Estimated Angular Velocity (ω)')
    ax2.grid(True)
    # 用灰色区域标记丢失时段
    for start, end in lost_periods:
        ax2.axvspan(start * 0.1, end * 0.1, color='orange', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('enhanced_ekf_simulation.png', dpi=150)
    plt.show()
    print("\n📊 增强版EKF模拟结果图已保存为 enhanced_ekf_simulation.png")


def main():
    """主模拟函数 - 使用增强版EKF测试复杂轨迹跟踪"""
    print("=== 模拟机器狗使用增强版EKF跟随行人 (包含角速度) ===")
    
    np.random.seed(42)
    dt = 0.1
    
    person_true_trajectory, maneuvers = generate_person_trajectory(num_steps=400, dt=dt)
    num_steps = len(person_true_trajectory)
    
    print("创建与机动动作耦合的丢失计划...")
    loss_schedule = np.zeros(num_steps, dtype=bool)
    lost_periods_for_plot = []
    
    current_step = 0
    for maneuver, duration, _ in maneuvers:
        if 'turn' in maneuver:
            turn_midpoint = duration // 2
            lost_start_global = current_step + turn_midpoint
            lost_end_global = current_step + duration
            loss_schedule[lost_start_global:lost_end_global] = True
            lost_periods_for_plot.append((lost_start_global, lost_end_global - 1))
        current_step += duration

    # 3. 初始化增强版EKF
    ekf = EnhancedEKF3D(
        process_noise_std=0.5,
        measurement_noise_std=5.0,
        initial_velocity_std=1.5,
        initial_acceleration_std=1.0,
        initial_angular_velocity_std=np.deg2rad(30) # 初始角速度不确定性
    )
    
    # 4. 模拟参数
    follow_distance = 1.5
    measurement_noise_std = 0.1
    
    # 存储历史数据
    person_measurements = []
    filtered_person_positions = []
    dog_positions = []
    angular_velocities = []

    print("\n开始模拟...")
    # 5. 主循环
    for i in range(num_steps):
        t = i * dt
        person_true_pos = person_true_trajectory[i]
        
        is_lost = loss_schedule[i]
        
        if is_lost and (i > 0 and not loss_schedule[i-1]):
             print(f"[{t:.1f}s] ❗ 目标在转弯中丢失... 开始预测。")
        if not is_lost and (i > 0 and loss_schedule[i-1]):
             print(f"[{t:.1f}s] ✅ 目标重新捕获！恢复正常跟踪。")

        measurement_noise = np.random.normal(0, measurement_noise_std, 3)
        person_measurement = person_true_pos + measurement_noise
        person_measurements.append(person_measurement)
        
        if not ekf.is_initialized():
            ekf.initialize(person_measurement, t)
        else:
            if is_lost:
                ekf.handle_lost_target(t)
            else:
                ekf.predict(t)
                ekf.update(person_measurement)
        
        filtered_person_pos = np.array(ekf.get_current_position())
        filtered_person_positions.append(filtered_person_pos)
        
        # 记录角速度用于绘图
        angular_velocities.append(ekf.get_current_angular_velocity())
        
        # ==================== 基于方向角(theta)的跟随逻辑 ====================
        # 以下逻辑已正确实现“在正后方1.5m处跟随”
        person_orientation = ekf.get_current_orientation()
        
        # 方向向量直接由theta计算，比速度向量更稳定
        direction_vec = np.array([np.sin(person_orientation), np.cos(person_orientation), 0])
        # 注意：这里假设theta=0时朝向X轴正方向，需要根据你的EKF定义调整
        # 如果你的EKF定义theta=0朝向Y轴，则direction_vec = np.array([np.sin(theta), np.cos(theta), 0])
        
        # 目标点位置 = 人的位置 - 1.5米 * 人的朝向向量
        # 这个计算结果就是人正后方1.5米的位置
        target_dog_pos = filtered_person_pos - follow_distance * direction_vec
        
        # 对狗的位置进行平滑处理，使其运动更自然
        if len(dog_positions) == 0:
            current_dog_pos = target_dog_pos
        else:
            smoothing_factor = 0.4
            last_dog_pos = dog_positions[-1]
            current_dog_pos = (1 - smoothing_factor) * last_dog_pos + smoothing_factor * target_dog_pos
            
        dog_positions.append(current_dog_pos)
        
    print("模拟完成！")
    
    person_measurements = np.array(person_measurements)
    filtered_person_positions = np.array(filtered_person_positions)
    dog_positions = np.array(dog_positions)
    angular_velocities = np.array(angular_velocities)
    
    # 6. 可视化结果
    plot_enhanced_simulation(person_true_trajectory, person_measurements, filtered_person_positions, dog_positions, lost_periods_for_plot, angular_velocities)


if __name__ == "__main__":
    main()
    