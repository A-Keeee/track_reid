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

from extended_kalman_filter import EnhancedEKF3D

def generate_complex_person_trajectory(num_steps, dt):
    """生成极其复杂的行人轨迹，包含各种急转弯、螺旋、8字形等"""
    print("生成极其复杂的行人轨迹...")
    trajectory = []
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 1.5, 0.0])  # 初始速度 m/s
    base_speed = 1.5

    # 定义更复杂的行走模式：('模式', 持续步数, '参数-可选')
    maneuvers = [
        ('straight', 30, None),
        # S-弯道
        ('sharp_turn_right', 10, None),  # 90度右急转
        ('straight', 15, 0.8),           # 慢速前进一小段
        ('sharp_turn_left', 10, None),   # 90度左急转
        # 加速与减速
        ('fast_straight', 40, 1.6),      # 加速
        ('slow_straight', 20, 0.7),      # 减速
        # 长而平缓的弯道
        ('turn_left', 40, None),
        # 犹豫和停顿
        ('hesitate', 20, 0.1),           # 速度降至几乎为0
        ('straight', 10, 0.9),           # 重新起步
        # U型转弯
        ('u_turn_right', 20, None),      # 180度大转弯
        # 返回路径
        ('fast_straight', 50, 1.5),
        # 连续小幅度变向
        ('sharp_turn_left', 5, None),
        ('straight', 10, None),
        ('sharp_turn_right', 5, None),
        ('straight', 40, None),
    ]

    total_steps = sum(duration for _, duration, _ in maneuvers)
    if num_steps > total_steps:
        maneuvers.append(('straight', num_steps - total_steps, None))

    spiral_radius = 3.0
    spiral_center = None
    
    for maneuver, duration, param in maneuvers:
        for step in range(duration):
            if 'spiral' in maneuver:
                if spiral_center is None:
                    spiral_center = pos.copy()
                
                # 螺旋运动
                progress = step / duration
                if 'spiral_in' in maneuver:
                    radius = spiral_radius * (1 - 0.7 * progress)
                    angle_speed = 0.3  # 角速度
                else:  # spiral_out
                    radius = spiral_radius * (0.3 + 0.7 * progress)
                    angle_speed = 0.25
                
                angle = angle_speed * step
                target_pos = spiral_center + np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0
                ])
                direction = target_pos - pos
                if np.linalg.norm(direction) > 0:
                    vel[:2] = direction[:2] / np.linalg.norm(direction[:2]) * base_speed
                    
            elif 'figure_eight' in maneuver:
                # 8字形运动
                t = step / duration * 2 * np.pi
                if 'left' in maneuver:
                    eight_x = 2 * np.sin(t)
                    eight_y = np.sin(2 * t)
                else:
                    eight_x = -2 * np.sin(t)
                    eight_y = -np.sin(2 * t)
                
                # 计算切线方向作为速度方向
                vel_x = 2 * np.cos(t) / duration * 2 * np.pi
                vel_y = 2 * np.cos(2 * t) / duration * 2 * np.pi
                vel_norm = np.sqrt(vel_x**2 + vel_y**2)
                if vel_norm > 0:
                    vel[:2] = np.array([vel_x, vel_y]) / vel_norm * base_speed
                    
            elif 'zigzag' in maneuver:
                # Z字形运动
                angle_deg = 45 if 'right' in maneuver else -45
                angle = np.deg2rad(angle_deg)
                rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                     [np.sin(angle), np.cos(angle)]])
                vel[:2] = rot_matrix @ vel[:2]
                
            elif 'jitter' in maneuver:
                # 抖动行走 - 小幅度随机转向
                jitter_angle = np.random.normal(0, 0.3)  # 随机角度
                rot_matrix = np.array([[np.cos(jitter_angle), -np.sin(jitter_angle)], 
                                     [np.sin(jitter_angle), np.cos(jitter_angle)]])
                vel[:2] = rot_matrix @ vel[:2]
                
            elif 'turn' in maneuver or 'u_turn' in maneuver:
                if 'sharp' in maneuver:
                    angle_deg = 15.0  # 每步15度，形成急转弯
                elif 'u_turn_execute' in maneuver:
                    angle_deg = 12.0  # U转时每步12度
                else:
                    angle_deg = 4.5   # 普通转弯
                    
                angle = np.deg2rad(angle_deg)
                if 'left' in maneuver:
                    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                         [np.sin(angle), np.cos(angle)]])
                else:  # right
                    angle = -angle
                    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                         [np.sin(angle), np.cos(angle)]])
                vel[:2] = rot_matrix @ vel[:2]
            
            # 处理速度变化
            if any(keyword in maneuver for keyword in ['straight', 'emergency', 'slow', 'fast', 'final', 'preparation', 'exit']):
                speed_multiplier = param if param is not None else 1.0
                current_speed = np.linalg.norm(vel[:2])
                if current_speed > 0:
                    vel[:2] = vel[:2] / current_speed * base_speed * speed_multiplier
            
            # 更新位置
            pos += vel * dt
            trajectory.append(pos.copy())
            
    return np.array(trajectory), maneuvers

def plot_enhanced_simulation(person_true_path, person_measurements, filtered_person_path, dog_path, lost_periods, angular_velocities=None):
    """可视化增强版EKF模拟结果"""
    if not MATPLOTLIB_AVAILABLE:
        print("📊 matplotlib未安装，无法绘图")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 第一个子图：轨迹图
    ax1.plot(person_true_path[:, 0], person_true_path[:, 1], 'g-', linewidth=3, label='Person True Path')
    ax1.plot(person_measurements[:, 0], person_measurements[:, 1], 'rx', markersize=3, alpha=0.4, label='Person Measurements')
    ax1.plot(filtered_person_path[:, 0], filtered_person_path[:, 1], 'b--', linewidth=2.5, label='Person Filtered Path (Enhanced EKF)')
    ax1.plot(dog_path[:, 0], dog_path[:, 1], 'c-', linewidth=2, label='Robot Dog Path')

    ax1.plot(person_true_path[0, 0], person_true_path[0, 1], 'go', markersize=10, label='Person Start')
    ax1.plot(person_true_path[-1, 0], person_true_path[-1, 1], 'gs', markersize=10, label='Person End')
    ax1.plot(dog_path[0, 0], dog_path[0, 1], 'co', markersize=10, label='Dog Start')
    ax1.plot(dog_path[-1, 0], dog_path[-1, 1], 'cs', markersize=10, label='Dog End')

    # 标记目标丢失区域
    lost_legend_added = False
    for start, end in lost_periods:
        end_idx = min(end, len(person_true_path) - 1)
        start_idx = min(start, end_idx)
        
        label = 'Target Lost Period' if not lost_legend_added else ""
        x_coords = person_true_path[start_idx:end_idx+1, 0]
        y_coords = person_true_path[start_idx:end_idx+1, 1]
        
        if len(x_coords) > 1:
            # 绘制丢失区域
            for i in range(len(x_coords)-1):
                ax1.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                        'r-', linewidth=6, alpha=0.3, label=label if i == 0 and not lost_legend_added else "")
            lost_legend_added = True
        
        text_pos_x = np.mean(x_coords)
        text_pos_y = np.mean(y_coords) + 1.5
        ax1.text(text_pos_x, text_pos_y, 'LOST', fontsize=10, color='red', ha='center', weight='bold')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Enhanced EKF Robot Dog Following Complex Trajectory')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.axis('equal')
    
    # 第二个子图：角速度图
    if angular_velocities is not None:
        time_steps = np.arange(len(angular_velocities)) * 0.1  # 假设dt=0.1
        ax2.plot(time_steps, np.rad2deg(angular_velocities), 'purple', linewidth=2, label='Angular Velocity (deg/s)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (deg/s)')
        ax2.set_title('Estimated Angular Velocity Over Time')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'Angular Velocity Data\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Angular Velocity Plot')

    plt.tight_layout()
    plt.savefig('enhanced_ekf_dog_following_simulation.png', dpi=150)
    plt.show()
    print("\n📊 增强版EKF模拟结果图已保存为 enhanced_ekf_dog_following_simulation.png")


# def plot_person_true_path(person_true_path):
#     """单独显示行人原始路径"""
#     if not MATPLOTLIB_AVAILABLE:
#         print("📊 matplotlib未安装，无法绘图")
#         return
#     plt.figure(figsize=(8, 8))
#     plt.plot(person_true_path[:, 0], person_true_path[:, 1], 'g-', linewidth=3, label='Person True Path')
#     plt.plot(person_true_path[0, 0], person_true_path[0, 1], 'go', markersize=10, label='Start')
#     plt.plot(person_true_path[-1, 0], person_true_path[-1, 1], 'gs', markersize=10, label='End')
#     plt.xlabel('X (m)')
#     plt.ylabel('Y (m)')
#     plt.title('Person True Path (Only)')
#     plt.legend()
#     plt.grid(True)
#     plt.axis('equal')
#     plt.savefig('person_true_path_only.png', dpi=150)
#     plt.show()
#     print("\n📊 行人原始路径图已保存为 person_true_path_only.png")


def plot_all_in_one(person_true_path, person_measurements, filtered_person_path, dog_path, lost_periods, angular_velocities=None):
    """三图合一：原始路径、主轨迹、角速度"""
    if not MATPLOTLIB_AVAILABLE:
        print("📊 matplotlib未安装，无法绘图")
        return
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(24, 7))
    # 1. 原始路径
    ax0.plot(person_true_path[:, 0], person_true_path[:, 1], 'g-', linewidth=3, label='Person True Path')
    ax0.plot(person_true_path[0, 0], person_true_path[0, 1], 'go', markersize=10, label='Start')
    ax0.plot(person_true_path[-1, 0], person_true_path[-1, 1], 'gs', markersize=10, label='End')
    ax0.set_xlabel('X (m)')
    ax0.set_ylabel('Y (m)')
    ax0.set_title('Person True Path (Only)')
    ax0.legend()
    ax0.grid(True)
    ax0.axis('equal')
    # 2. 主轨迹
    ax1.plot(person_true_path[:, 0], person_true_path[:, 1], 'g-', linewidth=3, label='Person True Path')
    ax1.plot(person_measurements[:, 0], person_measurements[:, 1], 'rx', markersize=3, alpha=0.4, label='Person Measurements')
    ax1.plot(filtered_person_path[:, 0], filtered_person_path[:, 1], 'b--', linewidth=2.5, label='Person Filtered Path (Enhanced EKF)')
    ax1.plot(dog_path[:, 0], dog_path[:, 1], 'c-', linewidth=2, label='Robot Dog Path')
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
            for i in range(len(x_coords)-1):
                ax1.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                        'r-', linewidth=6, alpha=0.3, label=label if i == 0 and not lost_legend_added else "")
            lost_legend_added = True
        text_pos_x = np.mean(x_coords)
        text_pos_y = np.mean(y_coords) + 1.5
        ax1.text(text_pos_x, text_pos_y, 'LOST', fontsize=10, color='red', ha='center', weight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Enhanced EKF Robot Dog Following')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.axis('equal')
    # 3. 角速度
    if angular_velocities is not None:
        time_steps = np.arange(len(angular_velocities)) * 0.1
        ax2.plot(time_steps, np.rad2deg(angular_velocities), 'purple', linewidth=2, label='Angular Velocity (deg/s)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (deg/s)')
        ax2.set_title('Estimated Angular Velocity Over Time')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'Angular Velocity Data\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Angular Velocity Plot')
    plt.tight_layout()
    plt.savefig('all_in_one_ekf_simulation.png', dpi=150)
    plt.show()
    print("\n📊 三图合一结果图已保存为 all_in_one_ekf_simulation.png")


def main():
    """主模拟函数 - 使用增强版EKF测试复杂轨迹跟踪"""
    print("=== 测试增强版EKF (包含角速度) 跟踪复杂行人轨迹 ===")
    
    np.random.seed(42)
    dt = 0.1
    
    # 1. 生成复杂的行人轨迹
    person_true_trajectory, maneuvers = generate_complex_person_trajectory(num_steps=500, dt=dt)
    num_steps = len(person_true_trajectory)
    
    # 2. 创建丢失计划 - 在复杂机动中丢失
    print("创建与复杂机动耦合的丢失计划...")
    loss_schedule = np.zeros(num_steps, dtype=bool)
    lost_periods_for_plot = []
    
    current_step = 0
    
    for maneuver, duration, _ in maneuvers:
        # 在复杂机动（螺旋、8字形、急转弯）中模拟目标丢失
        if any(keyword in maneuver for keyword in ['spiral', 'figure_eight', 'sharp_turn', 'zigzag', 'u_turn_execute']):
            loss_start = current_step + duration // 3
            loss_end = current_step + 2 * duration // 3
            
            loss_schedule[loss_start:loss_end] = True
            lost_periods_for_plot.append((loss_start, loss_end - 1))
        
        current_step += duration

    # 3. 初始化增强版EKF
    ekf = EnhancedEKF3D(
        process_noise_std=1.0,
        measurement_noise_std=10.0,
        initial_velocity_std=0.1,
        initial_acceleration_std=0.5,
        initial_angular_velocity_std=0.4
    )
    
    # 4. 模拟参数
    follow_distance = 1.8  # 机器狗跟随距离
    measurement_noise_std = 0.15
    
    # 存储历史数据
    person_measurements = []
    filtered_person_positions = []
    dog_positions = []
    angular_velocities = []  # 记录角速度
    
    print("\n开始模拟...")
    # 5. 主循环
    for i in range(num_steps):
        t = i * dt
        person_true_pos = person_true_trajectory[i]
        
        # 从预先生成的计划中获取当前是否丢失
        is_lost = loss_schedule[i]
        
        if is_lost and (i == 0 or not loss_schedule[i-1]):
             print(f"[{t:.1f}s] ❗ 目标在复杂机动中丢失... 开始增强预测。")
        if not is_lost and i > 0 and loss_schedule[i-1]:
             print(f"[{t:.1f}s] ✅ 目标重新捕获！恢复正常跟踪。")

        # 模拟带噪声的测量
        measurement_noise = np.random.normal(0, measurement_noise_std, 3)
        person_measurement = person_true_pos + measurement_noise
        person_measurements.append(person_measurement)
        
        # 增强EKF滤波过程
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
        
        # 记录角速度
        angular_vel = ekf.get_current_angular_velocity()
        angular_velocities.append(angular_vel)
        
        # 计算机器狗的位置 - 基于增强的速度和角度信息
        person_velocity_vec = np.array(ekf.get_current_velocity())
        person_speed = np.linalg.norm(person_velocity_vec)
        person_orientation = ekf.get_current_orientation()
        
        if person_speed < 0.1:
            if len(dog_positions) > 0:
                 dog_pos = dog_positions[-1]
            else:
                 dog_pos = filtered_person_pos - np.array([0, follow_distance, 0])
        else:
            # 使用预测的运动方向来定位机器狗
            direction_vec = person_velocity_vec / person_speed
            
            # 考虑角速度进行更平滑的跟随
            if abs(angular_vel) > 0.05:  # 如果在转弯
                # 在转弯时，机器狗应该走内侧弧线
                turn_compensation = follow_distance * 0.3 * np.sign(angular_vel)
                perpendicular_vec = np.array([-direction_vec[1], direction_vec[0], 0])
                dog_pos = filtered_person_pos - follow_distance * direction_vec + turn_compensation * perpendicular_vec
            else:
                dog_pos = filtered_person_pos - follow_distance * direction_vec
        
        dog_positions.append(dog_pos)
        
    print("增强版EKF模拟完成！")
    
    person_measurements = np.array(person_measurements)
    filtered_person_positions = np.array(filtered_person_positions)
    dog_positions = np.array(dog_positions)
    angular_velocities = np.array(angular_velocities)
    
    # 6. 打印统计信息
    print(f"\n📊 模拟统计:")
    print(f"   总步数: {num_steps}")
    print(f"   平均角速度: {np.mean(np.abs(angular_velocities)):.3f} rad/s ({np.rad2deg(np.mean(np.abs(angular_velocities))):.1f} deg/s)")
    print(f"   最大角速度: {np.max(np.abs(angular_velocities)):.3f} rad/s ({np.rad2deg(np.max(np.abs(angular_velocities))):.1f} deg/s)")
    print(f"   丢失时段数量: {len(lost_periods_for_plot)}")
    
    # 7. 可视化结果
    plot_all_in_one(person_true_trajectory, person_measurements, 
                   filtered_person_positions, dog_positions, 
                   lost_periods_for_plot, angular_velocities)
    # # 单独显示行人原始路径
    # plot_person_true_path(person_true_trajectory)


if __name__ == "__main__":
    main()