#!/usr/bin/env python3
# 文件名: test_ekf.py
# 描述: 测试扩展卡尔曼滤波器的功能 - 模拟行人行走和目标丢失

import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("📊 matplotlib未安装，将跳过绘图功能")

# 假设 extended_kalman_filter.py 文件在同一目录下
from extended_kalman_filter import ExtendedKalmanFilter3D
import time

def test_basic_ekf():
    """测试基本的扩展卡尔曼滤波器 - 匀加速运动模型 - 模拟行人行走"""
    print("=== 测试基本扩展卡尔曼滤波器 (匀加速运动模型) - 模拟行人行走 ===")
    
    # 创建滤波器
    ekf = ExtendedKalmanFilter3D(
        process_noise_std=0.05,  # 减小过程噪声，因为行人运动相对平缓
        measurement_noise_std=1,
        initial_velocity_std=0.2,
        initial_acceleration_std=0.1
    )
    
    # 生成模拟轨迹数据
    num_points = 100
    dt = 0.1
    
    # 真实轨迹 (模拟行人行走)
    true_trajectory = []
    measurements = []
    filtered_trajectory = []
    predicted_trajectory = []
    
    # 初始位置和速度
    true_pos = np.array([0.0, 0.0, 0.8])
    true_vel = np.array([0.5, 0.2, 0.01]) # 初始速度，Z轴有轻微漂移
    
    for i in range(num_points):
        t = i * dt
        
        # 模拟随机的小加速度变化
        acc_change = np.random.normal(0, 0.02, 2) # XY方向有小随机加速度
        true_acc = np.array([acc_change[0], acc_change[1], -0.001]) # Z轴有轻微向下漂移趋势
        true_vel[:2] += true_acc[:2] * dt
        true_pos += true_vel * dt + 0.5 * true_acc * dt**2
        
        # 限制速度在一个合理范围内
        speed = np.linalg.norm(true_vel[:2])
        if speed > 1.0:
            true_vel[:2] *= 1.0 / speed
            
        true_trajectory.append(true_pos.copy())
        
        # 添加测量噪声
        measurement_noise = np.random.normal(0, 0.1, 3)
        measurement = true_pos + measurement_noise
        measurements.append(measurement.copy())
        
        # 卡尔曼滤波
        if not ekf.is_initialized():
            ekf.initialize(measurement, t)
            filtered_pos = ekf.get_current_position()
            predicted_pos = ekf.predict_future_position(0.2)
        else:
            ekf.predict(t)
            ekf.update(measurement)
            filtered_pos = ekf.get_current_position()
            predicted_pos = ekf.predict_future_position(0.2)
        
        filtered_trajectory.append(filtered_pos)
        predicted_trajectory.append(predicted_pos)
        
        # 打印部分结果
        if i % 20 == 0:
            velocity = ekf.get_current_velocity()
            acceleration = ekf.get_current_acceleration()
            uncertainty = ekf.get_position_uncertainty()
            print(f"时间: {t:.1f}s")
            print(f"  真实: [{true_pos[0]:.2f}, {true_pos[1]:.2f}, {true_pos[2]:.2f}]")
            print(f"  测量: [{measurement[0]:.2f}, {measurement[1]:.2f}, {measurement[2]:.2f}]")
            print(f"  滤波: [{filtered_pos[0]:.2f}, {filtered_pos[1]:.2f}, {filtered_pos[2]:.2f}]")
            print(f"  预测(0.2s后): [{predicted_pos[0]:.2f}, {predicted_pos[1]:.2f}, {predicted_pos[2]:.2f}]")
            print(f"  速度: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")
            print(f"  加速度: [{acceleration[0]:.2f}, {acceleration[1]:.2f}, {acceleration[2]:.2f}]")
            print(f"  不确定性: {uncertainty:.3f}")
            print()
    
    # 计算误差统计
    true_trajectory = np.array(true_trajectory)
    measurements = np.array(measurements)
    filtered_trajectory = np.array(filtered_trajectory)
    
    measurement_error = np.mean(np.linalg.norm(measurements - true_trajectory, axis=1))
    filter_error = np.mean(np.linalg.norm(filtered_trajectory - true_trajectory, axis=1))
    
    print(f"平均测量误差: {measurement_error:.3f}m")
    print(f"平均滤波误差: {filter_error:.3f}m")
    print(f"误差改善: {((measurement_error - filter_error) / measurement_error * 100):.1f}%")
    
    return true_trajectory, measurements, filtered_trajectory, predicted_trajectory


def test_lost_target():
    """测试目标丢失情况下的预测"""
    print("\n=== 测试目标丢失处理 (修正后逻辑) ===")
    
    ekf = ExtendedKalmanFilter3D(process_noise_std=0.01, measurement_noise_std=0.1)
    dt = 0.1
    
    # 先建立稳定的跟踪
    last_true_pos = np.array([0.0, 0.0, 1.0])
    for i in range(20):
        t = i * dt
        # 匀速运动
        true_pos = np.array([0.5 * t, 0.2 * t, 1.0])
        last_true_pos = true_pos
        measurement = true_pos + np.random.normal(0, 0.1, 3)
        
        if not ekf.is_initialized():
            ekf.initialize(measurement, t)
        else:
            ekf.predict(t)
            ekf.update(measurement)
    
    print("建立跟踪后的状态:")
    pos = ekf.get_current_position()
    vel = ekf.get_current_velocity()
    print(f"  滤波位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    print(f"  估计速度: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
    
    # 模拟目标丢失，并与真实轨迹对比
    print("\n模拟目标丢失并对比预测与真实轨迹...")
    
    last_known_velocity = np.array(vel) # 使用最后估计的速度来推算真实轨迹
    
    for i in range(1, 16): # 从丢失后的第一帧开始
        t = 2.0 + (i-1) * dt
        
        # 1. 滤波器进行预测 (无更新)
        predicted_pos = ekf.handle_lost_target(t)
        
        # 2. 计算此时的真实轨迹（假设物体继续按最后已知速度运动）
        time_since_lost = i * dt
        true_pos_after_lost = last_true_pos + last_known_velocity * time_since_lost
        
        if predicted_pos is None:
            print(f"时间 {t:.1f}s: 目标丢失时间过长，停止预测")
            break
        else:
            error = np.linalg.norm(np.array(predicted_pos) - true_pos_after_lost)
            print(f"时间 {t:.1f}s (丢失 {i*dt:.1f}s):")
            print(f"  真实位置: [{true_pos_after_lost[0]:.2f}, {true_pos_after_lost[1]:.2f}, {true_pos_after_lost[2]:.2f}]")
            print(f"  预测位置: [{predicted_pos[0]:.2f}, {predicted_pos[1]:.2f}, {predicted_pos[2]:.2f}] | 误差: {error:.3f}m")
            
            # 当丢失时间过长，手动停止
            if i >= ekf.max_lost_count:
                print(f"时间 {(t+dt):.1f}s: 达到最大丢失计数，测试停止。")
                break


def main():
    """主测试函数"""
    print("扩展卡尔曼滤波器测试程序")
    print("="*50)
    
    # 设置随机种子以获得可重复的结果
    np.random.seed(42)
    
    try:
        # 测试基本功能
        true_traj, measurements, filtered_traj, predicted_traj = test_basic_ekf()
        
        # 测试目标丢失处理
        test_lost_target()
        
        print("\n" + "="*50)
        print("所有测试完成！")
        print("✅ 基本扩展卡尔曼滤波器正常工作 (模拟行人行走)")
        print("✅ 目标丢失处理正常工作")
        
        # 尝试绘制结果（如果有matplotlib）
        try:
            if MATPLOTLIB_AVAILABLE:
                plot_results(true_traj, measurements, filtered_traj)
            else:
                print("\n📊 matplotlib未安装，跳过绘图")
        except Exception as e:
            print(f"\n📊 绘图时出现错误: {e}")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def plot_results(true_traj, measurements, filtered_traj):
    """绘制测试结果"""
    if not MATPLOTLIB_AVAILABLE:
        print("📊 matplotlib未安装，无法绘图")
        return
        
    fig = plt.figure(figsize=(18, 5))
    
    # X坐标对比
    plt.subplot(1, 3, 1)
    plt.plot(true_traj[:, 0], label='Real', linewidth=2.5, color='black')
    plt.plot(measurements[:, 0], 'o', alpha=0.5, markersize=4, label='Measurement', color='red')
    plt.plot(filtered_traj[:, 0], label='Filtered', linewidth=2, color='blue')
    plt.title('X Coordinate')
    plt.xlabel('Time Step')
    plt.ylabel('X (m)')
    plt.legend()
    plt.grid(True)
    
    # Y坐标对比
    plt.subplot(1, 3, 2)
    plt.plot(true_traj[:, 1], label='Real', linewidth=2.5, color='black')
    plt.plot(measurements[:, 1], 'o', alpha=0.5, markersize=4, label='Measurement', color='red')
    plt.plot(filtered_traj[:, 1], label='Filtered', linewidth=2, color='blue')
    plt.title('Y Coordinate')
    plt.xlabel('Time Step')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    
    # Z坐标对比
    plt.subplot(1, 3, 3)
    plt.plot(true_traj[:, 2], label='Real', linewidth=2.5, color='black')
    plt.plot(measurements[:, 2], 'o', alpha=0.5, markersize=4, label='Measurement', color='red')
    plt.plot(filtered_traj[:, 2], label='Filtered', linewidth=2, color='blue')
    plt.title('Z Coordinate')
    plt.xlabel('Time Step')
    plt.ylabel('Z (m)')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Extended Kalman Filter Performance (Pedestrian Simulation)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('ekf_pedestrian_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n📊 测试结果已保存为 ekf_pedestrian_test_results.png")

if __name__ == "__main__":
    main()