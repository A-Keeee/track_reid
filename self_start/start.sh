#!/bin/bash

# ==============================================================================
# --- 阶段 1: 检查并重置 USB 相机 ---
# ==============================================================================

echo "--- 阶段 1: 正在检查 OAK 相机状态 ---"

# **警告：这里硬编码了 sudo 密码。这极不安全，仅用于个人测试。**
SUDO_PASSWORD="juwei"

# 检查相机状态
if lsusb | grep -q "03e7:2485"; then
    echo "相机已处于正常工作模式 (ID 03e7:2485)，无需重置。"
    echo "--- 阶段 1 完成 ---"
    echo ""
elif lsusb | grep -q "03e7:f63b"; then
    echo "检测到相机处于 bootloader 模式 (ID 03e7:f63b)，需要重置。"
    echo "正在尝试重置相机..."
    
    # ==================== 查找并重置 f63b 设备 ====================
    USB_DEVICE_SYS_NAME=""
    for dir in /sys/bus/usb/devices/*/; do
        if [ -f "$dir/idVendor" ] && [ -f "$dir/idProduct" ]; then
            VENDOR_ID=$(cat "$dir/idVendor" | tr -d '\n\r')
            PRODUCT_ID=$(cat "$dir/idProduct" | tr -d '\n\r')
            
            if [ "$VENDOR_ID" = "03e7" ] && [ "$PRODUCT_ID" = "f63b" ]; then
                USB_DEVICE_SYS_NAME=$(basename "$dir")
                break
            fi
        fi
    done
    
    # 如果上面的方法失败，尝试备用方法
    if [ -z "$USB_DEVICE_SYS_NAME" ]; then
        echo "第一种方法未找到设备，尝试备用方法..."
        
        # 从 lsusb 输出解析设备信息
        BUS_NUM=$(lsusb | grep "03e7:f63b" | sed 's/Bus \([0-9]*\) Device \([0-9]*\).*/\1/')
        DEV_NUM=$(lsusb | grep "03e7:f63b" | sed 's/Bus \([0-9]*\) Device \([0-9]*\).*/\2/')
        
        if [ -n "$BUS_NUM" ] && [ -n "$DEV_NUM" ]; then
            # 格式化总线和设备号
            BUS_NUM=$(printf "%d" $BUS_NUM)
            DEV_NUM=$(printf "%d" $DEV_NUM)
            USB_DEVICE_SYS_NAME="${BUS_NUM}-${DEV_NUM}"
            
            # 检查这个设备名是否存在
            if [ ! -d "/sys/bus/usb/devices/$USB_DEVICE_SYS_NAME" ]; then
                USB_DEVICE_SYS_NAME=""
            fi
        fi
    fi
    
    if [ -z "$USB_DEVICE_SYS_NAME" ]; then
        echo "错误：无法找到 bootloader 模式设备的系统路径。"
        echo "调试信息："
        echo "可用的USB设备："
        ls -la /sys/bus/usb/devices/ | grep -E "^d.*[0-9]+-[0-9]+$"
        exit 1
    fi
    
    echo "找到 bootloader 设备系统名称: $USB_DEVICE_SYS_NAME"
    echo "设备路径: /sys/bus/usb/devices/$USB_DEVICE_SYS_NAME"
    
    # 验证设备路径存在
    if [ ! -d "/sys/bus/usb/devices/$USB_DEVICE_SYS_NAME" ]; then
        echo "错误：设备路径不存在: /sys/bus/usb/devices/$USB_DEVICE_SYS_NAME"
        exit 1
    fi
    
    echo "正在尝试卸载 (禁用) 设备..."
    
    # 卸载 (禁用) 设备
    echo "$SUDO_PASSWORD" | sudo -S sh -c "echo '$USB_DEVICE_SYS_NAME' > /sys/bus/usb/drivers/usb/unbind" 2>/dev/null
    
    # 检查卸载是否成功
    if [ $? -ne 0 ]; then
        echo "警告：设备可能已经处于卸载状态，或者卸载失败。继续执行..."
    else
        echo "设备已卸载。"
    fi
    
    echo "等待 5 秒后重新加载..."
    sleep 5
    
    echo "正在尝试加载 (启用) 设备..."
    # 加载 (启用) 设备
    echo "$SUDO_PASSWORD" | sudo -S sh -c "echo '$USB_DEVICE_SYS_NAME' > /sys/bus/usb/drivers/usb/bind" 2>/dev/null
    
    # 检查加载是否成功
    if [ $? -ne 0 ]; then
        echo "警告：重新加载可能失败，但这可能是正常的。"
    else
        echo "设备已成功重新加载。"
    fi
    
    echo "等待 8 秒以确保设备完全重新初始化..."
    sleep 8
    
    # 检查重置后的状态
    echo "检查重置后的设备状态..."
    if lsusb | grep -q "03e7:2485"; then
        echo "✓ 相机已成功重置为正常工作模式 (ID 03e7:2485)"
    elif lsusb | grep -q "03e7:f63b"; then
        echo "⚠ 相机仍处于 bootloader 模式，可能需要手动干预"
        echo "建议：断开并重新连接相机，或检查相机固件"
    else
        echo "⚠ 未检测到相机设备，请检查连接"
    fi
    
    echo "--- 阶段 1 完成 ---"
    echo ""
else
    echo "错误：未找到 OAK 相机设备。"
    echo "请确认相机已连接。当前可见的 USB 设备："
    lsusb | grep -E "(03e7|Intel|Movidius)"
    exit 1
fi

# ...existing code...
# ==============================================================================
# --- 阶段 2 & 3: 启动和守护应用程序 (保持不变) ---
# ==============================================================================

#!/bin/bash
export DISPLAY=:0


# 初始化 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc

# 切换到工作目录
cd /home/juwei/track_torch

# 启动 gRPC 服务端
echo "正在启动 gRPC Server..."
gnome-terminal --title="gRPC Server" -- bash -ic "source ~/anaconda3/etc/profile.d/conda.sh && conda activate yolo && python grpc_server.py"
echo "等待 5 秒以确保服务端完全启动..."
sleep 5

# 启动主跟踪程序
echo "正在启动主跟踪程序 Track Pose..."
gnome-terminal --title="Track Pose" -- bash -ic "source ~/anaconda3/etc/profile.d/conda.sh && conda activate yolo && python track_torch.py --no-viz"
echo "主跟踪程序已启动。"

# --- 修改部分开始 ---

# 监控进程，如果任何一个进程中断，则终止所有相关进程并退出脚本
echo "开始监控进程... 如果任一进程中断，将终止所有相关进程。"

while true; do
    # 检查 gRPC server 是否仍在运行
    pgrep -f "grpc_server.py" > /dev/null
    GRPC_SERVER_RUNNING=$?

    # 检查主跟踪程序是否仍在运行
    pgrep -f "track_torch.py" > /dev/null
    TRACK_TORCH_RUNNING=$?

    # 如果两个进程中任何一个没有在运行 ($? 不为 0)，则执行清理并退出
    if [ $GRPC_SERVER_RUNNING -ne 0 ] || [ $TRACK_TORCH_RUNNING -ne 0 ]; then
        echo "检测到有进程已中断。正在关闭所有相关进程..."

        # 输出哪个进程中断了 (可选，用于调试)
        if [ $GRPC_SERVER_RUNNING -ne 0 ]; then
            echo "gRPC Server (grpc_server.py) 已停止运行。"
        fi
        if [ $TRACK_TORCH_RUNNING -ne 0 ]; then
            echo "主跟踪程序 (track_torch.py) 已停止运行。"
        fi

        # 使用 pkill 强制结束所有仍在运行的相关进程
        pkill -f "grpc_server.py"
        pkill -f "track_torch.py"

        echo "所有相关进程已被终止。监控脚本即将退出。"
        break # 退出 while 循环
    fi

    # 等待 10 秒后再次检查
    sleep 10
done

# --- 修改部分结束 ---

echo "监控脚本执行完毕。"
exit 0