#!/bin/bash
export DISPLAY=:1
# 初始化 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc

cd /home/ake/track/grpc

# 启动 gRPC 服务端
gnome-terminal --title="gRPC Server" -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate yolo && python grpc_server.py; exec bash"
sleep 5

# 启动 gRPC 客户端测试
gnome-terminal --title="gRPC Client Test" -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate yolo && python grpc_client_test.py; exec bash"
sleep 5

# 启动主跟踪程序
gnome-terminal --title="Track Pose" -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate yolo && python track_grpc_pose.py; exec bash"

# 保持脚本运行
while true; do
    sleep 10
    # 检查进程是否还在运行
    if ! pgrep -f "grpc_server.py" > /dev/null; then
        echo "gRPC server died, restarting..."
        gnome-terminal --title="gRPC Server" -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate yolo && python grpc_server.py; exec bash"
    fi
    
    if ! pgrep -f "grpc_client_test.py" > /dev/null; then
        echo "grpc_client_test died, restarting..."
        gnome-terminal --title="gRPC Client Test" -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate yolo && python grpc_client_test.py; exec bash"
    fi

    if ! pgrep -f "track_grpc_pose.py" > /dev/null; then
        echo "Track pose died, restarting..."
        gnome-terminal --title="Track Pose" -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate yolo && python track_grpc_pose.py; exec bash"
    fi
done




