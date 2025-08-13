# # #!/bin/bash
# gnome-terminal -- bash -i -c "conda deactivate && python3 track_torch_ros2/simple_pose_publisher.py; exec bash"
# gnome-terminal -- bash -i -c "conda deactivate && python3 track_torch_ros2/vision_control_subscriber.py; exec bash"


source ~/anaconda3/etc/profile.d/conda.sh

gnome-terminal -- bash -i -c "conda activate yolo && python grpc_server.py; exec bash"
gnome-terminal -- bash -i -c "conda activate yolo && python grpc_client_test.py; exec bash"
gnome-terminal -- bash -i -c "conda activate yolo && python track.py ; exec bash"

