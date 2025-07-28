#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

gnome-terminal -- bash -i -c "conda activate yolo && python grpc_server.py; exec bash"
gnome-terminal -- bash -i -c "conda activate yolo && python grpc_client_test.py; exec bash"
gnome-terminal -- bash -i -c "conda activate yolo && python track_torch.py; exec bash"

