#!/bin/bash
docker start a
docker exec a python tools/tmp.py
tmux new-session -d -s rank0 docker exec a python tools/cls_viz.py --testModel pretrained_models/MDEQ_Small_Cls.pkl --cfg experiments/imagenet/cls_mdeq_SMALL.yaml --rect -0.2 -0.2 0.2 0.2 --resolution 40 40 --rank 0
docker exec a python tools/cls_viz.py --testModel pretrained_models/MDEQ_Small_Cls.pkl --cfg experiments/imagenet/cls_mdeq_SMALL.yaml --modelDir output/o --rect -0.2 -0.2 0.2 0.2 --resolution 40 40 --rank 1
