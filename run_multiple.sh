#!/bin/bash

	#for seq in 00 01 02 03 04 05 06 07 08 09 10; do
	for seq in 04; do
		echo "Kitti $seq"
		
		python3 main.py --root_path="datasets/kitti" \
						--config="params/kitti_orb_brutematch_$seq.yaml" \
						--camera_folder="image_0" --timestamps_file="datasets/kitti/sequences/$seq/times.txt" \
						--save_format="tum"
	done
