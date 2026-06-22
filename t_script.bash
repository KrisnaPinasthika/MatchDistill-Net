#!/usr/bin/env bash

# Run the Python script with the specified arguments
# NYU
python ./train.py --same-lr --distributed --backbone eff_b5 --name ASPP_RetrainNoTeacher --sparta_type strip_normal --attention_type ifa --alpha_role ifa --epochs 25 --bs 18 --validate-every 250 --lr 0.000359 # --div_factor 15
# python ./evaluate.py args_test_nyu.txt

# KITTI
# python ./evaluate.py args_test_kitti_eigen.txt

