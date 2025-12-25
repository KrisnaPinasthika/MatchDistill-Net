#!/usr/bin/env bash

# Run the Python script with the specified arguments
# NYU
# python ./train.py --same-lr --distributed --backbone eff_b5 --name ASPP_RetrainNoTeacher --sparta_type strip_normal --attention_type ifa --alpha_role ifa --epochs 25 --bs 18 --validate-every 250 --lr 0.000359 # --div_factor 15
# python ./evaluate.py args_test_nyu.txt

# KITTI
# python ./train.py args_train_kitti_eigen.txt
python ./evaluate.py args_test_kitti_eigen.txt


#
# ASPP_Scaled : add rmse loss 
# ASPP Normed : add rmse loss 
# ASPP Normed Plus : add rmse loss and Attention weight

# ASPP_Inversed_SoftDeltaBracket_1_08-Nov_06-22-nodebs21-tep25-lr0.000359-wd0.1-2fe3b1ff-055a-4175-9c66-5a5fa4628389_best.pt
# ASPP_InversedPlus_SoftDeltaBracket_1_09-Nov_23-44-nodebs21-tep25-lr0.000359-wd0.1-7f36cc2d-4fbf-4716-819a-8ab297e57326_best.pt
# ASPP_InversedPlusConcat_SoftDeltaBracket_1_12-Nov_17-34-nodebs21-tep25-lr0.000359-wd0.1-88e7a9e2-3cb1-4fcf-bf51-346fe3d78aed_latest.pt