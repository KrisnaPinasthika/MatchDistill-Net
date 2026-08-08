# MatchDistill-Net
Official implementation for MatchDistill-Net: Knowledge Distillation of Global Context from Transformers to Convolutional Networks for Monocular Depth Estimation. <br>

# Workspace and Dataset Preparation
```
# Make a workspace
mkdir workspace
cd workspace
# Make a folder for datasets
mkdir dataset
# Clone this repository
$ git clone https://github.com/KrisnaPinasthika/MatchDistill-Net.git
```

To use the datasets, we encourage you to read their papers and visit the official websites.
## > <i>[NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)</i>
```
cd ~/workspace/MatchDistill-Net/utils
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
```
## > <i>[KITTI](https://www.cvlibs.net/datasets/kitti/)</i>
```
cd ~/workspace/dataset
mkdir kitti_dataset && cd kitti_dataset
mv ~/Downloads/data_depth_annotated.zip .
unzip data_depth_annotated.zip
```

# Training & Evaluating
To run the training and evaluation code, please use the args files.
```
# Training
python train.py args_train_nyu.txt
python train.py args_train_kitti_eigen.txt 

# Evaluating
python evaluate.py args_test_nyu.txt
python evaluate.py args_test_kitti_eigen.txt
```

# Credits
This code is mainly based on two previous research works.
We would like to express our sincere gratitude to [BTS](https://github.com/cleinc/bts/tree/master) and [AdaBins](https://github.com/shariqfarooq123/AdaBins).
