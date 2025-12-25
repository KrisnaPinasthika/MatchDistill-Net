import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import model_io
import models
from dataloader import DepthDataLoader
from utils import RunningAverageDict

from models.matchdistilnet import MatchDistillNet

def colorize(value, cmap='jet'):
    # normalize
    import matplotlib.pyplot as plt
    vmin = value.min() # if vmin is None else vmin
    vmax = value.max() # if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        value = value * 0.

    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :]

    return img


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def predict_tta(model, image, args):
    bins, pred, attn_weights = model(image, is_train=False)
    
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)
    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)

    # print(f"image shape : {image.shape}")
    b, c, h, w = image.shape
    if isinstance(pred, np.ndarray):  # Check if pred is a NumPy array
        pred = torch.from_numpy(pred)  # Convert it to a PyTorch tensor

    final = nn.functional.interpolate(pred, (h, w), mode='bilinear', align_corners=True)
    return torch.Tensor(final) # torch.Tensor([pred]) # 


def eval(model, test_loader, args, gpus=None, ):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]

    if args.save_dir is not None:
        if args.save_dir in os.listdir():
            print(f"File [{args.save_dir}] exist")
        else:
            os.makedirs(args.save_dir)

    metrics = RunningAverageDict()
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):

            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            final = predict_tta(model, image, args)
            final = final.squeeze().cpu().numpy()
            
            final[np.isinf(final)] = args.max_depth
            final[np.isnan(final)] = args.min_depth

            if args.save_dir is not None:
                if args.dataset == 'nyu':
                    impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
                    factor = 1000
                else:
                    dpath = batch['image_path'][0].split('/')
                    impath = dpath[1] + "_" + dpath[-1]
                    impath = impath.split('.')[0]
                    factor = 256

                pred_path = os.path.join(args.save_dir, f"{impath}.png")
                pred = colorize((final * factor).astype('uint16'))
                Image.fromarray(pred).save(pred_path)
                

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    # print("Invalid ground truth")
                    total_invalid += 1
                    continue

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            
            gt[gt <= args.min_depth] = args.min_depth
            gt[gt >= args.max_depth] = args.max_depth
            
            depth_path = os.path.join(args.save_dir, f"{impath}_gt.png")
            depth_real = colorize((gt * factor).astype('uint16'))
            Image.fromarray(depth_real).save(depth_path)
                
            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Model evaluator', fromfile_prefix_chars='@',
                                        conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args


    parser.add_argument('--backbone', default='eff_b5', type=str, help='The backbone you desire, only eff_b5')

    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument('--save-dir', '--save_dir', default=None, type=str, help='Store predictions in folder')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")

    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../dataset/nyu/official_splits/test/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/official_splits/test/', type=str,
                        help="path to dataset gt")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
    parser.add_argument('--checkpoint_path', '--checkpoint-path', type=str, required=True,
                        help="checkpoint file to use for prediction")

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--do_kb_crop', help='Use kitti benchmark cropping', action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # args = parser.parse_args()
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'online_eval').data
    
    model = MatchDistillNet(
        'cuda', 
        "eff_b5", 
        n_bins=128, 
        window_sizes=7, 
        layers=2, 
        qkv_bias=True, 
        drop_prob=0.15, 
        min_val=args.min_depth, 
        max_val=args.max_depth
    ).to('cuda')

    from collections import OrderedDict
    state_dict = torch.load(args.checkpoint_path, map_location='cuda')['model']

    # Remove the 'module.' prefix from state dict keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # Remove 'module.'
        new_state_dict[name] = v

    # Load the new state dict into your model
    model.load_state_dict(new_state_dict, strict=False)
    print("--> Success Loaded checkpoint")

    # model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()
    
    torch.backends.cudnn.enabled = False
    eval(model, test, args, gpus=[device])
