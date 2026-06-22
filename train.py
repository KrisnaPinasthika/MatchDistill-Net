import argparse
import os
import sys
import uuid
from datetime import datetime as dt
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd

import model_io
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BerHuLoss, BinsChamferLoss, DepthSmoothness, SoftDeltaLoss, LossSSIM, LogRMSE
from utils import RunningAverage, colorize
import matplotlib
import torch.nn.functional as F

from models.matchdistilnet import MatchDistillNet

PROJECT = "MatchDistillNet"
logging = True

# ─────────────────────────────────────────────────────────────
# CSV Logger — menyimpan semua metric ke file CSV
# ─────────────────────────────────────────────────────────────
class CSVLogger:
    def __init__(self, log_dir, run_name):
        os.makedirs(log_dir, exist_ok=True)
        self.train_path = os.path.join(log_dir, f"{run_name}_train.csv")
        self.val_path   = os.path.join(log_dir, f"{run_name}_val.csv")
        self.train_rows = []
        self.val_rows   = []
        print(f"[CSVLogger] Train log : {self.train_path}")
        print(f"[CSVLogger] Val   log : {self.val_path}")

    def log_train(self, step, epoch, soft_delta, chamfer=None):
        row = {"step": step, "epoch": epoch, "Train/SoftDeltaLoss": soft_delta}
        if chamfer is not None:
            row["Train/ChamferLoss"] = chamfer
        self.train_rows.append(row)
        # flush setiap 50 step agar tidak hilang kalau crash
        if len(self.train_rows) % 50 == 0:
            self._flush_train()

    def log_val(self, step, epoch, triple_loss, metrics: dict):
        row = {"step": step, "epoch": epoch, "Test/TripleLoss": triple_loss}
        row.update({f"Test/{k}": v for k, v in metrics.items()})
        self.val_rows.append(row)
        self._flush_val()

    def _flush_train(self):
        if not self.train_rows:
            return
        df = pd.DataFrame(self.train_rows)
        df.to_csv(self.train_path, index=False)

    def _flush_val(self):
        if not self.val_rows:
            return
        df = pd.DataFrame(self.val_rows)
        df.to_csv(self.val_path, index=False)

    def close(self):
        self._flush_train()
        self._flush_val()
        print(f"[CSVLogger] Saved → {self.train_path}")
        print(f"[CSVLogger] Saved → {self.val_path}")
# ─────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def is_rank_zero(args):
    return args.rank == 0

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.
    cmapper = matplotlib.colormaps[cmap]
    value = cmapper(value, bytes=True)
    img = value[:, :, :3]
    return img


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred  = colorize(pred,  vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input":      [wandb.Image(img)],
            "GT":         [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)

def count_parameters(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(f"Total Model Params\t: {total_params:,}")

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    window_sizes = 7
    model = MatchDistillNet(
        'cuda',
        args.backbone,
        n_bins=128,
        window_sizes=window_sizes,
        layers=2,
        qkv_bias=True,
        drop_prob=0.15,
        min_val=args.min_depth,
        max_val=args.max_depth, 
        normal_decoder=False, 
        normal_head=False,
    ).to('cuda')

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers    = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu,
            find_unused_parameters=True)
    elif args.gpu is None:
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch      = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu,
          root=args.root, experiment_name=args.name, optimizer_state_dict=None)


def train(model, args, epochs=25, teacher_epochs=20, experiment_name="test-01x", lr=0.000359, root=".", device=None, 
            optimizer_state_dict=[None, None]):
    global PROJECT

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"Training {experiment_name} | {args.backbone} | Bins : {args.n_bins}")

    run_id = (
                f"{dt.now().strftime('%d-%h_%H-%M')}-bs{args.bs}-ep{epochs}"
                f"-lr{lr}-wd{args.wd}-{str(uuid.uuid4())[:5]}"
            )
    name = f"{experiment_name}_{args.backbone}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging

    # ── Wandb init ──────────────────────────────────────────
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        os.environ["WANDB_MODE"] = "online"
        wandb.init(
            project=PROJECT, name=name, config=args,
            dir="./wandb", tags=tags, notes=args.notes,
        )
        print("Wandb init called on Rank 0")

    # ── CSV Logger init ──────────────────────────────────────
    # Folder: saved_log/<run_name>/  (sama persis dengan nama checkpoint)
    csv_log_dir = os.path.join(root, "saved_log", name)
    csv_logger  = CSVLogger(csv_log_dir, name) if should_write else None

    train_loader = DepthDataLoader(args, 'train').data
    test_loader  = DepthDataLoader(args, 'online_eval').data

    criterion_softdelta = SoftDeltaLoss(threshold=1.25)
    criterion_logrmse = LogRMSE()
    criterion_bins = BinsChamferLoss() if args.chamfer else None

    model.train()

    m = model.module if args.multigpu else model
    teacher_modules   = m.encoder.getTeacherRelatedLayer()
    teacher_params    = []
    for module in teacher_modules:
        teacher_params += list(module.parameters())
    teacher_param_ids = set(id(p) for p in teacher_params)
    student_params    = [p for p in m.parameters() if id(p) not in teacher_param_ids]

    optimizer_teacher = optim.AdamW(teacher_params, weight_decay=args.wd, lr=args.lr)
    optimizer_student = optim.AdamW(student_params, weight_decay=args.wd, lr=args.lr)

    if optimizer_state_dict is not None:
        if optimizer_state_dict[0] is not None:
            optimizer_student.load_state_dict(optimizer_state_dict[0])
        if optimizer_state_dict[1] is not None and optimizer_teacher is not None:
            optimizer_teacher.load_state_dict(optimizer_state_dict[1])

    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    scheduler_teacher = optim.lr_scheduler.OneCycleLR(
        optimizer_teacher, lr, epochs=teacher_epochs, steps_per_epoch=iters,
        cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
        last_epoch=args.last_epoch, div_factor=args.div_factor,
        final_div_factor=args.final_div_factor)

    scheduler_student = optim.lr_scheduler.OneCycleLR(
        optimizer_student, lr, epochs=epochs, steps_per_epoch=iters,
        cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
        last_epoch=args.last_epoch, div_factor=args.div_factor,
        final_div_factor=args.final_div_factor)

    if args.resume != '' and (scheduler_teacher is not None and scheduler_student is not None):
        scheduler_teacher.step(args.epoch + 1)
        scheduler_student.step(args.epoch + 1)


    for epoch in range(args.epoch, epochs):
        print(f"Epoch : {epoch}")

        if epoch == teacher_epochs:
            print("[INFO] Membebaskan VRAM dari Teacher...")
            for param in teacher_params:
                param.requires_grad = False
            optimizer_teacher = None
            scheduler_teacher = None
            torch.cuda.empty_cache()

        for i, batch in (tqdm(enumerate(train_loader),
                               desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                               total=len(train_loader))
                          if is_rank_zero(args) else enumerate(train_loader)):

            optimizer_student.zero_grad()
            if epoch < teacher_epochs:
                optimizer_teacher.zero_grad()

            img   = batch['image'].to(device)
            depth = batch['depth'].to(device)

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            is_train = epoch < teacher_epochs

            # with torch.cuda.amp.autocast():
            bin_edges, main_pred, attn_weights = model(img, is_train)
            pred = torch.clamp(main_pred, min=args.min_depth, max=args.max_depth)
            mask = (depth > args.min_depth).to(torch.bool)
            should_interpolate = True

            if bin_edges is not None:
                l_chamfer = criterion_bins(bin_edges, depth)
            else: 
                l_chamfer = torch.Tensor([0]).to(img.device)
                
            if attn_weights is not None:
                all_individual_attns = [t for sublist in attn_weights for t in sublist]
                if all_individual_attns:
                    entropies = [(-torch.sum(p.clamp(min=1e-4) * torch.log(p.clamp(min=1e-4)), dim=-1)).mean() for p in all_individual_attns]
                    l_reg = torch.stack(entropies).sum() / (len(attn_weights) * len(all_individual_attns))
                else:
                    l_reg = torch.tensor(0.0, device=device)
            else:
                l_reg = torch.tensor(0.0, device=img.device)

            l_soft_delta  = criterion_softdelta(pred, depth, mask=mask, interpolate=should_interpolate)
            loss_logrmse  = criterion_logrmse(pred,  depth, mask=mask, interpolate=should_interpolate)

            loss = l_soft_delta + (0.1 * l_chamfer) + (0.1 * loss_logrmse) + (0.01 * l_reg)

            # scaler.scale(loss).backward()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student_params, 0.1)
            optimizer_student.step()


            if is_train and optimizer_teacher is not None:
                has_teacher_grad = any(p.grad is not None for p in teacher_params)
                if has_teacher_grad:
                    torch.nn.utils.clip_grad_norm_(teacher_params, 1.0)
                    optimizer_teacher.step()

            scheduler_student.step()
            if is_train and scheduler_teacher is not None:
                scheduler_teacher.step()

            # ── Logging ─────────────────────────────────────
            chamfer_val = round(l_chamfer.item(), 5) if bin_edges is not None else None

            # Wandb
            if should_log:
                log_dict = {
                    f"Train/{criterion_softdelta.name}": round(l_soft_delta.item(), 5),
                    "Train/Epoch": epoch + 1,
                }
                if chamfer_val is not None:
                    log_dict[f"Train/{criterion_bins.name}"] = chamfer_val
                wandb.log(log_dict, step=step)

            # CSV
            if csv_logger:
                csv_logger.log_train(
                    step=step,
                    epoch=epoch + 1,
                    soft_delta=round(l_soft_delta.item(), 5),
                    chamfer=chamfer_val,
                )
            # ────────────────────────────────────────────────

            step += 1

            if should_write and step % args.validate_every == 0:
                model.eval()
                metrics, val_si = validate(
                    args, model, test_loader,
                    [criterion_softdelta, criterion_bins],
                    epoch, epochs, device)

                triple_loss = round(val_si.get_value(), 5)

                # Wandb
                if should_log:
                    wandb.log({
                        "Test/TripleLoss": triple_loss,
                        **{f"Test Metrics/{k}": v for k, v in metrics.items()}
                    }, step=step)

                # CSV
                if csv_logger:
                    csv_logger.log_val(
                        step=step,
                        epoch=epoch + 1,
                        triple_loss=triple_loss,
                        metrics=metrics,
                    )

                # ── Checkpoint — nama sama dengan folder log ──
                ckpt_root = os.path.join(root, "checkpoints")
                if epoch == (epochs - 1):
                    model_io.save_checkpoint(
                        model, optimizer_student, optimizer_teacher,
                        scheduler_student, scheduler_teacher, epoch,
                        f"{name}_latest.pt",
                        root=ckpt_root)

                if metrics['abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(
                        model, optimizer_student, optimizer_teacher,
                        scheduler_student, scheduler_teacher, epoch,
                        f"{name}_best.pt",
                        root=ckpt_root)
                    best_loss = metrics['abs_rel']

                model.train()

    # ── Tutup logger ────────────────────────────────────────
    if csv_logger:
        csv_logger.close()

    if should_log:
        wandb.finish()

    return model


def validate(args, model, test_loader, criterions, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si  = RunningAverage()
        metrics = utils.RunningAverageDict()

        for batch in (tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation")
                      if is_rank_zero(args) else test_loader):
            img   = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)

            is_train = False
            bin_edges, pred, attn_weights = model(img, is_train)
            pred = torch.clamp(pred, min=args.min_depth, max=args.max_depth)
            mask = (depth > args.min_depth).to(torch.bool)

            l_softdelta = criterions[0](pred, depth, mask=mask, interpolate=True)
            if bin_edges is not None:
                l_chamfer = criterions[1](bin_edges, depth)
                loss = l_softdelta + 0.1 * l_chamfer
            else:
                loss = l_softdelta

            val_si.append(loss.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)]             = args.max_depth_eval
            pred[np.isnan(pred)]             = args.min_depth_eval

            gt_depth   = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)
                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                              int(0.03594771 * gt_width) :int(0.96405229 * gt_width)] = 1
                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                                  int(0.0359477 * gt_width) :int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)

            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training script. Default values of all arguments are recommended for reproducibility',
        fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--backbone',         default='eff_b5',  type=str)
    parser.add_argument('--seed',             type=int,          default=42)
    parser.add_argument('--epochs',           default=25,        type=int)
    parser.add_argument('--n-bins','--n_bins',default=128,       type=int)
    parser.add_argument('--lr','--learning-rate', default=0.000357, type=float)
    parser.add_argument('--wd','--weight-decay',  default=0.1,      type=float)
    parser.add_argument('--w_chamfer','--w-chamfer', default=0.1,   type=float)
    parser.add_argument('--div-factor','--div_factor',       default=25,  type=float)
    parser.add_argument('--final-div-factor','--final_div_factor', default=100, type=float)

    parser.add_argument('--bs',               default=21,  type=int)
    parser.add_argument('--validate-every','--validate_every', default=1000, type=int)
    parser.add_argument('--gpu',              default=None, type=int)
    parser.add_argument('--name',             default="UnetAdaptiveBins")
    parser.add_argument('--norm',             default="linear", type=str,
                        choices=['linear','softmax','sigmoid'])
    parser.add_argument('--same-lr','--same_lr', default=False, action="store_true")
    parser.add_argument('--distributed',      default=False, action="store_true")
    parser.add_argument('--root',             default=".", type=str)
    parser.add_argument('--resume',           default='',  type=str)
    parser.add_argument('--notes',            default='',  type=str)
    parser.add_argument('--tags',             default='sweep', type=str)
    parser.add_argument('--workers',          default=10,  type=int)
    parser.add_argument('--dataset',          default='nyu', type=str)
    parser.add_argument('--data_path',        default='../dataset/nyu/sync/', type=str)
    parser.add_argument('--gt_path',          default='../dataset/nyu/sync/', type=str)
    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt", type=str)
    parser.add_argument('--input_height',     type=int, default=416)
    parser.add_argument('--input_width',      type=int, default=544)
    parser.add_argument('--max_depth',        type=float, default=10)
    parser.add_argument('--min_depth',        type=float, default=1e-3)
    parser.add_argument('--do_random_rotate', default=True, action='store_true')
    parser.add_argument('--degree',           type=float, default=2.5)
    parser.add_argument('--do_kb_crop',       action='store_true')
    parser.add_argument('--use_right',        action='store_true')
    parser.add_argument('--data_path_eval',   default="../dataset/nyu/official_splits/test/", type=str)
    parser.add_argument('--gt_path_eval',     default="../dataset/nyu/official_splits/test/", type=str)
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt", type=str)
    parser.add_argument('--min_depth_eval',   type=float, default=1e-3)
    parser.add_argument('--max_depth_eval',   type=float, default=10)
    parser.add_argument('--eigen_crop',       default=True, action='store_true')
    parser.add_argument('--garg_crop',        action='store_true')

    if sys.argv.__len__() == 2:
        args = parser.parse_args(['@' + sys.argv[1]])
    else:
        args = parser.parse_args()

    set_seed(args.seed)
    args.batch_size  = args.bs
    args.num_threads = args.workers
    args.mode        = 'train'
    args.chamfer     = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str       = os.environ['SLURM_JOB_NODELIST'].replace('[','').replace(']','')
        nodes          = node_str.split(',')
        args.world_size = len(nodes)
        args.rank       = int(os.environ['SLURM_PROCID'])
    except KeyError:
        args.world_size = 1
        args.rank       = 0
        nodes           = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('spawn', force=True)
        port         = np.random.randint(15000, 15025)
        args.dist_url     = 'tcp://{}:{}'.format(nodes[0], port)
        args.dist_backend = 'nccl'
        args.gpu          = None
        print(f"Rank : {args.rank}  URL : {args.dist_url}")

    ngpus_per_node  = torch.cuda.device_count()
    args.num_workers     = args.workers
    args.ngpus_per_node  = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)