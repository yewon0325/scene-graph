# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--sparse_query_k', default=200, type=int,
                        help="Number of triplet queries to execute in the decoder; must be <= num_triplets")
    parser.add_argument('--adaptive_query_budget', action='store_true',
                        help='Enable adaptive object-aware query budgeting before triplet decoding')
    parser.add_argument('--coverage_preserving_object_query', action='store_true',
                        help='Enable coverage-preserving object-aware query budgeting without object pruning')
    parser.add_argument('--collect_query_importance', action='store_true')
    parser.add_argument('--query_importance_output', default='query_importance_stats.pt', type=str)
    parser.add_argument('--query_importance_selection', action='store_true')
    parser.add_argument('--query_importance_path', default='', type=str)
    parser.add_argument('--query_selection_mode', default='importance', type=str,
                        choices=['importance', 'hybrid_prefix', 'rare_balanced'])
    parser.add_argument('--prefix_keep_ratio', default=0.5, type=float)
    parser.add_argument('--alpha_query_overall', default=1.0, type=float)
    parser.add_argument('--alpha_query_rare', default=0.5, type=float)
    parser.add_argument('--alpha_query_score', default=0.1, type=float)
    parser.add_argument('--budget_min', default=50, type=int,
                        help='Minimum image-level query budget for adaptive execution')
    parser.add_argument('--budget_max', default=200, type=int,
                        help='Maximum image-level query budget for adaptive execution')
    parser.add_argument('--budget_count_ref', default=20.0, type=float,
                        help='Reference object count used to normalize image complexity for adaptive budgeting')
    parser.add_argument('--budget_uncertainty_ref', default=0.35, type=float,
                        help='Reference mean uncertainty used to normalize image complexity for adaptive budgeting')
    parser.add_argument('--budget_score_bias', default=0.45, type=float,
                        help='Bias term for mapping image complexity to a continuous budget ratio')
    parser.add_argument('--budget_score_scale', default=0.18, type=float,
                        help='Scale term for mapping image complexity to a continuous budget ratio')
    parser.add_argument('--enable_budget_floor', action='store_true',
                        help='Clamp adaptive budget to a minimum floor after budget prediction')
    parser.add_argument('--adaptive_budget_floor', default=130, type=int,
                        help='Minimum adaptive budget applied when budget floor is enabled')
    parser.add_argument('--target_avg_budget', default=155.0, type=float,
                        help='Target average budget used for experiment tracking and tuning')
    parser.add_argument('--tail_boost_adaptive_budget', action='store_true',
                        help='Apply tail-boosted budget correction for low/high complexity images')
    parser.add_argument('--percentile_tail_boost_adaptive_budget', action='store_true',
                        help='Use percentile-based tail-boosted budget correction based on evaluation-set complexity distribution')
    parser.add_argument('--high_complexity_threshold', default=0.70, type=float,
                        help='Complexity threshold for raising the budget floor on hard images')
    parser.add_argument('--very_high_complexity_threshold', default=0.85, type=float,
                        help='Complexity threshold for raising the budget floor further on very hard images')
    parser.add_argument('--low_complexity_threshold', default=0.35, type=float,
                        help='Complexity threshold for capping the budget on easy images')
    parser.add_argument('--low_complexity_percentile', default=30.0, type=float,
                        help='Percentile cutoff for low-complexity images in percentile tail boost')
    parser.add_argument('--high_complexity_percentile', default=85.0, type=float,
                        help='Percentile cutoff for high-complexity images in percentile tail boost')
    parser.add_argument('--very_high_complexity_percentile', default=95.0, type=float,
                        help='Percentile cutoff for very-high-complexity images in percentile tail boost')
    parser.add_argument('--high_complexity_budget_floor', default=180, type=int,
                        help='Minimum budget for high-complexity images when tail boost is enabled')
    parser.add_argument('--very_high_complexity_budget_floor', default=195, type=int,
                        help='Minimum budget for very high-complexity images when tail boost is enabled')
    parser.add_argument('--low_complexity_budget_cap', default=145, type=int,
                        help='Maximum budget for low-complexity images when tail boost is enabled')
    parser.add_argument('--normal_complexity_budget_cap', default=170, type=int,
                        help='Maximum budget for normal-complexity images in percentile tail boost')
    parser.add_argument('--lambda_uncertainty', default=1.0, type=float,
                        help='Weight for object uncertainty in adaptive object scoring')
    parser.add_argument('--lambda_degree', default=1.0, type=float,
                        help='Weight for image-aware relation degree in adaptive object scoring')
    parser.add_argument('--degree_feature_weight', default=0.6, type=float,
                        help='Weight of object feature similarity in image-aware degree scoring')
    parser.add_argument('--degree_spatial_weight', default=0.4, type=float,
                        help='Weight of spatial affinity in image-aware degree scoring')
    parser.add_argument('--coverage_ratio', default=0.25, type=float,
                        help='Fraction of adaptive budget reserved for coverage queries')
    parser.add_argument('--explore_ratio', default=0.20, type=float,
                        help='Fraction of adaptive budget reserved for global exploration queries')
    parser.add_argument('--min_queries_per_object', default=1, type=int,
                        help='Minimum number of coverage queries per anchor object')
    parser.add_argument('--max_anchor_objects', default=30, type=int,
                        help='Maximum number of confidence-ranked objects used as anchor candidates')
    parser.add_argument('--use_object_query_proj', action='store_true',
                        help='Use object_query_proj to generate object-conditioned triplet queries in adaptive mode')
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--ann_path', default='./data/vg/', type=str)
    parser.add_argument('--img_folder', default='/home/cong/Dokumente/tmp/data/visualgenome/images/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_topk_triplets', default=0, type=int,
                        help='If > 0, keep only the top-K triplet predictions during relation evaluation')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    if args.adaptive_query_budget and args.coverage_preserving_object_query:
        raise ValueError(
            "adaptive_query_budget and coverage_preserving_object_query are mutually exclusive"
        )
    if args.collect_query_importance and args.query_importance_selection:
        raise ValueError("collect_query_importance and query_importance_selection are mutually exclusive")
    if args.collect_query_importance:
        if args.adaptive_query_budget or args.coverage_preserving_object_query:
            raise ValueError("collect_query_importance requires adaptive query budgeting to be disabled")
        if args.sparse_query_k != args.num_triplets:
            raise ValueError(
                f"collect_query_importance requires sparse_query_k == num_triplets ({args.num_triplets}), got {args.sparse_query_k}"
            )
    if args.query_importance_selection and not args.query_importance_path:
        raise ValueError("query_importance_selection requires --query_importance_path")
    if not (0.0 <= args.prefix_keep_ratio <= 1.0):
        raise ValueError(f"prefix_keep_ratio must be in [0, 1], got {args.prefix_keep_ratio}")
    if args.sparse_query_k <= 0 or args.sparse_query_k > args.num_triplets:
        raise ValueError(
            f"sparse_query_k must be in [1, {args.num_triplets}], got {args.sparse_query_k}"
        )
    if args.budget_min <= 0 or args.budget_min > args.budget_max:
        raise ValueError(
            f"budget_min must be in [1, budget_max], got budget_min={args.budget_min}, budget_max={args.budget_max}"
        )
    if args.budget_score_scale <= 0:
        raise ValueError(
            f"budget_score_scale must be > 0, got {args.budget_score_scale}"
        )
    if (args.adaptive_query_budget or args.coverage_preserving_object_query) and args.budget_max > args.sparse_query_k:
        raise ValueError(
            f"budget_max must be <= sparse_query_k ({args.sparse_query_k}), got {args.budget_max}"
        )
    if args.adaptive_budget_floor <= 0:
        raise ValueError(
            f"adaptive_budget_floor must be > 0, got {args.adaptive_budget_floor}"
        )
    if args.enable_budget_floor and args.adaptive_budget_floor > args.budget_max:
        raise ValueError(
            f"adaptive_budget_floor must be <= budget_max when budget floor is enabled, got floor={args.adaptive_budget_floor}, budget_max={args.budget_max}"
        )
    if args.tail_boost_adaptive_budget:
        if not (0.0 <= args.low_complexity_threshold <= args.high_complexity_threshold <= args.very_high_complexity_threshold <= 1.0):
            raise ValueError(
                "Tail-boost complexity thresholds must satisfy 0 <= low <= high <= very_high <= 1"
            )
        if args.high_complexity_budget_floor < args.budget_min or args.high_complexity_budget_floor > args.budget_max:
            raise ValueError(
                f"high_complexity_budget_floor must be within [budget_min, budget_max], got {args.high_complexity_budget_floor}"
            )
        if args.very_high_complexity_budget_floor < args.high_complexity_budget_floor or args.very_high_complexity_budget_floor > args.budget_max:
            raise ValueError(
                f"very_high_complexity_budget_floor must be within [high_complexity_budget_floor, budget_max], got {args.very_high_complexity_budget_floor}"
            )
        if args.low_complexity_budget_cap < args.budget_min or args.low_complexity_budget_cap > args.budget_max:
            raise ValueError(
                f"low_complexity_budget_cap must be within [budget_min, budget_max], got {args.low_complexity_budget_cap}"
            )
    if args.percentile_tail_boost_adaptive_budget:
        if not (0.0 <= args.low_complexity_percentile <= args.high_complexity_percentile <= args.very_high_complexity_percentile <= 100.0):
            raise ValueError(
                "Percentile tail-boost cutoffs must satisfy 0 <= low <= high <= very_high <= 100"
            )
        if args.normal_complexity_budget_cap < args.budget_min or args.normal_complexity_budget_cap > args.budget_max:
            raise ValueError(
                f"normal_complexity_budget_cap must be within [budget_min, budget_max], got {args.normal_complexity_budget_cap}"
            )
    if not (0.0 <= args.coverage_ratio <= 1.0):
        raise ValueError(f"coverage_ratio must be in [0, 1], got {args.coverage_ratio}")
    if not (0.0 <= args.explore_ratio <= 1.0):
        raise ValueError(f"explore_ratio must be in [0, 1], got {args.explore_ratio}")
    if args.coverage_ratio + args.explore_ratio >= 1.0:
        raise ValueError(
            f"coverage_ratio + explore_ratio must be < 1.0, got {args.coverage_ratio + args.explore_ratio}"
        )
    if args.min_queries_per_object <= 0:
        raise ValueError(
            f"min_queries_per_object must be > 0, got {args.min_queries_per_object}"
        )
    if args.max_anchor_objects <= 0 or args.max_anchor_objects > args.num_entities:
        raise ValueError(
            f"max_anchor_objects must be in [1, {args.num_entities}], got {args.max_anchor_objects}"
        )
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu', weights_only=False)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        load_result = model_without_ddp.load_state_dict(
            checkpoint['model'],
            strict=not (args.adaptive_query_budget or args.coverage_preserving_object_query)
        )
        if args.adaptive_query_budget or args.coverage_preserving_object_query:
            print('Adaptive load_state_dict missing keys:', load_result.missing_keys)
            print('Adaptive load_state_dict unexpected keys:', load_result.unexpected_keys)
        # del checkpoint['optimizer']
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print('It is the {}th checkpoint'.format(checkpoint['epoch']))
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        # lr_scheduler.step()
        # if args.output_dir:
        #     checkpoint_paths = [output_dir / 'checkpoint.pth'] # anti-crash
        #     # extra checkpoint before LR drop and every 100 epochs
        #     if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
        #         checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        #     for checkpoint_path in checkpoint_paths:
        #         utils.save_on_master({
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
