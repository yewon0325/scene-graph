# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
Train and eval functions used in main.py
"""
import json
import math
from pathlib import Path
import sys
from typing import Iterable
import numpy as np

import torch

from datasets.coco_eval import CocoEvaluator
import util.misc as utils
from util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, evaluate_recall
from lib.openimages_evaluation import task_evaluation_sg

def _select_topk_triplets(pred_sub_scores, pred_obj_scores, rel_scores, topk):
    num_triplets = rel_scores.shape[0]
    if topk is None or topk <= 0 or topk >= num_triplets:
        return slice(None)

    # Rank triplets by the joint confidence of predicate, subject, and object predictions.
    rel_conf = rel_scores.max(-1).values
    joint_conf = rel_conf * pred_sub_scores * pred_obj_scores
    return torch.topk(joint_conf, k=topk).indices

def _normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    x_min = x.min()
    x_max = x.max()
    if x_max <= x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min + 1e-6)

def _init_query_importance_stats(num_queries, num_predicates):
    return {
        'total_predictions': np.zeros(num_queries, dtype=np.float64),
        'correct_triplet_hits': np.zeros(num_queries, dtype=np.float64),
        'predicate_hits': np.zeros((num_queries, num_predicates), dtype=np.float64),
        'score_sum': np.zeros(num_queries, dtype=np.float64),
        'predicate_frequency': np.zeros(num_predicates, dtype=np.float64),
        'source_split': 'val',
    }

def _update_query_importance_stats(stats, gt_entry, pred_entry):
    rel_scores = pred_entry['rel_scores']
    if rel_scores.size == 0:
        gt_rels = gt_entry['gt_relations']
        if gt_rels.size > 0:
            pred_ids = gt_rels[:, 2].astype(np.int64)
            pred_ids = pred_ids[(pred_ids >= 0) & (pred_ids < stats['predicate_frequency'].shape[0])]
            np.add.at(stats['predicate_frequency'], pred_ids, 1)
        return

    pred_rels = 1 + rel_scores.argmax(1)
    predicate_scores = rel_scores.max(1)
    relation_scores = np.column_stack((pred_entry['sub_scores'], pred_entry['obj_scores'], predicate_scores))
    sorted_idx = relation_scores.prod(1).argsort()[::-1]

    sorted_query_ids = pred_entry['query_ids'][sorted_idx].astype(np.int64)
    sorted_pred_rels = pred_rels[sorted_idx].astype(np.int64)
    sorted_score_sum = relation_scores.prod(1)[sorted_idx]

    gt_rels = gt_entry['gt_relations']
    if gt_rels.size > 0:
        pred_ids = gt_rels[:, 2].astype(np.int64)
        pred_ids = pred_ids[(pred_ids >= 0) & (pred_ids < stats['predicate_frequency'].shape[0])]
        np.add.at(stats['predicate_frequency'], pred_ids, 1)

    pred_to_gt, _, _ = evaluate_recall(
        gt_rels, gt_entry['gt_boxes'], gt_entry['gt_classes'],
        pred_rels,
        pred_entry['sub_boxes'], pred_entry['obj_boxes'],
        pred_entry['sub_scores'], pred_entry['obj_scores'],
        predicate_scores, pred_entry['sub_classes'], pred_entry['obj_classes'],
        iou_thresh=0.5, phrdet=False
    )

    for rank_idx, qid in enumerate(sorted_query_ids):
        if qid < 0 or qid >= stats['total_predictions'].shape[0]:
            continue
        stats['total_predictions'][qid] += 1
        stats['score_sum'][qid] += float(sorted_score_sum[rank_idx])
        if len(pred_to_gt[rank_idx]) > 0:
            stats['correct_triplet_hits'][qid] += 1
            matched_gt = gt_rels[np.unique(pred_to_gt[rank_idx]), 2].astype(np.int64)
            matched_gt = matched_gt[(matched_gt >= 0) & (matched_gt < stats['predicate_hits'].shape[1])]
            if matched_gt.size > 0:
                np.add.at(stats['predicate_hits'][qid], matched_gt, 1)

def _finalize_query_importance_stats(stats, args):
    total_predictions = stats['total_predictions']
    correct_triplet_hits = stats['correct_triplet_hits']
    predicate_hits = stats['predicate_hits']
    score_sum = stats['score_sum']
    predicate_frequency = stats['predicate_frequency']

    overall_score = correct_triplet_hits / np.sqrt(total_predictions + 1.0)
    rare_weight = 1.0 / np.sqrt(predicate_frequency + 1.0)
    rare_weighted_score = (predicate_hits * rare_weight[None, :]).sum(axis=1)
    importance_score = (
        args.alpha_query_overall * _normalize_np(overall_score) +
        args.alpha_query_rare * _normalize_np(rare_weighted_score) +
        args.alpha_query_score * _normalize_np(score_sum)
    )
    rare_rank_score = _normalize_np(overall_score) + 1.0 * _normalize_np(rare_weighted_score)
    query_ranking = np.argsort(importance_score)[::-1]
    rare_balanced_ranking = np.argsort(rare_rank_score)[::-1]

    return {
        'importance_score': torch.from_numpy(importance_score.astype(np.float32)),
        'overall_score': torch.from_numpy(overall_score.astype(np.float32)),
        'rare_weighted_score': torch.from_numpy(rare_weighted_score.astype(np.float32)),
        'score_sum': torch.from_numpy(score_sum.astype(np.float32)),
        'correct_triplet_hits': torch.from_numpy(correct_triplet_hits.astype(np.float32)),
        'total_predictions': torch.from_numpy(total_predictions.astype(np.float32)),
        'predicate_hits': torch.from_numpy(predicate_hits.astype(np.float32)),
        'predicate_frequency': torch.from_numpy(predicate_frequency.astype(np.float32)),
        'query_ranking': torch.from_numpy(query_ranking.astype(np.int64)),
        'rare_balanced_ranking': torch.from_numpy(rare_balanced_ranking.astype(np.int64)),
        'num_queries': int(total_predictions.shape[0]),
        'importance_source_split': stats['source_split'],
    }

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()
    adaptive_mode = getattr(args, 'adaptive_query_budget', False) or getattr(args, 'coverage_preserving_object_query', False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    adaptive_model = model.module if hasattr(model, 'module') else model

    if adaptive_mode and getattr(args, 'percentile_tail_boost_adaptive_budget', False):
        complexity_scores = []
        for samples, _targets in data_loader:
            samples = samples.to(device)
            preview_stats = adaptive_model.preview_adaptive_complexity(samples)
            complexity_scores.append(preview_stats['complexity_score'])
        complexity_scores = np.array(complexity_scores, dtype=np.float32)
        low_threshold = float(np.percentile(complexity_scores, args.low_complexity_percentile))
        high_threshold = float(np.percentile(complexity_scores, args.high_complexity_percentile))
        very_high_threshold = float(np.percentile(complexity_scores, args.very_high_complexity_percentile))
        adaptive_model.set_runtime_tail_boost_thresholds(low_threshold, high_threshold, very_high_threshold)
        print('[SOAAQB] percentile thresholds used:', {
            'low_percentile': float(args.low_complexity_percentile),
            'high_percentile': float(args.high_complexity_percentile),
            'very_high_percentile': float(args.very_high_complexity_percentile),
            'p_low_threshold': low_threshold,
            'p_high_threshold': high_threshold,
            'p_very_high_threshold': very_high_threshold,
        })

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('relation_decoder_time_ms', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    if adaptive_mode:
        metric_logger.add_meter('adaptive_budget', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('selected_queries', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('adaptive_budget_time_ms', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    header = 'Test:'

    # initilize evaluator
    # TODO merge evaluation programs
    if args.dataset == 'vg':
        evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
        if args.eval:
            evaluator_list = []
            for index, name in enumerate(data_loader.dataset.rel_categories):
                if index == 0:
                    continue
                evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        else:
            evaluator_list = None
    else:
        all_results = []

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    recall_stats = {}
    mean_recall_stats = {}

    adaptive_log_path = None
    if adaptive_mode and args.output_dir:
        adaptive_log_path = Path(args.output_dir) / 'adaptive_query_log.jsonl'
    raw_budget_values = []
    rounded_budget_values = []
    quantized_budget_values = []
    relation_decoder_time_values = []
    adaptive_budget_time_values = []
    high_complexity_count = 0
    very_high_complexity_count = 0
    low_complexity_count = 0
    query_selection_examples = []
    query_importance_stats = None
    if getattr(args, 'collect_query_importance', False):
        num_predicates = len(getattr(data_loader.dataset, 'rel_categories', []))
        query_importance_stats = _init_query_importance_stats(args.num_triplets, num_predicates)

    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        adaptive_stats = getattr(adaptive_model, 'last_adaptive_stats', None)
        timing_stats = getattr(adaptive_model, 'last_timing_stats', None)
        if timing_stats is not None:
            metric_logger.update(relation_decoder_time_ms=timing_stats.get('relation_decoder_time_ms', 0.0))
            relation_decoder_time_values.append(timing_stats.get('relation_decoder_time_ms', 0.0))
        if adaptive_mode and adaptive_stats is not None:
            metric_logger.update(adaptive_budget=adaptive_stats['image_budget'])
            metric_logger.update(selected_queries=adaptive_stats['final_selected_query_count'])
            if timing_stats is not None:
                metric_logger.update(adaptive_budget_time_ms=timing_stats.get('adaptive_budget_time_ms', 0.0))
                adaptive_budget_time_values.append(timing_stats.get('adaptive_budget_time_ms', 0.0))
            raw_budget_values.append(adaptive_stats['raw_budget'])
            rounded_budget_values.append(adaptive_stats['rounded_budget'])
            quantized_budget_values.append(adaptive_stats['image_budget'])
            high_complexity_count += int(adaptive_stats.get('is_high_complexity', False))
            very_high_complexity_count += int(adaptive_stats.get('is_very_high_complexity', False))
            low_complexity_count += int(adaptive_stats.get('is_low_complexity', False))
            if adaptive_log_path is not None:
                payload = dict(adaptive_stats)
                if timing_stats is not None:
                    payload.update(timing_stats)
                payload['image_id'] = int(targets[0]['image_id'].item())
                with adaptive_log_path.open('a') as f:
                    f.write(json.dumps(payload) + '\n')
        if 'query_ids' in outputs and len(query_selection_examples) < 1:
            query_selection_examples.append(outputs['query_ids'][0].detach().cpu().tolist()[:20])
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])

        if args.dataset == 'vg':
            evaluate_rel_batch(outputs, targets, evaluator, evaluator_list, args.eval_topk_triplets, query_importance_stats)
        else:
            evaluate_rel_batch_oi(outputs, targets, all_results, args.eval_topk_triplets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    if args.dataset == 'vg':
        recall_stats = evaluator['sgdet'].print_stats()
    else:
        task_evaluation_sg.eval_rel_results(all_results, 100, do_val=True, do_vis=False)

    if args.eval and args.dataset == 'vg':
        mean_recall_stats = calculate_mR_from_evaluator_list(evaluator_list, 'sgdet')

    if adaptive_mode and raw_budget_values:
        raw_budget_np = np.array(raw_budget_values)
        rounded_budget_np = np.array(rounded_budget_values)
        final_budget_np = np.array(quantized_budget_values)
        raw_bins = [50, 75, 100, 125, 150, 175, 201]
        raw_hist, raw_edges = np.histogram(raw_budget_np, bins=raw_bins)
        rounded_hist, rounded_edges = np.histogram(rounded_budget_np, bins=raw_bins)
        final_hist, final_edges = np.histogram(final_budget_np, bins=raw_bins)
        print('Adaptive raw budget summary:',
              {'min': float(raw_budget_np.min()),
               'mean': float(raw_budget_np.mean()),
               'max': float(raw_budget_np.max())})
        print('Adaptive rounded budget summary:',
              {'min': int(rounded_budget_np.min()),
               'mean': float(rounded_budget_np.mean()),
               'max': int(rounded_budget_np.max())})
        print('Adaptive final budget summary:',
              {'min': int(final_budget_np.min()),
               'mean': float(final_budget_np.mean()),
               'max': int(final_budget_np.max())})
        print('Adaptive raw budget histogram:',
              {f'[{int(raw_edges[i])},{int(raw_edges[i+1])})': int(raw_hist[i]) for i in range(len(raw_hist))})
        print('Adaptive rounded budget histogram:',
              {f'[{int(rounded_edges[i])},{int(rounded_edges[i+1])})': int(rounded_hist[i]) for i in range(len(rounded_hist))})
        print('Adaptive final budget histogram:',
              {f'[{int(final_edges[i])},{int(final_edges[i+1])})': int(final_hist[i]) for i in range(len(final_hist))})
        bucket_150_175 = int(final_hist[4]) if len(final_hist) > 4 else 0
        bucket_175_201 = int(final_hist[5]) if len(final_hist) > 5 else 0
        bucket_125_150 = int(final_hist[3]) if len(final_hist) > 3 else 0
        print('[SOAAQB] summary:', {
            'average adaptive_budget': float(final_budget_np.mean()),
            'min adaptive_budget': int(final_budget_np.min()),
            'max adaptive_budget': int(final_budget_np.max()),
            'budget histogram': {f'[{int(final_edges[i])},{int(final_edges[i+1])})': int(final_hist[i]) for i in range(len(final_hist))},
            'number of images in [125,150)': bucket_125_150,
            'number of images in [150,175)': bucket_150_175,
            'number of images in [175,201)': bucket_175_201,
            'number of high_complexity_images': int(high_complexity_count),
            'number of very_high_complexity_images': int(very_high_complexity_count),
            'number of low_complexity_images': int(low_complexity_count),
            'number of normal_complexity_images': int(len(final_budget_np) - high_complexity_count - low_complexity_count),
            'percentile thresholds used': {
                'low': float(adaptive_model.runtime_low_complexity_threshold),
                'high': float(adaptive_model.runtime_high_complexity_threshold),
                'very_high': float(adaptive_model.runtime_very_high_complexity_threshold),
            },
            'object pruning count': 0,
            'object query projection': 'disabled',
            'relation query embedding modification': 'disabled',
        })
        if adaptive_budget_time_values:
            print('[SOAAQB] timing:', {
                'average adaptive_budget_time_ms': float(np.mean(adaptive_budget_time_values)),
            })
    if relation_decoder_time_values:
        print('Relation decoder timing summary:', {
            'average relation_decoder_time_ms': float(np.mean(relation_decoder_time_values)),
        })
    if getattr(args, 'query_importance_selection', False):
        print('[QuerySelection]', {
            'mode': args.query_selection_mode,
            'adaptive_budget average': float(np.mean(quantized_budget_values)) if quantized_budget_values else float(args.sparse_query_k),
            'selected query count average': float(np.mean(quantized_budget_values)) if quantized_budget_values else float(args.sparse_query_k),
            'top selected query ids example': query_selection_examples[0] if query_selection_examples else [],
            'prefix_keep_ratio': float(args.prefix_keep_ratio),
            'importance path': args.query_importance_path,
            'query importance selection': 'enabled',
            'object pruning count': 0,
            'object query projection': 'disabled',
            'relation query embedding modification': 'disabled',
        })
    if torch.cuda.is_available():
        max_mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"GPU max memory allocated: {max_mem_allocated:.0f} MB")
        print(f"GPU max memory reserved: {max_mem_reserved:.0f} MB")
        print(f"max mem: {max_mem_allocated:.0f}")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if recall_stats:
        stats['sgdet_recall'] = recall_stats
    if mean_recall_stats:
        stats['sgdet_mean_recall'] = mean_recall_stats
    stats['triplet_query_budget'] = args.sparse_query_k
    if getattr(args, 'eval_topk_triplets', 0) > 0:
        stats['eval_topk_triplets'] = args.eval_topk_triplets
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    if query_importance_stats is not None:
        finalized = _finalize_query_importance_stats(query_importance_stats, args)
        torch.save(finalized, args.query_importance_output)
        print(f"[QueryImportance] importance source split: {finalized['importance_source_split']}")
        print(f"[QueryImportance] saved to {args.query_importance_output}")
        print(f"[QueryImportance] top-20 query ids: {finalized['query_ranking'][:20].tolist()}")
        print(f"[QueryImportance] bottom-20 query ids: {finalized['query_ranking'][-20:].tolist()}")

    return stats, coco_evaluator

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list, topk_triplets=0, query_importance_stats=None):
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
        rel_scores = outputs['rel_logits'][batch][:,1:-1].softmax(-1)
        query_ids = outputs['query_ids'][batch].cpu().clone().numpy() if 'query_ids' in outputs else np.arange(rel_scores.shape[0])
        keep = _select_topk_triplets(pred_sub_scores, pred_obj_scores, rel_scores, topk_triplets)
        keep_np = keep if isinstance(keep, slice) else keep.cpu().numpy()

        pred_entry = {'sub_boxes': sub_bboxes_scaled[keep_np],
                      'sub_classes': pred_sub_classes[keep].cpu().clone().numpy(),
                      'sub_scores': pred_sub_scores[keep].cpu().clone().numpy(),
                      'obj_boxes': obj_bboxes_scaled[keep_np],
                      'obj_classes': pred_obj_classes[keep].cpu().clone().numpy(),
                      'obj_scores': pred_obj_scores[keep].cpu().clone().numpy(),
                      'rel_scores': rel_scores[keep].cpu().clone().numpy(),
                      'query_ids': query_ids[keep_np]}

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)
        if query_importance_stats is not None:
            _update_query_importance_stats(query_importance_stats, gt_entry, pred_entry)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)


def evaluate_rel_batch_oi(outputs, targets, all_results, topk_triplets=0):

    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        rel_scores = outputs['rel_logits'][batch][:, :-1].softmax(-1)
        keep = _select_topk_triplets(pred_sub_scores, pred_obj_scores, rel_scores, topk_triplets)
        keep_np = keep if isinstance(keep, slice) else keep.cpu().numpy()

        relation_idx = target['rel_annotations'].cpu().numpy()
        gt_sub_boxes = target_bboxes_scaled[relation_idx[:, 0]]
        gt_sub_labels = target['labels'][relation_idx[:, 0]].cpu().clone().numpy()
        gt_obj_boxes = target_bboxes_scaled[relation_idx[:, 1]]
        gt_obj_labels = target['labels'][relation_idx[:, 1]].cpu().clone().numpy()

        img_result_dict = {'sbj_boxes': sub_bboxes_scaled[keep_np],
                           'sbj_labels': pred_sub_classes[keep].cpu().clone().numpy(),
                           'sbj_scores': pred_sub_scores[keep].cpu().clone().numpy(),
                           'obj_boxes': obj_bboxes_scaled[keep_np],
                           'obj_labels': pred_obj_classes[keep].cpu().clone().numpy(),
                           'obj_scores': pred_obj_scores[keep].cpu().clone().numpy(),
                           'prd_scores': rel_scores[keep].cpu().clone().numpy(),
                           'image': str(target['image_id'].item())+'.jpg',
                           'gt_sbj_boxes': gt_sub_boxes,
                           'gt_sbj_labels': gt_sub_labels,
                           'gt_obj_boxes': gt_obj_boxes,
                           'gt_obj_labels': gt_obj_labels,
                           'gt_prd_labels': relation_idx[:, 2]
                           }
        all_results.append(img_result_dict)
