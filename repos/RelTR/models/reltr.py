# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer

class RelTR(nn.Module):
    """ RelTR: Relation Transformer for Scene Graph Generation """
    def __init__(self, backbone, transformer, num_classes, num_rel_classes, num_entities, num_triplets,
                 aux_loss=False, matcher=None, sparse_query_k=None, adaptive_query_budget=False,
                 coverage_preserving_object_query=False,
                 budget_min=50, budget_max=200, lambda_uncertainty=1.0, lambda_degree=1.0,
                 class_degree_prior=None, budget_count_ref=20.0, budget_uncertainty_ref=0.35,
                 budget_score_bias=0.45, budget_score_scale=0.18,
                 enable_budget_floor=False, adaptive_budget_floor=130, target_avg_budget=155.0,
                 tail_boost_adaptive_budget=False, high_complexity_threshold=0.70,
                 very_high_complexity_threshold=0.85, low_complexity_threshold=0.35,
                 percentile_tail_boost_adaptive_budget=False,
                 low_complexity_percentile=30.0, high_complexity_percentile=85.0,
                 very_high_complexity_percentile=95.0,
                 high_complexity_budget_floor=180, very_high_complexity_budget_floor=195,
                 low_complexity_budget_cap=145, normal_complexity_budget_cap=170,
                 degree_feature_weight=0.6, degree_spatial_weight=0.4,
                 coverage_ratio=0.25, explore_ratio=0.20,
                 min_queries_per_object=1, max_anchor_objects=30,
                 use_object_query_proj=False,
                 query_importance_selection=False, query_importance_path='',
                 query_selection_mode='importance', prefix_keep_ratio=0.5,
                 query_importance_stats=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of entity classes
            num_entities: number of entity queries
            num_triplets: number of coupled subject/object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_triplets = num_triplets
        self.sparse_query_k = num_triplets if sparse_query_k is None else sparse_query_k
        self.adaptive_query_budget = adaptive_query_budget
        self.coverage_preserving_object_query = coverage_preserving_object_query
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.budget_count_ref = budget_count_ref
        self.budget_uncertainty_ref = budget_uncertainty_ref
        self.budget_score_bias = budget_score_bias
        self.budget_score_scale = budget_score_scale
        self.enable_budget_floor = enable_budget_floor
        self.adaptive_budget_floor = adaptive_budget_floor
        self.target_avg_budget = target_avg_budget
        self.tail_boost_adaptive_budget = tail_boost_adaptive_budget
        self.high_complexity_threshold = high_complexity_threshold
        self.very_high_complexity_threshold = very_high_complexity_threshold
        self.low_complexity_threshold = low_complexity_threshold
        self.percentile_tail_boost_adaptive_budget = percentile_tail_boost_adaptive_budget
        self.low_complexity_percentile = low_complexity_percentile
        self.high_complexity_percentile = high_complexity_percentile
        self.very_high_complexity_percentile = very_high_complexity_percentile
        self.high_complexity_budget_floor = high_complexity_budget_floor
        self.very_high_complexity_budget_floor = very_high_complexity_budget_floor
        self.low_complexity_budget_cap = low_complexity_budget_cap
        self.normal_complexity_budget_cap = normal_complexity_budget_cap
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_degree = lambda_degree
        self.degree_feature_weight = degree_feature_weight
        self.degree_spatial_weight = degree_spatial_weight
        self.coverage_ratio = coverage_ratio
        self.explore_ratio = explore_ratio
        self.min_queries_per_object = min_queries_per_object
        self.max_anchor_objects = max_anchor_objects
        self.use_object_query_proj = use_object_query_proj
        self.query_importance_selection = query_importance_selection
        self.query_importance_path = query_importance_path
        self.query_selection_mode = query_selection_mode
        self.prefix_keep_ratio = prefix_keep_ratio
        self.query_importance_stats = query_importance_stats
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.object_conf_threshold = 0.3
        self.spatial_tau = 0.25
        self.last_adaptive_stats = None
        self.last_timing_stats = None
        self.runtime_low_complexity_threshold = low_complexity_threshold
        self.runtime_high_complexity_threshold = high_complexity_threshold
        self.runtime_very_high_complexity_threshold = very_high_complexity_threshold

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.entity_embed = nn.Embedding(num_entities, hidden_dim*2)
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim*3)
        self.so_embed = nn.Embedding(2, hidden_dim) # subject and object encoding
        if self.adaptive_query_budget:
            self.object_query_proj = nn.Linear(hidden_dim, hidden_dim * 3)
            nn.init.xavier_uniform_(self.object_query_proj.weight)
            nn.init.zeros_(self.object_query_proj.bias)
        # entity prediction
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # mask head
        self.so_mask_conv = nn.Sequential(torch.nn.Upsample(size=(28, 28)),
                                          nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 128))

        # predicate classification
        self.rel_class_embed = MLP(hidden_dim*2+128, hidden_dim, num_rel_classes + 1, 2)

        # subject/object label classfication and box regression
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the entity classification logits (including no-object) for all entity queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": the normalized entity boxes coordinates for all entity queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "sub_logits": the subject classification logits
               - "obj_logits": the object classification logits
               - "sub_boxes": the normalized subject boxes coordinates
               - "obj_boxes": the normalized object boxes coordinates
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        if self.coverage_preserving_object_query or self.adaptive_query_budget:
            hs, hs_t, so_masks = self._forward_adaptive(self.input_proj(src), mask, pos[-1])
            query_ids = torch.as_tensor(self.last_adaptive_stats['selected_query_ids'], device=src.device, dtype=torch.long).unsqueeze(0)
        else:
            selected_query_ids = self._select_query_ids(self.sparse_query_k, device=self.triplet_embed.weight.device)
            active_triplet_embed = self.triplet_embed.weight[selected_query_ids]
            hs, hs_t, so_masks, _, timing_stats = self.transformer(
                self.input_proj(src), mask, self.entity_embed.weight,
                active_triplet_embed, pos[-1], self.so_embed.weight
            )
            self.last_adaptive_stats = None
            self.last_timing_stats = {
                'adaptive_budget_time_ms': 0.0,
                'relation_decoder_time_ms': timing_stats.get('relation_decoder_time_ms', 0.0),
            }
            query_ids = selected_query_ids.unsqueeze(0)
        so_masks = so_masks.detach()
        so_masks = self.so_mask_conv(so_masks.view(-1, 2, src.shape[-2],src.shape[-1])).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2],-1)
        so_masks = self.so_mask_fc(so_masks)

        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)

        outputs_class = self.entity_class_embed(hs)
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()

        outputs_class_sub = self.sub_class_embed(hs_sub)
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()

        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()

        outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
               'rel_logits': outputs_class_rel[-1],
               'query_ids': query_ids}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj, outputs_class_rel)
        return out

    def _forward_adaptive(self, src, mask, pos_embed):
        if src.shape[0] != 1:
            raise NotImplementedError("Adaptive query budgeting currently supports batch_size=1 only.")

        src_flat, mask_flat, entity_pos, entity, _, _, pos_flat, _, _, h, w = \
            self.transformer._prepare_inputs(src, mask, self.entity_embed.weight,
                                             self.triplet_embed.weight[:self.sparse_query_k], pos_embed)
        memory = self.transformer.encode(src_flat, mask_flat, pos_flat)
        entity_states = self.transformer.decoder.forward_entity(
            entity, memory, memory_key_padding_mask=mask_flat, pos=pos_flat, entity_pos=entity_pos
        )
        hs = entity_states.transpose(1, 2)

        entity_logits = self.entity_class_embed(hs[-1, 0])
        entity_boxes = self.entity_bbox_embed(hs[-1, 0]).sigmoid()
        entity_features = hs[-1, 0]
        if memory.is_cuda:
            torch.cuda.synchronize(memory.device)
        budget_start = time.perf_counter()
        budget, debug_stats = self._build_adaptive_budget(entity_logits, entity_features, entity_boxes)
        if memory.is_cuda:
            torch.cuda.synchronize(memory.device)
        budget_time_ms = (time.perf_counter() - budget_start) * 1000.0
        selected_query_ids = self._select_query_ids(budget, device=self.triplet_embed.weight.device)
        relation_query_embed = self.triplet_embed.weight[selected_query_ids]
        debug_stats['query_conditioning_mode'] = 'disabled'
        debug_stats['object_query_projection'] = 'disabled'
        debug_stats['relation_query_embedding_modification'] = 'disabled'
        debug_stats['adaptive_budget_time_ms'] = float(budget_time_ms)
        debug_stats['selected_query_ids'] = selected_query_ids.detach().cpu().tolist()

        triplet_init, triplet_pos = self.transformer.prepare_triplet_queries(relation_query_embed, batch_size=1)
        if memory.is_cuda:
            torch.cuda.synchronize(memory.device)
        relation_start = time.perf_counter()
        triplet_states, sub_maps, obj_maps = self.transformer.decoder.forward_triplet(
            triplet_init, entity_states[:, :, 0:1, :], memory[:, 0:1, :],
            memory_key_padding_mask=mask_flat[0:1], pos=pos_flat[:, 0:1, :],
            triplet_pos=triplet_pos, so_pos=self.so_embed.weight
        )
        if memory.is_cuda:
            torch.cuda.synchronize(memory.device)
        relation_decoder_time_ms = (time.perf_counter() - relation_start) * 1000.0
        hs_t = triplet_states.transpose(1, 2)
        so_masks = torch.cat((sub_maps.reshape(sub_maps.shape[0], 1, sub_maps.shape[2], 1, h, w),
                              obj_maps.reshape(obj_maps.shape[0], 1, obj_maps.shape[2], 1, h, w)), dim=3)

        self.last_adaptive_stats = debug_stats
        self.last_timing_stats = {
            'adaptive_budget_time_ms': float(budget_time_ms),
            'relation_decoder_time_ms': float(relation_decoder_time_ms),
        }
        return hs, hs_t, so_masks

    def _select_query_ids(self, budget, device):
        budget = int(budget)
        if not self.query_importance_selection or self.query_importance_stats is None:
            return torch.arange(budget, device=device, dtype=torch.long)

        ranking = self.query_importance_stats['query_ranking'].to(device=device, dtype=torch.long)
        if self.query_selection_mode == 'importance':
            return ranking[:budget]
        if self.query_selection_mode == 'rare_balanced':
            rare_ranking = self.query_importance_stats['rare_balanced_ranking'].to(device=device, dtype=torch.long)
            return rare_ranking[:budget]
        if self.query_selection_mode == 'hybrid_prefix':
            num_prefix = min(int(budget * self.prefix_keep_ratio), budget)
            prefix_ids = torch.arange(num_prefix, device=device, dtype=torch.long)
            num_importance = budget - num_prefix
            selected = prefix_ids.tolist()
            prefix_set = set(selected)
            for q in ranking.tolist():
                if q in prefix_set:
                    continue
                selected.append(q)
                if len(selected) == budget:
                    break
            return torch.tensor(selected, device=device, dtype=torch.long)
        return ranking[:budget]

    @torch.no_grad()
    def preview_adaptive_complexity(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        src = self.input_proj(src)
        src_flat, mask_flat, entity_pos, entity, _, _, pos_flat, _, _, _, _ = \
            self.transformer._prepare_inputs(src, mask, self.entity_embed.weight,
                                             self.triplet_embed.weight[:self.sparse_query_k], pos[-1])
        memory = self.transformer.encode(src_flat, mask_flat, pos_flat)
        entity_states = self.transformer.decoder.forward_entity(
            entity, memory, memory_key_padding_mask=mask_flat, pos=pos_flat, entity_pos=entity_pos
        )
        hs = entity_states.transpose(1, 2)
        entity_logits = self.entity_class_embed(hs[-1, 0])
        entity_boxes = self.entity_bbox_embed(hs[-1, 0]).sigmoid()
        entity_features = hs[-1, 0]
        _, debug_stats = self._build_adaptive_budget(entity_logits, entity_features, entity_boxes)
        return debug_stats

    def set_runtime_tail_boost_thresholds(self, low_threshold, high_threshold, very_high_threshold):
        self.runtime_low_complexity_threshold = float(low_threshold)
        self.runtime_high_complexity_threshold = float(high_threshold)
        self.runtime_very_high_complexity_threshold = float(very_high_threshold)

    def _build_adaptive_budget(self, entity_logits, entity_features, entity_boxes):
        probs = F.softmax(entity_logits, dim=-1)
        fg_probs = probs[:, :-1]
        object_conf, object_class = fg_probs.max(dim=-1)
        uncertainty = object_conf * (1.0 - object_conf)

        object_mask = object_conf > self.object_conf_threshold
        if object_mask.sum() == 0:
            object_mask[object_conf.argmax()] = True

        selected_idx = torch.nonzero(object_mask, as_tuple=False).squeeze(1)
        candidate_uncertainty = uncertainty[selected_idx]
        candidate_classes = object_class[selected_idx]
        candidate_features = entity_features[selected_idx]
        candidate_boxes = entity_boxes[selected_idx]
        candidate_degree_scores = self._compute_image_degree_scores(candidate_features, candidate_boxes)
        candidate_scores = self.lambda_uncertainty * candidate_uncertainty + self.lambda_degree * candidate_degree_scores
        candidate_object_count = int(selected_idx.numel())
        selected_uncertainty = candidate_uncertainty
        selected_classes = candidate_classes
        selected_features = candidate_features
        selected_boxes = candidate_boxes
        degree_scores = candidate_degree_scores
        object_scores = candidate_scores

        prebudget_selected_object_count = candidate_object_count
        avg_uncertainty = float(selected_uncertainty.mean().item())
        avg_degree = float(degree_scores.mean().item())
        normalized_count = min(1.0, prebudget_selected_object_count / float(self.budget_count_ref))
        normalized_uncertainty = min(1.0, avg_uncertainty / float(self.budget_uncertainty_ref))
        complexity = 0.45 * normalized_count + 0.35 * normalized_uncertainty + 0.20 * avg_degree
        complexity = min(max(complexity, 0.0), 1.0)
        budget_ratio = torch.sigmoid(torch.tensor(
            (complexity - self.budget_score_bias) / self.budget_score_scale
        )).item()
        raw_budget = self.budget_min + budget_ratio * float(self.budget_max - self.budget_min)
        rounded_budget = int(round(raw_budget))
        budget = max(self.budget_min, min(rounded_budget, self.budget_max, self.sparse_query_k))
        if self.enable_budget_floor:
            budget = max(budget, min(self.adaptive_budget_floor, self.budget_max, self.sparse_query_k))
        low_threshold = self.runtime_low_complexity_threshold
        high_threshold = self.runtime_high_complexity_threshold
        very_high_threshold = self.runtime_very_high_complexity_threshold

        if self.tail_boost_adaptive_budget or self.percentile_tail_boost_adaptive_budget:
            if complexity >= high_threshold:
                budget = max(budget, min(self.high_complexity_budget_floor, self.budget_max, self.sparse_query_k))
            if complexity >= very_high_threshold:
                budget = max(budget, min(self.very_high_complexity_budget_floor, self.budget_max, self.sparse_query_k))
            elif self.percentile_tail_boost_adaptive_budget:
                budget = min(budget, min(self.normal_complexity_budget_cap, self.budget_max, self.sparse_query_k))
            if complexity <= low_threshold:
                budget = min(budget, min(self.low_complexity_budget_cap, self.budget_max, self.sparse_query_k))
        budget = min(budget, self.sparse_query_k)

        debug_stats = {
            'algorithm': 'SOAAQB',
            'image_budget': int(budget),
            'raw_budget': float(raw_budget),
            'rounded_budget': int(rounded_budget),
            'detected_object_count': int(candidate_object_count),
            'avg_uncertainty': avg_uncertainty,
            'avg_degree': avg_degree,
            'normalized_object_count': float(normalized_count),
            'normalized_uncertainty': float(normalized_uncertainty),
            'complexity_score': float(complexity),
            'budget_ratio': float(budget_ratio),
            'budget_floor_enabled': bool(self.enable_budget_floor),
            'adaptive_budget_floor': int(self.adaptive_budget_floor),
            'target_avg_budget': float(self.target_avg_budget),
            'tail_boost_adaptive_budget': bool(self.tail_boost_adaptive_budget),
            'percentile_tail_boost_adaptive_budget': bool(self.percentile_tail_boost_adaptive_budget),
            'high_complexity_threshold': float(high_threshold),
            'very_high_complexity_threshold': float(very_high_threshold),
            'low_complexity_threshold': float(low_threshold),
            'low_complexity_percentile': float(self.low_complexity_percentile),
            'high_complexity_percentile': float(self.high_complexity_percentile),
            'very_high_complexity_percentile': float(self.very_high_complexity_percentile),
            'high_complexity_budget_floor': int(self.high_complexity_budget_floor),
            'very_high_complexity_budget_floor': int(self.very_high_complexity_budget_floor),
            'low_complexity_budget_cap': int(self.low_complexity_budget_cap),
            'normal_complexity_budget_cap': int(self.normal_complexity_budget_cap),
            'is_high_complexity': bool(complexity >= high_threshold),
            'is_very_high_complexity': bool(complexity >= very_high_threshold),
            'is_low_complexity': bool(complexity <= low_threshold),
            'selected_object_indices': selected_idx.detach().cpu().tolist(),
            'selected_object_classes': selected_classes.detach().cpu().tolist(),
            'object_uncertainty': selected_uncertainty.detach().cpu().tolist(),
            'object_degree_scores': degree_scores.detach().cpu().tolist(),
            'object_scores': object_scores.detach().cpu().tolist(),
            'object_allocated_budgets': [],
            'final_selected_query_count': int(budget),
            'object_pruning': 0,
            'object_query_projection': 'disabled',
            'relation_query_embedding_modification': 'disabled',
        }
        return budget, debug_stats

    def _compute_image_degree_scores(self, object_features, object_boxes):
        num_objects = object_features.shape[0]
        if num_objects <= 1:
            return torch.zeros(num_objects, device=object_features.device)

        normalized_features = F.normalize(object_features, dim=-1)
        feature_similarity = torch.mm(normalized_features, normalized_features.t())
        feature_similarity = (feature_similarity + 1.0) * 0.5
        feature_similarity.fill_diagonal_(0.0)

        centers = object_boxes[:, :2]
        center_dist = torch.cdist(centers, centers, p=2)
        spatial_proximity = torch.exp(-center_dist / 0.25)
        spatial_proximity.fill_diagonal_(0.0)

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(object_boxes)
        overlap_iou, _ = box_ops.box_iou(boxes_xyxy, boxes_xyxy)
        overlap_iou.fill_diagonal_(0.0)

        spatial_score = 0.5 * spatial_proximity + 0.5 * overlap_iou
        pair_affinity = self.degree_feature_weight * feature_similarity + self.degree_spatial_weight * spatial_score
        degree_scores = pair_affinity.mean(dim=1)

        min_score = degree_scores.min()
        max_score = degree_scores.max()
        if max_score > min_score:
            degree_scores = (degree_scores - min_score) / (max_score - min_score + 1e-6)
        else:
            degree_scores = torch.zeros_like(degree_scores)
        return degree_scores


    def _allocate_object_budgets(self, object_scores, total_budget):
        num_objects = int(object_scores.numel())
        base = torch.ones(num_objects, dtype=torch.long, device=object_scores.device)
        remaining = total_budget - num_objects
        if remaining <= 0:
            return base

        score_sum = object_scores.sum()
        if score_sum <= 0:
            weights = torch.full_like(object_scores, 1.0 / num_objects)
        else:
            weights = object_scores / score_sum

        extra_float = weights * remaining
        extra_floor = torch.floor(extra_float).long()
        remainder = remaining - int(extra_floor.sum().item())
        if remainder > 0:
            fractional = extra_float - extra_floor.float()
            remainder_idx = torch.topk(fractional, k=remainder).indices
            extra_floor[remainder_idx] += 1
        return base + extra_floor

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                 'rel_logits': g}
                for a, b, c, d, e, f, g in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                               outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                               outputs_class_rel[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for RelTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_rel_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.num_rel_classes = 51 if num_classes == 151 else 31 # Using entity class numbers to adapt rel class numbers
        empty_weight_rel = torch.ones(num_rel_classes+1)
        empty_weight_rel[-1] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Entity/subject/object Classification loss
        """
        assert 'pred_logits' in outputs

        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o

        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
        target_relo_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])

        target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64, device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64, device=obj_logits.device)

        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o

        target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)
        src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

        loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2]*0.5, indices[3]*0.5), dim=-1)
        losses = {'loss_ce': (loss_ce * loss_weight).sum()/self.empty_weight[target_classes].sum()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['rel_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):
        """Compute the predicate classification loss
        """
        assert 'rel_logits' in outputs

        src_logits = outputs['rel_logits']
        idx = self._get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J,2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"])+len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    num_classes = 151 if args.dataset != 'oi' else 289 # some entity categories in OIV6 are deactivated.
    num_rel_classes = 51 if args.dataset != 'oi' else 31

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    class_degree_prior = None
    query_importance_stats = None
    if getattr(args, 'query_importance_selection', False):
        query_importance_stats = torch.load(args.query_importance_path, map_location='cpu', weights_only=False)
    model = RelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes = num_rel_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss,
        matcher=matcher,
        sparse_query_k=args.sparse_query_k,
        adaptive_query_budget=args.adaptive_query_budget,
        coverage_preserving_object_query=args.coverage_preserving_object_query,
        budget_min=args.budget_min,
        budget_max=args.budget_max,
        lambda_uncertainty=args.lambda_uncertainty,
        lambda_degree=args.lambda_degree,
        class_degree_prior=class_degree_prior,
        budget_count_ref=args.budget_count_ref,
        budget_uncertainty_ref=args.budget_uncertainty_ref,
        budget_score_bias=args.budget_score_bias,
        budget_score_scale=args.budget_score_scale,
        enable_budget_floor=args.enable_budget_floor,
        adaptive_budget_floor=args.adaptive_budget_floor,
        target_avg_budget=args.target_avg_budget,
        tail_boost_adaptive_budget=args.tail_boost_adaptive_budget,
        high_complexity_threshold=args.high_complexity_threshold,
        very_high_complexity_threshold=args.very_high_complexity_threshold,
        low_complexity_threshold=args.low_complexity_threshold,
        percentile_tail_boost_adaptive_budget=args.percentile_tail_boost_adaptive_budget,
        low_complexity_percentile=args.low_complexity_percentile,
        high_complexity_percentile=args.high_complexity_percentile,
        very_high_complexity_percentile=args.very_high_complexity_percentile,
        high_complexity_budget_floor=args.high_complexity_budget_floor,
        very_high_complexity_budget_floor=args.very_high_complexity_budget_floor,
        low_complexity_budget_cap=args.low_complexity_budget_cap,
        normal_complexity_budget_cap=args.normal_complexity_budget_cap,
        degree_feature_weight=args.degree_feature_weight,
        degree_spatial_weight=args.degree_spatial_weight,
        coverage_ratio=args.coverage_ratio,
        explore_ratio=args.explore_ratio,
        min_queries_per_object=args.min_queries_per_object,
        max_anchor_objects=args.max_anchor_objects,
        use_object_query_proj=args.use_object_query_proj,
        query_importance_selection=args.query_importance_selection,
        query_importance_path=args.query_importance_path,
        query_selection_mode=args.query_selection_mode,
        prefix_keep_ratio=args.prefix_keep_ratio,
        query_importance_stats=query_importance_stats)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rel'] = args.rel_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', "relations"]

    criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
