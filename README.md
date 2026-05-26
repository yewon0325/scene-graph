# Scene Graph Project

## 연구 개요

본 프로젝트는 Scene Graph Generation(SGG)에서 관계 후보 수가 객체 수에 따라 급격히 증가하는 `relation candidate explosion` 문제를 줄이는 방향을 탐구하기 위한 연구 작업 공간이다. 현재 연구의 중심 질문은 다음과 같다.

- one-stage/query-based SGG에서 relation query 수를 줄여도 성능을 얼마나 유지할 수 있는가
- fixed sparse query와 adaptive query budgeting 사이의 성능-효율 trade-off는 어떠한가
- 평균 query budget을 낮추면서도 `R@100`, `mR@100`을 얼마나 보존할 수 있는가

현재 기준 baseline은 `RelTR`이며, 향후 `Scene-Graph-Benchmark.pytorch` 계열 two-stage baseline과 비교 분석하는 것을 목표로 한다.

## 연구 상태

현재까지 완료된 핵심 단계는 다음과 같다.

1. 실험 환경 구축
2. Visual Genome 기반 RelTR baseline 재현
3. fixed sparse / post-hoc top-K / adaptive query budgeting 실험
4. query importance statistics 수집 및 selection 실험 준비

즉 현재 단계는 `baseline 재현 완료 후 효율화 실험 및 비교 프레임 정리 단계`로 볼 수 있다.

## 저장소 구성

- `repos/RelTR`
  현재 one-stage/query-based baseline 및 query budget 관련 실험의 주 작업 공간
- `repos/Scene-Graph-Benchmark.pytorch`
  two-stage SGG baseline 분석 및 추후 비교 실험용 코드베이스

## RelTR 코드베이스 메모

RelTR는 query-based one-stage Scene Graph Generation 모델이다. 원본 README는 주로 Python 3.6 / PyTorch 1.6 환경을 기준으로 작성되어 있지만, 현재 로컬 실험은 수정된 GPU 환경과 호환성 패치를 반영한 상태에서 진행하고 있다.

따라서 실제 baseline 재현과 후속 실험은 원본 설치 절차 전체를 그대로 따르기보다, 이 프로젝트 루트 README에 정리된 실행 명령, 데이터 경로, 체크포인트 경로, 그리고 로컬 코드 수정 사항을 기준으로 수행한다.

## 실험 원칙

현재 adaptive 계열 실험에서는 아래 기능을 메인 proposed method에서 사용하지 않는 방향으로 정리하고 있다.

- object pruning
- object filtering
- object query projection
- relation query embedding modification

즉 pretrained RelTR의 learned relation query distribution을 최대한 유지한 상태에서, `query budget 자체를 조절하는 방식`을 우선 실험 대상으로 삼는다.

## 실험 환경

- OS: Windows + WSL Ubuntu
- Editor: VS Code (WSL)
- Python: Miniconda
- 대표 conda env: `sgg_onestage_gpu`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgg_onestage_gpu
```

## 데이터 및 체크포인트 경로

```text
RelTR code
/home/com/sgg-project/repos/scene-graph/repos/RelTR

Visual Genome data
/home/com/sgg-project/repos/RelTR/data/vg/

Pretrained checkpoint
/home/com/sgg-project/repos/RelTR/ckpt/checkpoint0149.pth
```

## 대표 실험 흐름

### 1. Baseline evaluation

```bash
cd /home/com/sgg-project/repos/scene-graph/repos/RelTR

python -u main.py \
  --dataset vg \
  --img_folder /home/com/sgg-project/repos/RelTR/data/vg/images/ \
  --ann_path /home/com/sgg-project/repos/RelTR/data/vg/ \
  --eval \
  --batch_size 1 \
  --resume /home/com/sgg-project/repos/RelTR/ckpt/checkpoint0149.pth
```

### 2. Fixed sparse evaluation

예시: `K=180`

```bash
python -u main.py \
  --dataset vg \
  --img_folder /home/com/sgg-project/repos/RelTR/data/vg/images/ \
  --ann_path /home/com/sgg-project/repos/RelTR/data/vg/ \
  --eval \
  --batch_size 1 \
  --resume /home/com/sgg-project/repos/RelTR/ckpt/checkpoint0149.pth \
  --sparse_query_k 180 \
  --output_dir fixed_sparse_k180_eval \
  2>&1 | tee fixed_sparse_k180.log
```

### 3. Post-hoc top-K evaluation

예시: `K=100`

```bash
python -u main.py \
  --dataset vg \
  --img_folder /home/com/sgg-project/repos/RelTR/data/vg/images/ \
  --ann_path /home/com/sgg-project/repos/RelTR/data/vg/ \
  --eval \
  --batch_size 1 \
  --resume /home/com/sgg-project/repos/RelTR/ckpt/checkpoint0149.pth \
  --sparse_query_k 200 \
  --eval_topk_triplets 100 \
  --output_dir eval_topk_100 \
  2>&1 | tee eval_topk_100.log
```

### 4. Safe Object-Aware Adaptive Query Budgeting

예시: 평균 budget 155 근처를 목표로 한 설정

```bash
python -u main.py \
  --dataset vg \
  --img_folder /home/com/sgg-project/repos/RelTR/data/vg/images/ \
  --ann_path /home/com/sgg-project/repos/RelTR/data/vg/ \
  --eval \
  --batch_size 1 \
  --resume /home/com/sgg-project/repos/RelTR/ckpt/checkpoint0149.pth \
  --sparse_query_k 200 \
  --adaptive_query_budget \
  --budget_min 125 \
  --budget_max 185 \
  --budget_score_bias 0.68 \
  --budget_score_scale 0.24 \
  --enable_budget_floor \
  --adaptive_budget_floor 125 \
  --target_avg_budget 155 \
  --lambda_uncertainty 0.5 \
  --lambda_degree 1.0 \
  --output_dir soaaqb_budget155_b068_s024 \
  2>&1 | tee soaaqb_budget155_b068_s024.log
```

### 5. Query importance statistics 수집

```bash
python -u main.py \
  --dataset vg \
  --img_folder /home/com/sgg-project/repos/RelTR/data/vg/images/ \
  --ann_path /home/com/sgg-project/repos/RelTR/data/vg/ \
  --eval \
  --batch_size 1 \
  --resume /home/com/sgg-project/repos/RelTR/ckpt/checkpoint0149.pth \
  --sparse_query_k 200 \
  --collect_query_importance \
  --query_importance_output query_importance_stats.pt \
  --output_dir query_importance_collect \
  2>&1 | tee query_importance_collect.log
```

### 6. Query importance selection 평가

예시: `hybrid_prefix`

```bash
python -u main.py \
  --dataset vg \
  --img_folder /home/com/sgg-project/repos/RelTR/data/vg/images/ \
  --ann_path /home/com/sgg-project/repos/RelTR/data/vg/ \
  --eval \
  --batch_size 1 \
  --resume /home/com/sgg-project/repos/RelTR/ckpt/checkpoint0149.pth \
  --sparse_query_k 200 \
  --adaptive_query_budget \
  --budget_min 125 \
  --budget_max 185 \
  --budget_score_bias 0.62 \
  --budget_score_scale 0.25 \
  --enable_budget_floor \
  --adaptive_budget_floor 125 \
  --target_avg_budget 155 \
  --lambda_uncertainty 0.5 \
  --lambda_degree 1.0 \
  --query_importance_selection \
  --query_importance_path query_importance_stats.pt \
  --query_selection_mode hybrid_prefix \
  --prefix_keep_ratio 0.5 \
  --output_dir soaaqb163_query_importance_hybrid \
  2>&1 | tee soaaqb163_query_importance_hybrid.log
```

## 주요 평가 지표

RelTR 계열 실험에서는 아래 지표를 중심으로 비교한다.

- `R@20`, `R@50`, `R@100`
- `mR@20`, `mR@50`, `mR@100`
- `AP@[.50:.95]`, `AP@0.50`, `AR@100`
- `Test: Total time`
- `time / it`
- `relation_decoder_time_ms`
- `adaptive_budget_time_ms`
- `GPU max memory allocated`
- `GPU max memory reserved`
- `max mem`
- adaptive budget summary / histogram

## 코드 수정 핵심 파일

현재 실험에서 주로 수정된 파일은 아래와 같다.

- `repos/RelTR/main.py`
- `repos/RelTR/engine.py`
- `repos/RelTR/models/reltr.py`
- `repos/RelTR/models/transformer.py`
- `repos/RelTR/model_stats.py`

## 기록 및 관리 메모

- `repos/RelTR`는 상위 저장소에 submodule로 연결되어 있다.
- 대표 adaptive baseline은 submodule commit/tag와 상위 repo commit으로 함께 관리한다.
- 실험 로그, 출력 폴더, 임시 산출물은 기본적으로 버전 관리 대상이 아니라 로컬 실험 산출물로 취급한다.
- 이후 README는 단순 사용 설명서가 아니라, `실험 재현 절차 + 연구 진행 메모`를 함께 담는 문서로 유지한다.
