import os

NEGA_CTX = 2
LOG = 1
distance_weight = 0.1
negative_weight = 1
nega_nega_weight = 0.05
open_set_method = 'MSP'
open_score = 'OE'
clip_backbone = 'ViT-B/16'
stage = 1
batch_size = 64
dataset  = "OOD_ImageNet_dtd"
few_shot = 16

os.system('CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python osr_nega_prompt.py --dataset {dataset} --batch_size {batch_size} --NEGA_CTX {NEGA_CTX} --LOG {LOG}\
--distance_weight {distance_weight}  --negative_weight {negative_weight} --open_score {open_score}\
--clip_backbone {clip_backbone} --nega_nega_weight {nega_nega_weight} --stage {stage} \
--POMP {POMP} --POMP_k {POMP_k} --few_shot {few_shot}'.format(
    dataset = dataset, batch_size = batch_size, NEGA_CTX = NEGA_CTX, distance_weight=distance_weight,
    negative_weight = negative_weight,  open_score = open_score, clip_backbone=clip_backbone,
    nega_nega_weight = nega_nega_weight, LOG = LOG, stage = 1, few_shot = 16))


os.system('CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python osr_nega_prompt.py --dataset {dataset} --batch_size {batch_size} --NEGA_CTX {NEGA_CTX} --LOG {LOG}\
--distance_weight {distance_weight}  --negative_weight {negative_weight} --open_score {open_score}\
--clip_backbone {clip_backbone} --nega_nega_weight {nega_nega_weight} --stage {stage} \
--POMP {POMP} --POMP_k {POMP_k} --few_shot {few_shot}'.format(
    dataset = dataset, batch_size = batch_size, NEGA_CTX = NEGA_CTX, distance_weight=distance_weight,
    negative_weight = negative_weight,  open_score = open_score, clip_backbone=clip_backbone,
    nega_nega_weight = nega_nega_weight, LOG = LOG, stage = 3, few_shot = 16))


datasets = ["OOD_ImageNet_dtd", "OOD_ImageNet_iNaturalist", "OOD_ImageNet_places365", "OOD_ImageNet_SUN"]
ori_dataset = "ImageNet1k_dtd" 
for dataset in datasets:
   os.system('CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
        python osr_nega_prompt.py --dataset {dataset} --ori_dataset {ori_dataset} --NEGA_CTX {NEGA_CTX} --CSC {CSC}  --LOG {LOG}\
        --distance_weight {distance_weight} --negative_weight {negative_weight} --open_score {open_score}\
        --clip_backbone {clip_backbone} --nega_nega_weight {nega_nega_weight}  --stage {stage} \
        --few_shot {few_shot} --open_set_method {open_set_method}'.format(
            dataset = dataset, ori_dataset = ori_dataset, NEGA_CTX = NEGA_CTX, distance_weight=distance_weight,
            negative_weight = negative_weight, open_score = open_score, clip_backbone=clip_backbone,
            nega_nega_weight = nega_nega_weight, LOG = LOG, stage = stage, open_set_method = open_set_method, few_shot = few_shot))