srun -p optimal --quotatype=auto --gres=gpu:4 -J moco \
    python train_detection.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
    --num-gpus 4 MODEL.WEIGHTS ./checkpoints/moco_200ep_detection_pretrain.pth