# srun -p optimal --quotatype=auto --gres=gpu:8 -J moco \
#     python main_lincls.py \
#     -a resnet50 \
#     --lr 30.0 \
#     --batch-size 256 \
#     --pretrained checkpoints/moco_200ep_pretrain.pth.tar \
#     --dist-url 'tcp://localhost:12345' --multiprocessing-distributed --world-size 1 --rank 0 \
#     /mnt/petrelfs/share/imagenet/images


srun -p iopen --quotatype=auto --gres=gpu:8 -J moco \
    python main_lincls.py \
    -a resnet50 \
    --lr 30.0 \
    --batch-size 256 \
    --pretrained checkpoints/moco_200ep_pretrain.pth.tar \
    --dist-url 'tcp://localhost:12345' --multiprocessing-distributed --world-size 1 --rank 0 \
    /mnt/petrelfs/share/imagenet/images