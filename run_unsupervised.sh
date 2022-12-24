srun -p optimal --quotatype=auto --gres=gpu:4 -J moco \
    python main_unsupervised.py \
    -a resnet50 \
    --lr 0.03 \
    --batch-size 256 \
    --dist-url 'tcp://localhost:12345' --multiprocessing-distributed --world-size 1 --rank 0 \
    /mnt/petrelfs/share/imagenet/images