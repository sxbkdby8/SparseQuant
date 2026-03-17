EPOCHS=90
LR=1e-3


torchrun --nnodes=1 --nproc_per_node=8 --master_port 23432 timm_train.py \
    --model deit_small_distilled_patch16_224 \
    --sparsity-mode hybird \
    --mask-only \
    -b 128 \
    --opt adamw \
    --lr ${LR} \
    --weight-decay 0.01 \
    --epochs ${EPOCHS} \
    --warmup-epochs 0 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --sched cosine \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --aa rand-m8-inc1-mstd101 \
    --remode pixel --reprob 0.3 \
    --amp \
    --crop-pct 0.9 \
    --mean 0.5 0.5 0.5 \
    --std 0.5 0.5 0.5 \
    --output output/sparse_deit_small_patch16_224_${EPOCHS}_epochs_lr${LR}_.augreg_in1k.hybird \
    --scaling-range 1e1 1e2 \
    --tau-range 4 0.05 \
    --log-wandb \
    --sparse-weight-reg 1e-5 \
    --clip-grad 2.0 \
    --min-lr 1e-4 \
    --model-ema \
    --model-ema-decay 0.9998 \
    --prior-strength 3 \
    --sparse-budget /home/chenzhiqiang/MaskLLM-4V/deit_small_layerwise_importance_20251223.pkl \
    --sparse-checkpoint /home/chenzhiqiang/MaskLLM-4V/output/pruned/deit_small_distilled_patch16_224_oneshot_mag_hybird.augreg_in1k.pt \
