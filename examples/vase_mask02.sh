#!/bin/bash

# 设置默认参数值
MASK_RATIO=${1:-0.2}
LATENT_DIM=${2:-128}
HIDDEN_DIM=${3:-256}
ACTIVATION=${4:-'leaky'}
EN_LAYERS=${5:-2}
DE_LAYERS=${6:-2}
DROPOUT_RATE=${7:-0.5}
NUM_LAYERS=${8:-2}
RNN_TYPE=${9:-'lstm'}
FUSE_METHOD=${10:-'prod'}
BATCH_SIZE=${11:-256}
EVAL_STEPS=${12:-300}
SAVE_STEPS=${13:-2000}
USE_SKIP_CONNECTION=${14:-'--use_skip_connection'}
USE_WANDB=${15:-'--use_wandb'}
USE_TENSORBOARD=${16:-''}
EPOCHS=${17:-100}
BETA=${18:-0.02}
LR_WARMUP_STEPS=${19:-200}
LR_DECAY_FACTOR=${20:-0.98}
LR_DECAY_STEPS=${21:-800}
LEARNING_RATE=${22:-1e-4}
GRAD_CLIP_VALUE=${23:-1.0}
PROJ_NAME=${24:-'vase_mask02'}
RUN_TIME=${25:-1}

# 重复执行 main.py
for ((i=1; i<=RUN_TIME; i++)); do
    echo "🔁 Running main.py, round $i/$RUN_TIME..."
    python main.py \
        --mask_ratio $MASK_RATIO \
        --latent_dim $LATENT_DIM \
        --hidden_dim $HIDDEN_DIM \
        --activation $ACTIVATION \
        --en_layers $EN_LAYERS \
        --de_layers $DE_LAYERS \
        --dropout_rate $DROPOUT_RATE \
        --num_layers $NUM_LAYERS \
        --rnn $RNN_TYPE \
        --fuse $FUSE_METHOD \
        --batch_size $BATCH_SIZE \
        --eval_steps $EVAL_STEPS \
        --save_steps $SAVE_STEPS \
        $USE_SKIP_CONNECTION \
        $USE_WANDB \
        $USE_TENSORBOARD \
        --epoch_nums $EPOCHS \
        --beta $BETA \
        --lr_warmup_steps $LR_WARMUP_STEPS \
        --lr_decay_factor $LR_DECAY_FACTOR \
        --lr_decay_steps $LR_DECAY_STEPS \
        --lr $LEARNING_RATE \
        --grad_clip_value $GRAD_CLIP_VALUE \
        --proj_name $PROJ_NAME
    echo "✅ Done round $i"
done