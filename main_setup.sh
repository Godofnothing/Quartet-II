#!/bin/bash

# Stop if any command fails and on unbound variables
set -eu

# ==========================================
# This is a simplified script showing the minimal setup to run Quartet V2 pre-training.
# It's recommended to put https://huggingface.co/datasets/ISTA-DASLab/C4-tokenized-llama2 in /dev/shm/datasets/c4 instead of re-tokenizing it.
# ==========================================


# Select specific parameters based on calculated indices
CURRENT_MODEL_CFG="30M:6:640:5:0.0012:3000000000"
CURRENT_MULT="0.25"
CURRENT_SETUP="128:false:false:false:true"

# Parse Model Config
IFS=":" read -r MODEL_SIZE_PREFIX N_LAYER N_EMBD N_HEAD LR BASE_TOKENS <<< "$CURRENT_MODEL_CFG"

# Calculate Tokens (using python for float math)
TOKENS=$(python3 -c "print(int($BASE_TOKENS * $CURRENT_MULT))")

# Parse Quant Setup
IFS=":" read -r HADAMARD_DIM DELAYED_AMAX DISABLE_FORWARD_QUANT DISABLE_BACKWARD_QUANT FOUR_OVER_SIX <<< "$CURRENT_SETUP"

# ==========================================
# 1. Static Environment Setup
# ==========================================

echo "START TIME: $(date)"
echo "Running on host: $(hostname)"
echo "Config: ${MODEL_SIZE_PREFIX} | Multiplier: ${CURRENT_MULT} | Tokens: ${TOKENS}"
echo "Scheme: Hadamard DIM=${HADAMARD_DIM}, delayed amax=${DELAYED_AMAX}"

export VOCAB_SIZE=32000 
export BATCH_SIZE=128
export ACC_STEPS=4
export SEQUENCE_LENGTH=512
export DATASET="c4"
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export WANDB_ENTITY=ist


export DATASET_BUFFER="/dev/shm/datasets"

# ==========================================
# 2. Quantization Configuration
# ==========================================

# Special
export SPECIAL_SCHEME="quartet_v2"
export SPECIAL_SCHEME_KWARGS="{\"hadamard_dim\": $HADAMARD_DIM, \"delayed_amax\": $DELAYED_AMAX, \"disable_forward_quant\": ${DISABLE_FORWARD_QUANT}, \"disable_backward_quant\": ${DISABLE_BACKWARD_QUANT}, \"four_over_six\": ${FOUR_OVER_SIX}}"

# ==========================================
# 3. Calculation & Execution
# ==========================================

export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

# WandB Prefix
SETUP_STR="${HADAMARD_DIM};${DELAYED_AMAX};${DISABLE_FORWARD_QUANT};${DISABLE_BACKWARD_QUANT}"
if [ "${FOUR_OVER_SIX}" = "true" ]; then
    SETUP_STR="${SETUP_STR};${FOUR_OVER_SIX}"
fi
WANDB_PREFIX="${MODEL_SIZE_PREFIX}-TOK${TOKENS}-${SPECIAL_SCHEME}@${SETUP_STR}-${DATASET}"

echo "Launching torchrun..."

torchrun --nproc_per_node=4 ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --datasets-dir $DATASET_BUFFER \
    --latest-ckpt-interval 1000 \
    --model llama \
    --vocab-size $VOCAB_SIZE \
    --compile \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --wandb \
    --wandb-project "backprop-scaling-laws" \
    --wandb-run-prefix "${WANDB_PREFIX}" \
    --log-interval 1 \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --special-scheme ${SPECIAL_SCHEME} \
    --special-scheme-kwargs "${SPECIAL_SCHEME_KWARGS}"

echo "END TIME: $(date)"
