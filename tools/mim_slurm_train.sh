#!/usr/bin/env bash

set -x

PARTITION=$1
TASK=$2
CONFIG=$3
WORK_DIR=$4
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH='.':$PYTHONPATH mim train $TASK $CONFIG \
    --launcher slurm -G $GPUS \
    --gpus-per-node $GPUS_PER_NODE \
    --cpus-per-task $CPUS_PER_TASK \
    --partition $PARTITION \
    --work-dir $WORK_DIR \
    --srun-args "$SRUN_ARGS" \
    $PY_ARGS
