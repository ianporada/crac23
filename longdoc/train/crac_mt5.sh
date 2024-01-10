## training preco

salloc --gres=gpu:rtx8000:1 -c 4 --mem=10G -t 24:00:00 --partition=unkillable

#

module load anaconda/3
conda activate $SCRATCH/envs/longformer

export TRANSFORMERS_CACHE="${SCRATCH}/cache/transformers_cache"
export HF_DATASETS_CACHE="${SCRATCH}/cache/datasets_cache"

PROJECT_DIR=$SCRATCH/crac23

LONGDOC_DIR=$PROJECT_DIR/longdoc
LONGDOC_DATA_DIR=$LONGDOC_DIR/data

#### train dataset

OUTPUT_DIR=$LONGDOC_DIR/models/crac/mt5

export WANDB_DIR=$OUTPUT_DIR/wandb

mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

cd ~/fast-coref/src
python -m main experiment=crac \
    paths.base_data_dir=$LONGDOC_DATA_DIR \
    paths.base_model_dir=$OUTPUT_DIR \
    use_wandb=True \
    model/doc_encoder/transformer=mt5_base

#### train dataset

OUTPUT_DIR=$LONGDOC_DIR/models/crac/mt5_large

export WANDB_DIR=$OUTPUT_DIR/wandb

mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

cd ~/fast-coref/src
python -m main experiment=crac_mt5_large \
    paths.base_data_dir=$LONGDOC_DATA_DIR \
    paths.base_model_dir=$OUTPUT_DIR \
    use_wandb=True

####

#### --partition=short-unkillable

salloc --gres=gpu:a100l:1 -c 4 --mem=10G -t 24:00:00 --partition=unkillable
salloc --gres=gpu:a100l:1 -c 4 --mem=10G -t 24:00:00 --partition=unkillable