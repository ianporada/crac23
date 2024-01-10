## training preco

salloc --gres=gpu:rtx8000:1 -c 4 --mem=16G -t 24:00:00 --partition=main

#

module load anaconda/3
conda activate $SCRATCH/envs/longformer

export TRANSFORMERS_CACHE="${SCRATCH}/cache/transformers_cache"
export HF_DATASETS_CACHE="${SCRATCH}/cache/datasets_cache"

PROJECT_DIR=$SCRATCH/crac23

LONGDOC_DIR=$PROJECT_DIR/longdoc
LONGDOC_DATA_DIR=$LONGDOC_DIR/data

#### train dataset

OUTPUT_DIR=$LONGDOC_DIR/models/crac/xlmroberta

export WANDB_DIR=$OUTPUT_DIR/wandb

mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

cd ~/fast-coref/src
python -m main experiment=crac \
    paths.base_data_dir=$LONGDOC_DATA_DIR \
    paths.base_model_dir=$OUTPUT_DIR \
    use_wandb=True

#### large

OUTPUT_DIR=$LONGDOC_DIR/models/crac/xlmroberta_large

export WANDB_DIR=$OUTPUT_DIR/wandb

mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

cd ~/fast-coref/src
python -m main experiment=crac_xlm_large \
    paths.base_data_dir=$LONGDOC_DATA_DIR \
    paths.base_model_dir=$OUTPUT_DIR \
    use_wandb=True

#### large continue

ENCODER_DIR=/network/scratch/p/poradaia/crac23/longdoc/models/crac/xlmroberta_large/coref_crac_eba9a9193140a40d490e4a6b9ccf0f49_2/doc_encoder
NEW_DIR=/network/scratch/p/poradaia/crac23/longdoc/models/crac/xlmroberta_large/coref_crac_6b26858dd0958b4d21880b6703dda415_2

OUTPUT_DIR=$LONGDOC_DIR/models/crac/xlmroberta_large

export WANDB_DIR=$OUTPUT_DIR/wandb

mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

cd ~/fast-coref/src
python -m main experiment=crac_xlm_large \
    paths.base_data_dir=$LONGDOC_DATA_DIR \
    paths.base_model_dir=$OUTPUT_DIR \
    use_wandb=True