
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

#### large

OUT_DIR=/network/scratch/p/poradaia/crac23/longdoc/models/crac/xlmroberta_large/coref_crac_6b26858dd0958b4d21880b6703dda415_2/best/*
cp -r $OUT_DIR ~/scratch/best/

cd ~/fast-coref/src
python -m main experiment=crac_xlm_large \
    paths.base_data_dir=$LONGDOC_DATA_DIR \
    paths.model_dir=/home/mila/p/poradaia/scratch/best \
    train=False