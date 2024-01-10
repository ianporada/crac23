## training preco

salloc --gres=gpu:a100l:1 -c 24 --mem=64G -t 1:00:00 --partition=short-unkillable
salloc --gres=gpu:a100l:1 -c 6 --mem=32G -t 12:00:00 --partition=unkillable

##

rsync -v -r -a --delete ~/Documents/research/fast-coref2/src mila:~/fast-coref2/

##

module load anaconda/3
conda activate crac23

export TRANSFORMERS_CACHE="${SCRATCH}/cache/transformers_cache"
export HF_DATASETS_CACHE="${SCRATCH}/cache/datasets_cache"

PROJECT_DIR=$SCRATCH/crac23

LONGDOC_DIR=$PROJECT_DIR/longdoc
LONGDOC_DATA_DIR=$LONGDOC_DIR/data

####

cd ~/fast-coref2/src
python -m main experiment=debug \
    paths.base_data_dir=$LONGDOC_DATA_DIR \
    paths.base_model_dir=/home/mila/p/poradaia/scratch/debug

#### train dataset

OUTPUT_DIR=$LONGDOC_DIR/models/crac/mt5_xl

export WANDB_DIR=$OUTPUT_DIR/wandb

mkdir -p $OUTPUT_DIR
mkdir -p $WANDB_DIR

cd ~/fast-coref2/src
python -m main experiment=crac_mt5_xl \
    paths.base_data_dir=$LONGDOC_DATA_DIR \
    paths.base_model_dir=$OUTPUT_DIR \
    use_wandb=True

####