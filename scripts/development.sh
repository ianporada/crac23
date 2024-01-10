### Setup Env

module purge

module load anaconda/3
conda create --name crac23 --clone babylm_env
conda activate crac23

pip install udapi
pip install nltk

### Run

# not enough mem: salloc --gres=gpu:a100l:1 -c 6 --mem=32G -t 1:00:00 --partition=unkillable

salloc --gres=gpu:a100l:1 -c 24 --mem=64G -t 1:00:00 --partition=short-unkillable

###

module load anaconda/3
conda activate crac23

# set cache directory
mkdir -p $SCRATCH/cache
export TRANSFORMERS_CACHE="${SCRATCH}/cache/transformers_cache"
export HF_DATASETS_CACHE="${SCRATCH}/cache/datasets_cache"

