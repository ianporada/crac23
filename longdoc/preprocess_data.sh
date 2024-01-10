## sync code to cluster

rsync -v -r -a --delete ~/Documents/current_projects/fast-coref/src mila:~/fast-coref/
rsync -v -r -a --delete ~/Documents/current_projects/coref_resources mila:~/

## sync data to cluster
mkdir -p ~/scratch/crac23/data/input/jsonl/

rsync -v -r -a --delete ~/Documents/data/crac23/input/jsonl/*.jsonl mila:~/scratch/crac23/data/input/jsonl/

## preprocessing

salloc --partition=unkillable-cpu -c 2 --mem=16G -t 2:00:00

module load anaconda/3
conda activate $SCRATCH/envs/longformer

export TRANSFORMERS_CACHE="${SCRATCH}/cache/transformers_cache"
export HF_DATASETS_CACHE="${SCRATCH}/cache/datasets_cache"

PROJECT_DIR=$SCRATCH/crac23
RAW_DATA_DIR=$PROJECT_DIR/data/input/jsonl
LONGDOC_DIR=$PROJECT_DIR/longdoc
LONGDOC_DATA_DIR=$LONGDOC_DIR/data

#### crac

INPUT_DIR=$RAW_DATA_DIR
OUTPUT_DIR=$LONGDOC_DATA_DIR/crac/longformer

mkdir -p $OUTPUT_DIR

cd ~/fast-coref/src
python -m data_processing.process_crac $INPUT_DIR \
    -output_dir $OUTPUT_DIR \
    -add_speaker

### xlmroberta

INPUT_DIR=$RAW_DATA_DIR
OUTPUT_DIR=$LONGDOC_DATA_DIR/crac/xlmroberta

mkdir -p $OUTPUT_DIR

cd ~/fast-coref/src
python -m data_processing.process_crac $INPUT_DIR \
    -output_dir $OUTPUT_DIR \
    -add_speaker \
    -model xlmroberta \
    -seg_len 512

### mt5 
#  conda install -y sentencepiece --override-channels -c conda-forge

INPUT_DIR=$RAW_DATA_DIR
OUTPUT_DIR=$LONGDOC_DATA_DIR/crac/mt5

mkdir -p $OUTPUT_DIR

cd ~/fast-coref/src
python -m data_processing.process_crac $INPUT_DIR \
    -output_dir $OUTPUT_DIR \
    -add_speaker \
    -model mt5 \
    -seg_len 1024

# train / dev / test = 9595 / 1325 / 1113
