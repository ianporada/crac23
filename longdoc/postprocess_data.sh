
OUT_DIR=/network/scratch/p/poradaia/crac23/longdoc/models/crac/xlmroberta_large/coref_crac_eba9a9193140a40d490e4a6b9ccf0f49_2/crac/ # original
OUT_DIR=/network/scratch/p/poradaia/crac23/longdoc/models/crac/xlmroberta_large/coref_crac_6b26858dd0958b4d21880b6703dda415_2/crac/ # best


OUT_DIR=/network/scratch/p/poradaia/best/crac/
LOCAL_OUT_DIR=/Users/ianporada/Documents/data/crac23/output/raw

mkdir -p $LOCAL_OUT_DIR
rsync -v -r -a mila:$OUT_DIR/* $LOCAL_OUT_DIR

###

python preprocessing/longdoc_to_conllu.py

###

cd ~/Documents/research/tools/
for fname in ~/Documents/data/crac23/output/test-est/*.conllu
do
    basefname=$(basename $fname)
    lang="${basefname:0:2}"
    echo $basefname
    python3 validate.py --level 2 --coref --lang $lang $fname
done

##

python3 validate.py --level 2 --coref --lang pl ~/Documents/data/crac23/output/test-est/pl_pcc-corefud-test.conllu

udapy -s read.Conllu split_docs=1 corefud.MergeSameSpan corefud.IndexClusters < ~/Documents/data/crac23/output/dev-est/es_ancora-corefud-dev.conllu   > ~/Documents/data/crac23/output/dev-fixed/es_ancora-corefud-dev.conllu 

##

cd ~/Documents/data/crac23/output/test-est
zip -r test_output.zip *.conllu
