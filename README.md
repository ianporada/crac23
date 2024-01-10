# crac23
CRAC 2023 Shared Task on Multilingual Coreference Resolution

## Details

* Shared task: https://ufal.mff.cuni.cz/corefud/crac23
* Baseline system: https://github.com/ondfa/coref-multiling

## Data
Train/Dev:
```bash
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5053{/CorefUD-1.1-public.zip}
```

## Eval
Scorer: https://github.com/ufal/corefud-scorer
```bash
python corefud-scorer.py KEY_FILE RESPONSE_FILE
```

### Notes:

* "the primary ranking score will be computed by macro-averaging CoNLL F1 scores over all datasets"
* "If the submitted system is not able to predict the mention heads (i.e. it predicts mention spans only, and the head index is always 1), mention heads can be estimated using the provided dependency tree and heuristics, e.g. the ones provided by Udapi (see below), using the following command: `udapy -s corefud.MoveHead < in.conllu > out.conllu`"

### Validation:

```bash
git clone git@github.com:UniversalDependencies/tools.git
cd tools
python3 validate.py -h
```

```bash
python3 validate.py --level 2 --coref --lang cs cs_pdt-corefud-test.conllu
```
Lang is code for each dataset

Post-processing:
```bash
udapy -s read.Conllu split_docs=1 corefud.MergeSameSpan corefud.IndexClusters < orig.conllu > fixed.conllu
```