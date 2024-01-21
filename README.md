# crac23
CRAC 2023 Shared Task on Multilingual Coreference Resolution

Code for the McGill shared task submission. The code is currently undocumented, but please create an issue in this repo if you would like any clarifications.

See more details in the paper description at:
```
@inproceedings{porada-cheung-2023-mcgill,
    title = "{M}c{G}ill at {CRAC} 2023: Multilingual Generalization of Entity-Ranking Coreference Resolution Models",
    author = "Porada, Ian  and
      Cheung, Jackie Chi Kit",
    editor = "{\v{Z}}abokrtsk{\'y}, Zden{\v{e}}k  and
      Ogrodniczuk, Maciej",
    booktitle = "Proceedings of the CRAC 2023 Shared Task on Multilingual Coreference Resolution",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.crac-sharedtask.5",
    doi = "10.18653/v1/2023.crac-sharedtask.5",
    pages = "52--57",
    abstract = "Our submission to the CRAC 2023 shared task, described herein, is an adapted entity-ranking model jointly trained on all 17 datasets spanning 12 languages. Our model outperforms the shared task baselines by a difference in F1 score of +8.47, achieving an ultimate F1 score of 65.43 and fourth place in the shared task. We explore design decisions related to data preprocessing, the pretrained encoder, and data mixing.",
}
```

## Details

* Shared task: https://ufal.mff.cuni.cz/corefud/crac23
* Baseline system: https://github.com/ondfa/coref-multiling

## Model

Model code is based on https://github.com/shtoshni/fast-coref

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
