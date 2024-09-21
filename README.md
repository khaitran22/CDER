# CDER - Collaborative Evidence Retrieval for DocRE
The source code for the ACIIDS'24 paper "[CDER: Collaborative Evidence Retrieval for Document-level Relation Extraction](https://doi.org/10.1007/978-981-97-4982-9_3)"

If you make use of this code in your work, please kindly cite the following paper:
```bibtex
@inproceedings{tran2024cder,
  title={CDER: Collaborative Evidence Retrieval for Document-Level Relation Extraction},
  author={Tran, Khai Phan and Li, Xue},
  booktitle={Asian Conference on Intelligent Information and Database Systems},
  pages={28--39},
  year={2024},
  organization={Springer}
}
```
## Requirements
To install requirements:
```python
pip install requirements.txt
```
## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data).
```
|-- docred
  |-- dev.json
  |-- meta
    |-- rel2id.json
    |-- rel_info.json
  |-- test.json
  |-- train_annotated.json
  |-- train_distant.json
```
