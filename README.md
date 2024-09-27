# CDER - Collaborative Evidence Retrieval for DocRE
The source code for the ACIIDS'24 paper "[CDER: Collaborative Evidence Retrieval for Document-level Relation Extraction](https://doi.org/10.1007/978-981-97-4982-9_3)"
## Requirements
To install requirements:
```python
pip install requirements.txt
```
## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset used in our model can be downloaded by following the instructions at this [link](https://drive.google.com/drive/folders/1owp7ZRbrMl_s1ljIh6AvnmniLJSliV6h?usp=sharing). 

You can find the current version of DocRED dataset and the `meta` folder from [their public repo](https://github.com/thunlp/DocRED).

Please note that the `dev.json` file available in the current DocRED was modified in August 2021, and the updated version contains 998 documents. We use the original version of `dev.json`, which includes 1000 documents.

The expected structure of files is:
```
|-- CDER
  |-- dataset
    |-- docred
      |-- dev.json
      |-- test.json
      |-- train_annotated.json
      |-- train_distant.json
      |-- meta
        |-- rel2id.json
        |-- rel_info.json
```
## Training and Evaluation
Train CDER on DocRED with the following command:
```bash
>> sh scripts/train.sh
```
After training, testing CDER with the following command:
```bash
>> sh scripts/test.sh
```
Inferring CDER result for integrating to DocRE with the following command:
```bash
>> sh scripts/infer.sh
```
## DocRE result
We utilize the [GitHub repository](https://github.com/youmima/dreeam) for the DREEAM model to integrate the extracted evidence results from CDER. To reproduce the results, simply replace DREEAM's evidence output with the evidence results from CDER.
## Citation
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
## Reference
```
[1] Wenxuan Zhou, Kevin Huang, Tengyu Ma, and Jing Huang. 2021. Document-level relation extraction with adaptive thresholding and localized context pool- ing. In Proceedings of the AAAI Conference on Arti- ficial Intelligence.
[2] Yiqing Xie, Jiaming Shen, Sha Li, Yuning Mao, and Jiawei Han. 2022. Eider: Empowering Document-level Relation Extraction with Efficient Evidence Extraction and Inference-stage Fusion. In Findings of the Association for Computational Linguistics: ACL 2022, pages 257–268, Dublin, Ireland. Association for Computational Linguistics.
[3] Youmi Ma, An Wang, and Naoaki Okazaki. 2023. DREEAM: Guiding Attention with Evidence for Improving Document-Level Relation Extraction. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1971–1983, Dubrovnik, Croatia. Association for Computational Linguistics.
```
