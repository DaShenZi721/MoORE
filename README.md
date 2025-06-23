<div align=center>

# MoORE: SVD-based Model MoE-ization for Conflict- and Oblivion-Resistant Multi-Task Adaptation

[![arXiv](https://img.shields.io/badge/arXiv-2502.14637-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2506.14436)

</div>

<div align="center">
  <img src="assets/moore_scheme.png" width="1100"/>
</div>

## Introduction

This repository includes the official implementation of [MoORE](https://arxiv.org/abs/2506.14436). 
We propose a simple yet effective multi-task adaptation method, called Mixture of Orthogonal Rank-one Experts (MoORE).


## Environment Setup

Please refer to [MoE-PEFT Install Guide](https://github.com/TUDB-Labs/MoE-PEFT/blob/main/Install.md).

## Models
we utilize [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) as the base model and adapt it by various multi-task adaptation methods.

## Training and Evaluation

You can train and evaluate on the CSR-MTL dataset by running the code below.  
```bash
bash train.sh
```
If you want to run experiments on the NLU-MTL dataset, you need to replace the "task_name" field in [moe_peft_moore.json](moe_peft_moore.json) with "glue:cola;glue:mnli;glue:mrpc;glue:qnli;glue:qqp;glue:rte;glue:sst2".

## Acknowledgement

The repo is based on the [MoE-PEFT](https://github.com/TUDB-Labs/MoE-PEFT), we greatly appreciate the authors for their contributions.

## 📌 Citing our work
If you find our work useful, please cite it:
```bibtex
@misc{yuan2025mooresvdbasedmodelmoeization,
      title={MoORE: SVD-based Model MoE-ization for Conflict- and Oblivion-Resistant Multi-Task Adaptation}, 
      author={Shen Yuan and Yin Zheng and Taifeng Wang and Binbin Liu and Hongteng Xu},
      year={2025},
      eprint={2506.14436},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.14436}, 
}
```