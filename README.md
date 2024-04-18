# MetaCRL

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![Static Badge](https://img.shields.io/badge/Meta_Learning-Task_Representation-blue)
![Static Badge](https://img.shields.io/badge/Hacking_Task_Confounder-green)
![Static Badge](https://img.shields.io/badge/to_be_continue-orange)
![Stars](https://img.shields.io/github/stars/WangJingyao07/MetaCRL)

🌈 The PyTorch implementation of MetaCRL, described in IJCAI 2024 "Hacking Task Confounder in Meta-Learning".

> Version 1.0: Run example `miniImagenet` Datasets are supported.

> Version 2.0: Re-write meta learner and basic learner. Solved some serious bugs in version 1.0.

> Version 3.0: Implement multiple model basic units in `models` that can be replaced as needed.


This method can be plug-and-played into any meta-learning framework:

For more frameworks and datasets, please visit [HERE](https://github.com/WangJingyao07/MetaLearning-Lab) with [WEBSITE](https://wangjingyao07.github.io/Awesome-Meta-Learning-Platform/)

For more sampling strategies and settings, please visit [HERE](https://github.com/WangJingyao07/Adaptive-Sampler)

## Introduction

Meta-learning enables rapid generalization to new tasks by learning knowledge from various tasks. It is intuitively assumed that as the training progresses, a model will acquire richer knowledge, leading to better generalization performance. However, our experiments reveal an unexpected result: there is negative knowledge transfer between tasks, affecting generalization performance. To explain this phenomenon, we conduct Structural Causal Models (SCMs) for causal analysis. Our investigation uncovers the presence of spurious correlations between task-specific causal factors and labels in meta-learning. Furthermore, the confounding factors differ across different batches. We refer to these confounding factors as ``Task Confounders". Based on these findings, we propose a plug-and-play Meta-learning Causal Representation Learner (MetaCRL) to eliminate task confounders. It encodes decoupled generating factors from multiple tasks and utilizes an invariant-based bi-level optimization mechanism to ensure their causality for meta-learning.

Brief overview of the meta-learning process with MetaCRL:

![OVERVIEW](https://github.com/WangJingyao07/MetaCRL/assets/45681444/dcdfb8bd-13ab-4622-b2c0-9c12d4c14602)


## Platform

- python: 3.x
  
- Pytorch: 0.4+

## Create Environment

For easier use and to avoid any conflicts with existing Python setup, it is recommended to use [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/) to work in a virtual environment. Now, let's start:

**Step 1:** Install [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/)

```bash
pip install --upgrade virtualenv
```

or using `conda create`.

**Step 2:** Create a virtual environment, activate it:

```bash
virtualenv venv
source venv/bin/activate
```

**Step 3:** Install the requirements in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```


## Data Availability

For 5-way 1-shot exp., it allocates nearly 6GB GPU memory.

1. download `MiniImagenet` dataset from [here](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4), the splitted file: `train/val/test.csv` are provided in `data/split`
   
2. for image split, extract it like:

```shell
miniimagenet/
├── images
	├── n0210891500001298.jpg  
	├── n0287152500001298.jpg 
	...
├── test.csv
├── val.csv
└── train.csv

```

`data/data_generator` provides the python file for data generator.

3. modify the `path` in `example.py`:

```python
        mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                    k_query=args.k_qry,
                    batchsz=10000, resize=args.imgsz)
		...
        mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                    k_query=args.k_qry,
                    batchsz=100, resize=args.imgsz)
```

to your actual data path.

## Run

We provide the example for training on miniImagenet:

```
python example.py
```

more examples will be provided after the paper being published.

### Cite

If you find our work and codes useful, please consider citing our paper and star our repository (🥰🎉Thanks!!!):

```
@misc{wang2024hacking,
      title={Hacking Task Confounder in Meta-Learning}, 
      author={Jingyao Wang and Yi Ren and Zeen Song and Jianqi Zhang and Changwen Zheng and Wenwen Qiang},
      year={2024},
      eprint={2312.05771},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

(arXiv version, the final version will be updated after the paper is published.)
