# NILMFormer

<div align="center">

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2502.15637&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2506.05880)
[![Python](https://img.shields.io/badge/Python-3.9|3.10|3.11|3.12-blue)]()

</div>

Official PyTorch codebase for **NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-Stationarity** (ACM SIGKDD 2025).

📄 **Arxiv preprint**: [NILMFormer](https://arxiv.org/abs/2506.05880).

<p align="center">
    <img width="300" src="https://github.com/adrienpetralia/NILMFormer/blob/main/assets/intro.png" alt="Intro">
</p>

Millions of smart meters have been deployed worldwide, collecting the power consumed by individual households. Based on these measurements, electricity suppliers provide feedback on consumption behaviors. To help customers better understand their usage, suppliers need to provide **detailed** (per-appliance) feedback—a challenging problem known as **Non-Intrusive Load Monitoring (NILM)**.

NILM aims to disaggregate a household’s total power consumption and retrieve the individual power usage of different appliances. Current state-of-the-art (SotA) solutions rely on deep learning and process household consumption in subsequences. However, real-world smart meter data are **non-stationary**—distribution drifts within each window segment can severely impact model performance.

We introduce **NILMFormer**, a sequence-to-sequence Transformer-based architecture designed to tackle this problem.

<p align="center">
    <img width="700" src="https://github.com/adrienpetralia/NILMFormer/blob/main/assets/results_sample.png" alt="Results sample">
</p>



## Outline 📝

This repository contains the **source code** of NILMFormer, as well as the code needed to reproduce the experimental evaluation from our paper.  
It also includes **10 recent SotA NILM baselines** re-implemented in PyTorch.

---

### Getting Started 🚀

To install the dependencies, you can use the following commands. Life is much easier thanks to [uv](https://astral.sh/blog/uv)!

```bash
pip install uv
git clone https://github.com/adrienpetralia/NILMFormer
cd NILMFormer
uv sync
```


### Code Structure 📁

```
.
├── assets                 # assets for the README file 
├── configs                # configs folder (i.e., '.yaml' files)
├── data                   # data info folder
├── results                # detailed experiment results folder
├── scripts                # scripts to launch experiments
│   ├── run_one_expe.py    #   python script to launch one experiment
│   └── run_all_expe.sh    #   bash script to launch all experiments
├── src                    # source package
│   ├── helpers            #   helper functions (processing, training loops, metrics, ...)
│   ├── baselines          #   nilm and tser baselines
│   └── nilmformer         #   nilmformer model
├── pyproject.toml         # project setup file
└── uv.lock                # lock to resolve dependencies
```

### Launch an Experiment ⚙️

To run a **specific** experiment, use the command below:
```
uv run -m scripts.run_one_expe \
    --dataset "UKDALE" \
    --sampling_rate "1min" \
    --appliance "WashingMachine" \
    --window_size 128 \
    --name_model NILMFormer \
    --seed 0
```

To run **all** experiments conducted in our paper (this may take some time), use:
```
. scripts/run_all_expe.sh
```

## NILMFormer ⚡

**TL;DR** : **NILMFormer** is a **sequence-to-sequence Transformer-based architecture** purpose-built for **Non-Intrusive Load Monitoring (NILM)**. It tackles the **non-stationary** nature of smart meter data by splitting and separately encoding the **shape**, **temporal** dynamics, and **intrinsic statistics** of each subsequence. These components are then fused within the Transformer block. Finally, the prediction is refined through a **linear transformation** of the input series statistics, accounting for **power loss** in the disaggregation process.

### Architecture Details 🔎

<p align="center">
    <img width="600" src="https://github.com/adrienpetralia/NILMFormer/blob/main/assets/nilmformer_details.png" alt="NILMFormer">
</p>

**Mechanisms for Handling Non-Stationarity:** To handle the non-stationarity aspect of electricity consumption data, NILMFormer operates by first stationnarizing the input subsequence by subtracting its mean and standard deviation.
While the normalized subsequence is passed through a robust convolutional block that serves as a features extractor, the removed statistics are linearly projected in a higher space (referred to as *TokenStats*), and the timestamps are used by the proposed TimeRPE module to compute a positional encoding matrix.
These features are concatenated and fed into the Transformer block, followed by a simple Head to obtain a 1D sequence of values.
The final step consists of linearly projecting back the *TokenStats* (referred to as *ProjStats*) to 2 scalar values that are then used to denormalize the output, providing the final individual appliance consumption.

**Timestamps-Related Positional Encoding (TimeRPE):** TimeRPE leverages discrete timestamps (minutes, hours, days, months) extracted from each input subsequence. Each timestamp is transformed through a sinusoidal function, capturing periodic behaviors. 
These signals are then projected into a higher-dimensional space via a 1D convolution (kernel size = 1). 
This approach provides a more **time-aware** embedding than standard positional encoding, helping the model better handle real-world temporal patterns. 


## Contributors 👨‍👨‍👦‍👦

* Adrien Petralia (Université Paris Cité, EDF Research)
* Philippe Charpentier (EDF Research)
* Youssef Kadhi (EDF Research)
* Themis Palpanas (IUF, Université Paris Cité) 

## Citing NILMFormer 📚

If you use NILMFormer in your work, please cite our paper:

```bibtex
@inproceedings{petralia2025nilmformer,
author = {Petralia, Adrien and Charpentier, Philippe and Kadhi, Youssef and Palpanas, Themis},
title = {NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-Stationarity},
year = {2025},
publisher = {Association for Computing Machinery},
doi = {10.1145/3711896.3737251},
booktitle = {Proceedings of the 31th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining V.2},
location = {Toronto, ON, Canada},
series = {KDD '25}
}
```

## Acknowledgments 🎅

Work supported by EDF R&D and ANRT French program.
