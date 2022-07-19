# Trojan Detection Competition Starter Kit

This repository contains code for loading data and generating submissions for the Trojan Detection Challenge (TDC) NeurIPS 2022 competition. To learn more, please see the [competition website](https://trojandetection.ai/).

## Contents

There are four folders corresponding to different tracks and subtracks: 1) Trojan Detection, 2) Trojan Analysis (Target Label Prediction), 3) Trojan Analysis (Trigger Synthesis), and 4) Evasive Trojans. We provide starter code for submitting baselines in `example_submission.ipynb` under each folder. The `tdc_datasets` folder is expected to be under the same parent directory as `tdc-starter-kit`. The datasets are available [here](https://zenodo.org/record/6812318). You can download them from the Zenodo website or by running `download_datasets.py`.

The `utils.py` file contains helper functions for loading new models, generating new attack specifications, and training clean/Trojaned networks. This is primarily used for the Evasive Trojans Track starter kit. It also contains a function for loading data sources (CIFAR-10/100, GTSRB, MNIST), which may be of general use. To load GTSRB images, unzip `gtsrb_preprocessed.zip` in the data folder (NOTE: This folder is only for storing data sources. The network datasets are stored in tdc_datasets, which must be downloaded from Zenodo). The `wrn.py` file contains the definition of the Wide Residual Network class used for CIFAR-10 and CIFAR-100 models. When loading these models, `wrn.py` must be in your path. See the example submission notebooks for details.

## How to Use

Clone this repository, then download the competition datasets and unzip adjacent to the repository. Ensure that your PyTorch and Jupyter versions are up-to-date (fairly recent). You can now run one of the example notebooks or start building your own submission.
