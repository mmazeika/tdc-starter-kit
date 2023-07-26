# Trojan Detection Challenge Starter Kit

**Update:** The validation phase annotations are available [here](https://drive.google.com/drive/folders/1362q69BGJktPYKOXRt0fmp4eIlMi8-Ek?usp=share_link).

This repository contains code for loading data and generating submissions for the Trojan Detection Challenge (TDC) NeurIPS 2022 competition. To learn more, please see the [competition website](https://trojandetection.ai/).

## Contents

There are four folders corresponding to different tracks and subtracks: 1) Trojan Detection, 2) Trojan Analysis (Target Label Prediction), 3) Trojan Analysis (Trigger Synthesis), and 4) Evasive Trojans. We provide starter code for submitting baselines in `example_submission.ipynb` under each folder. The `tdc_datasets` folder is expected to be under the same parent directory as `tdc-starter-kit`. The datasets are available [here](https://zenodo.org/record/6894041). You can download them from the Zenodo website or by running `download_datasets.py`.

The `utils.py` file contains helper functions for loading new models, generating new attack specifications, and training clean/Trojaned networks. This is primarily used for the Evasive Trojans Track starter kit. It also contains the load_data function for loading data sources (CIFAR-10/100, GTSRB, MNIST), which may be of general use. To load GTSRB images, unzip `gtsrb_preprocessed.zip` in the data folder (NOTE: This folder is only for storing data sources. The network datasets are stored in tdc_datasets, which must be downloaded from Zenodo). You may need to adjust the paths in the load_data function depending on your working directory. The `wrn.py` file contains the definition of the Wide Residual Network class used for CIFAR-10 and CIFAR-100 models. When loading networks from the competition datasets, `wrn.py` must be in your path. See the example submission notebooks for details.

## How to Use

Clone this repository, then download the competition datasets and unzip adjacent to the repository. Ensure that your Jupyter version is up-to-date (fairly recent). To avoid errors with model incompatibility, please use PyTorch version 1.11.0. You can now run one of the example notebooks or start building your own submission.
