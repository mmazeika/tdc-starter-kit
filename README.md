# Trojan Detection Competition Starter Kit

This repository contains code for loading data and generating submissions for the Trojan Detection Challenge (TDC) NeurIPS 2022 competition. To learn more, please see the [competition website](https://www.trojandetection.ai/).

## Contents

There are four folders corresponding to different tracks and subtracks: 1) Trojan Detection, 2) Trojan Analysis (Target Label Prediction), 3) Trojan Analysis (Trigger Synthesis), and 4) Evasive Trojans. We provide starter code for submitting baselines in `example_submission.ipynb` under each folder. The `tdc_datasets` folder is expected to be under the same parent directory as `tdc-starter-kit`. The datasets are available [here](https://www.example.com/) (link will be made live on 7/15).

The `utils.py` file contains helper functions for loading new models, generating new attack specifications, and training clean/Trojaned networks. Currently, this is primarily used for the Evasive Trojans Track starter kit. The `wrn.py` file contains the definition of the Wide Residual Network class used for CIFAR-10 and CIFAR-100 models. When loading these models, `wrn.py` must be in your path. See the example submission notebooks for details.