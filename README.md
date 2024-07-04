# ProfileML

This repository contains the code for the talk "Profiling GPU-driven machine learning code" presented at the [Best Practices in AI Afternoon](https://rse.shef.ac.uk/events/seminar-2024-07-05-best-practices-in-ai-afternoon.html) hosted by the Research Software Engineering group and Centre of Machine Intelligence (CMI) at the University of Sheffield.

Links to the slides can be found [here](https://docs.google.com/presentation/d/1Vjus8Bshb1jdZ1W9WcgNMiiIESk8SgCAmI7PJGpBUUg/edit?usp=sharing).

## Installation

I recommend using a virtual environment to install the dependencies. Requirements can be installed via pip:

```pip install -r requirements.txt```

Note: I've frozen the  versions of the packages to ensure reproducibility. There is no reason to my knowledge that the code won't work with newer versions of the packages.

## Data

The data used in this repository is from the [TGS Salt Identification Competition](https://www.kaggle.com/c/tgs-salt-identification-challenge/data) from Kaggle. The data is not included in this repository, but can be downloaded from the link. Simply unzip it into the top level of this repository and no further changes should be needed to the code.

## Running the Code

Simple run the following command to train the model:

```python train.py``` or ```python train-ddp.py``` for single gpu and distributed training respectively.

Note: This code is not optimized for training but rather a toy example to demonstrate profiling. Beware when copying for your own use.

## Profiling

Logs will be created in a `logs` directory. To view the logs, run the following command:

```tensorboard --logdir logs```

This will start a tensorboard server that you can view in your browser. The default port is 6006.

Alternatively, you can get a trace viewer by dragging the logs into a browser window at `chrome://tracing` or `edge://tracing`.

