This repository contains the official code from __Event Transformer. A sparse-aware solution for efficient event data processing__. 
Arxiv: ...
Citation:
```
{}
```

### REPOSITORY REQUIREMENTS
-----------------------

The present work has been developed and tested with Python 3.7.10, pytorch 1.9.0 and Ubuntu 18.04
To reproduce our results we suggest to create a Python environment as follows.

```
conda create --name evt python=3.7.10
conda activate evt
pip install -r requirements.txt
```



### PRETRAINED MODELS

The pretrained models must be located under a ./pretrained_models directory and can be downloaded from Drive (
[DVS128 10 classes](url), 
[DVS128 11 classes](url), 
[Sl-Animals 3-Sets](url), 
[Sl-Animals 4-Sets](url), 
[ASL](url)).



### DATA DOWNLOAD AND PRE-PROCESSING

The datasets involved in the present work must be downloaded from their source and stored under a './datasets' path:
 - DVS128: https://research.ibm.com/interactive/dvsgesture/
 - SL-Animals-DVS: http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database
 - ASL: https://github.com/PIX2NVS/NVS2Graph
 - N-Caltech-101: https://www.garrickorchard.com/datasets/n-caltech101

In order to have a faster training process we pre-process the source data by building intermediate sparse frame representations, that will be later loaded by our data generator.
This transformation can be perfomed with the files located under './dataset_scripts'.
In the case of DVS128, it is mandatory to execute first 'dvs128_split_dataset.py' and later 'dvs128.py'.



### EvT EVALUATION

The evaluation of our pretrained models can be performed by executing: `python evaluation_stats.py`
At the beginning of the file you can select the pretrained model to evaluate and the device where to evaluate it (CPU or GPU). Evaluation results include FLOPs, parameters, average activated patches, average processing time, and validation accuracy.



### EvT TRAINING

The training of a new model can be performed by executing: `python train.py`
At the beginning of the file you can select the pretraining model from where to copy its training hyper-parameters.
Note that, since the involved datasets do not contain many training samples and there is data augmentation involed in the training, final results might not be exactly equal than the ones reported in the article. If so, please perform several training executions.
