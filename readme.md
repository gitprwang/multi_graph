## Structure:

* data: including PEMSD4 and PEMSD8 dataset used in our experiments, which are released by and available at  [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data).

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our model

The code for training and loading data are forked from [AGCRN](https://github.com/LeiBAI/AGCRN).

## Requirements
```
python=3.6
pytorch=1.9.0
numpy=1.19
argparse
configparser
```

## Training
To train the model, you can simply run the following code in terminal:

```
cd model
python Run.py
```

We provide pretrained model, you can run the following code to test:
```
python Run.py --model test
```
