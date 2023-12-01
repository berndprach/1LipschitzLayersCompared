# 1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness
This repository contains code for the paper [1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness](https://arxiv.org/abs/2311.16833),
where we compare different methods of generating 1-Lipschitz layers. A figure summarizing our findings is below.
Higher values means better performance.

<img src="https://github.com/berndprach/1LipschitzLayersCompared/blob/main/data/radar_plot.png" alt="Radar plot of results" width="800"/>

In this repository we provide the code for both the random hyperparameter search, as well as training the final models.
We also provide code to estimate the memory and time requirements for each method.

## Requirements
- torch==1.12
- python==3.9


## Reproducing the results
The results can be replicated by followng the steps below.

### Epoch estimation
The following script create a yml file with the number of epochs to train for each model and each dataset <DATASET>. The file is saved in the directory `data/settings/<DATASET>`. An example of usage is the following:

```[bash]
$ python3 epoch_budget_estimator.py --dataset CIFAR10
```

Other arguments are available (e.g. training time, update exisiting measurment for a specific method). See the help of the script for more information.

### Random Search

In order to run a random search, you need to create a settings file for the specific model and the specific method. This can be done manually or by using the script ```make_tree.py``` that automatically generates the settings file for a random search. The setting files will be saved in directory tree that will be created by the script. The directory tree will be created in the directory specified by the argument `root_dir`.
```[bash]
$ python3 make_tree.py --root_dir /data/runs/CIFAR10/my_random_search \
    --default settings/default_CIFAR10.yml \
    --mode random_search \
    --training_time  2 
```

 The directory tree will have the following structure (only two layers are shown for simplicity):

```[bash]
CIFAR10
├── my_random_search
│   ├── ConvNetL
│   │   ├── AOLConv2d
│   │   │   └── settings.yml
│   │   └── BCOP
│   │       └── settings.yml
│   ├── ConvNetM
│   │   ├── AOLConv2d
│   │   │   └── settings.yml
│   │   └── BCOP
│   │       └── settings.yml
│   ├── ConvNetS
│   │   ├── AOLConv2d
│   │   │   └── settings.yml
│   │   └── BCOP
│   │       └── settings.yml
│   └── ConvNetXS
│       ├── AOLConv2d
│       │   └── settings.yml
│       └── BCOP
│           └── settings.yml
└── my_final_training...
```

where the argument `default` allows using a default settings file for the fixed options (e.g. the dataset, the optimizer, the loss function, etc.).


The settings file that we used for our implementations are available in the directory `settings`.

Once the settings file and the directory tree are created, you can run the random search by runnign the training script for how many times you want. The training script has the following arguments:

```[bash]
$ python3 ./train.py /data/runs/CIFAR10/my_random_search/modelsize/mymethod \
    --device cuda \
    --jobid test  \
    --save-memory
```

To evaluate the best `learning_rate` and the best `weight_decay` for a specific model, you can use the script `eval_best_lr_wd.py` by considering the following command:

```[bash]
$ python3 eval_best_lr_wd.py --root_dir /data/runs/my_random_search \
    --output-file /data/settings/CIFAR10/best_lr_wd.csv
```

### Training the models with the best hyperparameters
Once the best hyperparameters are found, you can train the models with the best hyperparameters by creating the setting files with the proper hyperparameters and by running the training script. The training script has the following

The settings file can be created manually or by using the script `make_tree.py` by considering the following command:

```[bash]
$ python3 make_tree.py --root_dir /data/runs/CIFAR10/my_final_runs \
    --default settings/default_CIFAR10.yml \
    --mode final_training \
    --training_time  2 \
    --best_lr_wd /data/settings/CIFAR10/best_lr_wd.csv
```

where the argument `best_lr_wd` is the path to the csv file containing the best hyperparameters for each model.

Once the settings file is created, you can run the training script by considering the following command:

```[bash]
$ python3 ./train.py /data/runs/my_final_runs/modelsize/mymethod \
    --device cuda \
    --jobid test
```

## Evaluate a custom model or layer
Follow the steps below to train a custom model or a custom layer. 

### Create a model
Write your own model of the models in the `models` directory. The model must be a subclass of `torch.nn.Module` or a callable that returns a `torch.nn.Module` instance. The model must be defined in the python module `models` contained in the directory `models`. Remember to import the model in the `models/__init__.py` file.

### Create a lipschitz layer
Write your own layer of the layers in the `models/layers/lipschitz` directory. The layer must be a subclass of `torch.nn.Module` or a callable that returns a `torch.nn.Module` instance. The layer should be properly importend in the `models/layers/__init__.py` file.

## Manually train a model using the train script
The train scripts leverages a setting file to load all the requirements for the training (e.g. the model, the dataset, the optimizer, the loss function, etc.)
### Create a setting file
An example of a settings file is given in `data/runs/example/settings.yml`.


Create a directory in the direcotry `data/runs` with a name of your choice (e.g. `my_experiment`) and create a settings file in this directory. The settings file must be a yaml file and must be named `settings.yml`.

**Usage of tags**: The settings file uses tags to help the setting parser in initializing the model, the dataset, the optimizer and the loss function.

The tags can be deterministic:
- `!model`
- `!layer`
- `!dataset`
- `!optimizer`
- `!lr_scheduler`
- `!metric`

and not deterministic
- `!choice`, choose a random element from a list of elements.
- `!randint`, choose a random integer between two integers.
- `!randfloat`, choose a random float between two floats. 

The settings file contains the following information:
- `model` followed by the tag `!model`, which helps the setting parses in intializing the model.
    - `name`: The name of the model to train. The model must be defined in the python module `models`.
    - `kwargs`: The keyword arguments to pass to the model constructor.
- `trainset` followed by the tag `!dataset`, which helps the setting parses in intializing the dataset.
    - `name`: The name of the dataset to use. The dataset must be defined in the python module `datasets`.
    - `<kwargs>`: The keyword arguments to pass to the dataset constructor.
- `batch_size`: The batch size to use for training.
- `epochs`: The number of epochs to train.
- `optimizer` followed by the tag `!optimizer`, which helps the setting parses in intializing the optimizer.
    - `name`: The name of the optimizer to use. The optimizer must be defined in the python module `optimizers`.
    - `<kwargs>`: The keyword arguments to pass to the optimizer constructor.
- `loss` followed by the tag `!metric`, which helps the setting parses in intializing the loss function.
    - `name`: The name of the loss function to use. The loss function must be defined in the python module `metrics`.
- `eval_metrics`: A metric or a list of metrics to evaluate the model on. Each metric is followed by the tag `!metric`.

### Train the model
Run the training script with the following command: 
```[bash]
    .\train.py data\runs\my_experiment --device cuda:0
```
to train the model.


## Measuring Time and Memory for models:
In order to measure batch time and memory usage for a model, run e.g.
```[bash]
    python run.py --job-id=0 --results-file-name=memory.csv --dataset=cifar10 --task=measure-memory
```
and find the results in `data/evaluations/memory.csv`.
Here, each combination of model size and method has a specific job-id.
The assignment between job-id and model size/method can be printed by running
```[bash]
    python run.py --task=print-job-id-assignment
```
