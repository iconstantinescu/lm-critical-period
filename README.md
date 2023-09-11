# Language Models Critical Period


## Creating environment on Euler

Make sure you are in the root of your project.

```
module load eth_proxy gcc/8.2.0 python_gpu/3.9.9
python -m venv --system-site-packages venv
source ./venv/bin/activate

pip install --upgrade pip
```

Whenever you install a new package, make sure your venv is activated!

To install new packages:
```
pip install --no-cache-dir <package_names>
```

If you need a different version of an existing package, then you can also install it in your virtual environment. For instance for installing a newer numpy version:

```
OPENBLAS=$OPENBLAS_ROOT/lib/libopenblas.so pip install --ignore-installed --no-deps numpy==1.20.0
```

List of existing packages on Euler:  https://scicomp.ethz.ch/wiki/Python_on_Euler

To install required packages for this project:

```
OPENBLAS=$OPENBLAS_ROOT/lib/libopenblas.so pip install --ignore-installed --no-cache-dir -r requirements.txt
```

You might also want to login to your wandb account (only once): ```wandb login```

## Running the scripts on Euler

Before running the scripts, make sure you modify the *.euler files accordingly:
* Modify SBATCH options (e.g. resource requests) 
* Modify the venv source path: ```source <path_to_project>/venv/bin/activate```

Run the scripts from the project root.

```
### See help for all the available options ###
./scripts/train.sh -h

### Train a model with default configuration ###
./scripts/train.sh

### Train a model with custom configuration ###
./scripts/train.sh -n <model_name> -d <dataset> -1 <lang1> -s <seed> -p <project_name>
```

Run a sweep:

```
1. Create a new sweep yaml configuration in ./src/learn/configs folder
2. ./scripts/sweep.sh -n <config_name> -1 <lang1> -p <project_name>
3. ./scripts/train.sh -n <model_name> -1 <lang1> -w <sweep_id>

```

## Creating environment locally with Conda

First, create a conda environment with
```bash
$ conda env create -f util/environment.yml
```
Then activate the environment with `source util/activate.sh` and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers[torch]
$ pip install datasets
$ pip install evaluate
$ pip install wandb
```


## Running the scripts Locally

First, train the model in lang1 with command:
```bash
$ DO_TEST=true MODE=sequential MODEL=gpt2 CONFIG=config11 DATASET=unified_clean LANG1=fi LANG2=en SEED=7 source src/learn/train_model.sh
```

And then run to finetune the model in the lang2:
```bash
$ MODE=sequential CHECKPOINT=gpt2-config11-fien-sequential-7-1109-2/checkpoint-2500 MODEL=gpt2 DATASET=unified_clean LANG1=fi LANG2=en SE
ED=7 source src/learn/train_model.sh
```