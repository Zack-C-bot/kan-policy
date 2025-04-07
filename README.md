# KAN POLICY
## Simulation
###  Installation
To reproduce our simulation benchmark results, install the provided conda environment on a Linux machine equipped with Nvidia GPUs. For Ubuntu, you will need to install the following apt packages to support MuJoCo:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```
build a conda environment frist:
```console
$ conda env create -f conda_environment.yaml & conda activate kp
```
for transformer-based models we apply Kat Group,  please follow the instuctions to install rational_kat:
```console
$ git clone https://github.com/Adamdad/rational_kat_cu.git
$ cd rational_kat_cu
$ pip install -e .
```

###  Train

build the data folder:
```console
$ mkdir data & cd data
```
download the dataset, as example:
```console
$ wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
```
Extract it to the folder and back:
```console
$ unzip pusht.zip & cd ..
```
Launch training with seed 42 on GPU 0, we trained the Push-T on the NVIDIA RTX 2080 Ti GPU:
```console
$ python train.py --config-dir=. --config-name=pusht.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/pusht/42'
```
###  Eval
you can eval with single seed on GPU 0.
```console
$ python eval.py --checkpoint data/your_name.ckpt --output_dir data/pusht_eval_output --device cuda:0
```

#### Reference statistical code
if your experimental directory like this:
```
data
├── outputs
    ├── pusht
       ├── 42 
           ├── checkpoints
               ├── epoch=0300-test_mean_score=1.000.ckpt
               └── latest.ckpt  
      ├── 43
          ├── checkpoints
              ├── epoch=0250-test_mean_score=1.000.ckpt
              └── latest.ckpt
      └── 44
          ├── checkpoints
              ├── epoch=0250-test_mean_score=1.000.ckpt
              └── latest.ckpt
```
you can eval with all ckpts across three seeds with editing the code in evals.py and then run, the code also can generate trajectories.
```console
$ python evals.py
```
samely with the mean time, you can edit the code in time_summary.py and run.
```console
$ python time_summary.py
```


### Notice

1.We utilized a significant number of GPUs for model training, and we'll be sharing full details along with parameter configurations and our model ckpts in subsequent updates.

2.Our code is built upon Diffusion Policy, with modifications implemented to accommodate our experimental requirements.


## Real-World
coming soon...
