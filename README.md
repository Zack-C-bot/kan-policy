# KAN POLICY

## Simulation

<div align="center">
  <p align="center">
    <img src="./media/preview.gif" alt="GIF演示" width="900" style="max-width:100%; border-radius:8px;" />
  </p>
  <p align="center" style="margin-top:8px; font-size:14px;">
    Preview of our work (The full demo video has been uploaded to our codebase)
  </p>
</div>


###  Installation
The code environment setup follows the same installation steps as [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), and we thank the authors for sharing their codebase:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```
Build a conda environment frist:
```console
$ conda env create -f conda_environment.yaml & conda activate kp
```
For transformer-based models we apply Kat Group,  please follow the instuctions to install rational_kat:
```console
$ git clone https://github.com/Adamdad/rational_kat_cu.git
$ cd rational_kat_cu
$ pip install -e .
```

###  Train

Build the data folder:
```console
$ mkdir data & cd data
```
Download the dataset, as example (you can also get Robotmimic datasets from https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip):
```console
$ wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
```
Extract it to the folder and back:
```console
$ unzip pusht.zip & cd ..
```
Launch training with seed 42 on GPU 0, we trained the Push-T on the NVIDIA RTX 2080 Ti GPU. More configs can be found in the directory ```config```:
```console
$ python train.py --config-dir=. --config-name=pusht.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/pusht/42'
```
###  Eval
You can eval with single seed on GPU 0.
```console
$ python eval.py --checkpoint data/your_name.ckpt --output_dir data/pusht_eval_output --device cuda:0
```

#### Reference statistical code
If your experimental directory like this:
```
data
├── outputs
    ├── pusht
       ├── 42 
           ├── checkpoints
               ├── epoch=2100-test_mean_score=0.977.ckpt
               ... 
               └── latest.ckpt  
      ├── 43
          ├── checkpoints
              ├── epoch=2850-test_mean_score=0.997.ckpt
              ... 
              └── latest.ckpt
      └── 44
          ├── checkpoints
              ├── epoch=3150-test_mean_score=0.978.ckpt
              ...  
              └── latest.ckpt
```
You can eval with all ckpts across three seeds with editing the code in evals.py and then run, the code also can generate trajectories.
```console
$ python evals.py
```
Samely with the mean time, you can edit the code in time_summary.py and run.
```console
$ python time_summary.py
```

### Notice
1.We utilized a significant number of GPUs for model training, including 2080Ti, 4080super, 4090 and 5090. If needed, we can initially provide [ckpts](https://pan.baidu.com/s/18in2PwhOfLJ_01XPG6XTmQ?pwd=kpkp) for a subset of the tasks.

### Acknowledgement
We are pleased to announce that our paper titled "KAN Policy: Learning Efficient and Smooth Robotic Trajectories via Kolmogorov-Arnold Networks" has been accepted for publication in IEEE RA-L. We extend our sincere gratitude to the reviewers and editors for their valuable feedback and recognition of our work.

