import os
import torch
torch.backends.cudnn.enabled = False
def evals(path,name,seed='42',device=0,latest=False):
    seed=path.split('/')[-1]
    path=path+'/checkpoints'
    if latest:
        ckpts=[os.path.join(path, file_name) for file_name in os.listdir(path) if 'latest' in file_name]
    else:
        ckpts=[os.path.join(path, file_name) for file_name in os.listdir(path) if not 'latest' in file_name]
    print('--------------start--------------')
    for i,ckpt in enumerate(ckpts):
        epochscore=ckpt.split('/')[-1][:-5]
        os.system('python eval.py --checkpoint {} --output_dir data/{}/{} --device cuda:{}'.format(ckpt,name,seed+epochscore,device))
    print('--------------done--------------')
    
def single_evals(ckpt_paths,test=50,device=1):
    print('--------------start--------------')
    for ckpt in ckpt_paths:
        gen=ckpt.split('/')[-4]
        for i in range(test):
            os.system('python eval.py --checkpoint {} --output_dir data/{}/{} --device cuda:{}'.format(ckpt,'tvideos/'+gen,str(i),device))
    print('--------------done--------------')



#evals for all ckpts
ckpt_dirs=['./data/outputs/yourname/'+seed for seed in os.listdir('./data/outputs/yourname')]

for i,j in enumerate(ckpt_dirs):
    evals(j,'your name',str(i+seed),latest=False)



#evals for single ckpt and generate the trajectory and videos , if 'videos' in output_dir the code will be automatic to generate. Refer to ./test for test smoothness.
ckpt_path=['./data/outputs/lift_ph_cnn/44/checkpoints/epoch=0350-test_mean_score=1.000.ckpt']

single_evals(ckpt_path)

