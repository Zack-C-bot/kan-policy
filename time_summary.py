import os
from moviepy.editor import VideoFileClip
import pandas as pd
import json
import numpy as np

def calculate_video_durations(folder_path,limit=30):
    total_duration = 0
    num_videos = 0
    count=0
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            with VideoFileClip(video_path) as video:
                duration = video.duration 
                total_duration += duration
                num_videos += 1
                if duration!=limit:
                    count+=1
    if num_videos > 0:
        avg_duration = total_duration / num_videos
    else:
        avg_duration = 0

    return total_duration, avg_duration,count/num_videos
def score_single(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["test/mean_score"]
     
def time_summary(file_path,score,base=False):
    ckpt_results=[os.path.join(file_path, file_name) for file_name in os.listdir(file_path) if not '.csv' in file_name]
    results=[]
    maxs,score=score
    sum=0
    avg_sum=0
    scount=0
    times=[]
    for result in ckpt_results:
        media=result+'/media'
        total,avg,count=calculate_video_durations(media)
        sum=sum+total
        avg_sum=avg_sum+avg
        scount=scount+count
        results.append({
            "ckpt_name": result,
            "Total Duration (seconds)": total,
            "Average Duration (seconds)": avg,
            'Percentage':count
        })
        if result in maxs:
            times.append(avg)
    assert len(times)==3
    results.append({
        "ckpt_name": 'summary',
        "Total Duration (seconds)": sum/len(ckpt_results),
        "Average Duration (seconds)": np.mean(times),
        "Percentage (%)": scount/len(ckpt_results),
    })
    
    df = pd.DataFrame(results)
    df["score"]=score['score']
    df = (
        df
        .sort_values(by=['score', 'Average Duration (seconds)'], ascending=[False, True])  
        .drop_duplicates(subset=['ckpt_name'])  
    )
    if base==True:
        df.to_csv("./total.csv", index=False)
    else:
        df.to_csv("{}/toseed.csv".format(file_path),index=False)
    filtered_df = df[df['ckpt_name'].str.contains('42|43|44')]
    result = filtered_df.groupby(filtered_df['ckpt_name'].str.extract(r'(42|43|44)')[0])['Average Duration (seconds)'].min()
    name=file_path.split('/')[-1]
    print(f'model:{name}, best score times: {np.mean(times)}, best_times: {result.mean()}')
    #intersection
    # idx = filtered_df.groupby(filtered_df['ckpt_name'].str.extract(r'(42|43|44)')[0])['Average Duration (seconds)'].idxmin()
    # name = filtered_df['ckpt_name'].loc[idx]
    idx = df.groupby(df['ckpt_name'].str.extract(r'(42|43|44)')[0])['score'].idxmax()
    name = df['ckpt_name'].loc[idx]
    return name.to_list()
        
def score_summary(file_path,base=False):
    ckpt_results=[os.path.join(file_path, file_name) for file_name in os.listdir(file_path) if not '.csv' in file_name]
    results=[]
    sum=0.
    for result in ckpt_results:
        eval_log=result+'/eval_log.json'
        score=score_single(eval_log)
        sum=sum+score
        results.append({
            "ckpt_name": result,
            "score": score,
        })
    results.append({
        "ckpt_name": 'summary',
        "score": sum/len(ckpt_results)
    })
    df = pd.DataFrame(results)
    if base==True:
        df.to_csv("./total.csv", index=False)
    else:
        df.to_csv("{}/soseed.csv".format(file_path),index=False)
    filtered_df = df[df['ckpt_name'].str.contains('42|43|44')]
    result = filtered_df.groupby(filtered_df['ckpt_name'].str.extract(r'(42|43|44)')[0])['score'].max()
    name=file_path.split('/')[-1]
    print(f'model:{name},score :{result.mean()}') 
    idx = filtered_df.groupby(filtered_df['ckpt_name'].str.extract(r'(42|43|44)')[0])['score'].idxmax()
    idxs=filtered_df.loc[idx]
    return idxs['ckpt_name'].to_list(),df


def time_one(ori,kan,limit=30.0,inter=True,save=True):
    assert len(ori)==len(kan)==3
    for o in range(len(ori)):
        for j in range(len(ori)):
            time_dicts=[]
            file_paths= [ori[o],kan[j]]
            for file_path in file_paths:
                results={}
                map1={}
                map2={}
                folder_path=file_path+'/media'
                media_paths=os.listdir(folder_path)
                with open(file_path+'/eval_log.json', 'r', encoding='utf-8') as file:
                    score_data = json.load(file)
                for k in range(len(media_paths)):
                    if k==50:
                        break
                    map1[k]=score_data["test/sim_video_{}".format(k+100000)].split('/')[-1]
                for filename in media_paths:
                    if filename.endswith(".mp4"):
                        video_path = os.path.join(folder_path, filename)
                        with VideoFileClip(video_path) as video:
                            duration = video.duration 
                            results[filename]=duration
                map2 = {key: results[value] for key, value in map1.items()}
                if save:
                    with open(file_path+'/best_times.json', 'w') as json_file:
                        json.dump(results, json_file,indent=4)
                time_dicts.append(map2)
            if inter:
                total=0
                count=0
                assert len(file_paths)==2
                count1=0
                count2=0
                kcount=[]
                ocount=[]
                fcount=[]
                temp=0
                index=-1
                for i in range(len(time_dicts[0])):
                    if temp<time_dicts[0][i]-time_dicts[1][i]:
                        temp=time_dicts[0][i]-time_dicts[1][i]
                        index=i
                    if time_dicts[0][i]!=limit and time_dicts[1][i]!=limit:
                        count+=1
                        count1=count1+time_dicts[0][i]
                        count2=count2+time_dicts[1][i]
                        total+=time_dicts[0][i]-time_dicts[1][i]
                    elif time_dicts[0][i]!=limit and time_dicts[1][i]==limit :
                        kcount.append(i)
                    elif time_dicts[0][i]==limit  and time_dicts[1][i]!=limit :
                        ocount.append(i)
                    else:
                        fcount.append(i)
                name=file_paths[0].split('/')[-2]+'_'+file_paths[0].split('/')[-1]+'————'+file_paths[1].split('/')[-1]
                data=[name,total,total/count,count,count1/count,count2/count,kcount,ocount,fcount,temp,index]
                print(f'model:{name}',total,total/count,count,count1/count,count2/count,kcount,ocount,fcount,temp,index)
                path = "YOURPATH"
                if not os.path.exists(path):
                    os.mkdir(path)
                    with open(path+'output.txt', "w") as f:
                        for item in data:
                            f.write(str(item),' ') 
                else:
                    with open(path+'output.txt', "a") as f:
                        for item in data:
                            f.write(str(item)+'  ') 
                        f.write('\n')
        with open(path+'output.txt', "a") as f:
            f.write('\n')
    with open(path+'output.txt', "a") as f:
        f.write('**************')
        
ckpt_lists=['YOURPATH']


for i in ckpt_lists:
    time_summary(i,score_summary(i))

# this code is to get the mean average time for the intersection of test environments


# ckpt_lists=['YOURPATH']
#
# time_dict={'lift_ph':20.0,'lift_mh':25.0,'can_ph':20.0,'can_mh':25.0,'square_ph':20.0,'square_mh':25.0,'tool_hang':35.0,'pusht':30.0,'transport':35.0}
# def map_time(di,target):
#     limit=-1
#     for key,value in di.items():
#         if key in target:
#             limit=value
#             break
#     assert limit!=-1
#     return limit
# for i in ckpt_lists:
#     ori =time_summary(i,score_summary(i))
#     kan= time_summary(i+'_k',score_summary(i+'_k'))
#     time_one(ori,kan,map_time(time_dict,ori[0]))

                

        
    
