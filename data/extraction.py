import json 
import os
import pathlib
 
folder= "C:/Users/Dell/Desktop/Exam papers/thesis/data/data/test"
for file in os.listdir(folder):
    if file.endswith(".json"):
        file_path= os.path.join(folder,file)
        with open(file_path ,'r',encoding = 'utf-8') as f:
            data =json.load(f)
            print(file_path)


