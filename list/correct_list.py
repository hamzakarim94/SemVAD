import pandas as pd
import pickle
file_name = "./list/ucf_CLIP_rgbtest.csv"
df = pd.read_csv(file_name)
with open("./list/data.pkl", "rb") as f:
    loaded_data = pickle.load(f)
for i,row in enumerate(df.values):
    label = row[1]
    if label == "Normal":
        continue
    new_label = loaded_data[label]
    df.values[i][1] = new_label

df.to_csv(file_name,index=False)


for i,row in enumerate(df.values):
    path = row[0].split("Desktop/")[-1]
    df.values[i][0] = path
df.to_csv(file_name,index=False)