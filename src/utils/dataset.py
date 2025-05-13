import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        #self.text_test_feats = np.load("text_feats_test.npy", allow_pickle=True)
        #self.text_256_feats = np.load("text_feats_256.npy", allow_pickle=True)
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        #print(self.df.loc[index]['path'])
        V_label = self.df.loc[index]['path'].split("/")[-1].replace(".npy", "").split("__")[0]
        text_feats = []
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            '''if len(clip_feature) > 256:
                label_idx = "vlm_feats/"+V_label +".npy"
                text_feats = np.load(label_idx)
                clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
                text_feats = np.asarray(text_feats)
                clip_feature = np.concatenate((clip_feature, text_feats), axis=1)
            else:
                label_idx = "vlm_feats/"+V_label + ".npy"
                text_feats = np.load(label_idx)
                clip_feature = clip_feature[:len(text_feats)]
                clip_feature = np.concatenate((clip_feature, text_feats), axis=1)
                clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)#'''
        else:
            '''label_idx = "vlm_feats/"+V_label +".npy"
            text_feats = np.load(label_idx)

            try:
                clip_feature = np.concatenate((clip_feature, text_feats), axis=1)
            except:
                if len(clip_feature) != len(text_feats):
                    text_feats = np.vstack([text_feats, text_feats[-1]])
                    clip_feature = np.concatenate((clip_feature, text_feats), axis=1)
                #print(label_idx,len(clip_feature)-len(text_feats))'''
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length,V_label

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        V_label = self.df.loc[index]['path'].split("/")[-1].replace(".npy", "").split("__")
        v_label = V_label[0]
        for l in range(1,len(V_label) - 1,1):
            v_label = v_label + "__" +V_label[l]
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length,v_label