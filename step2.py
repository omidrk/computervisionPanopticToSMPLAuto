from collections import defaultdict
from torch.utils.data import Dataset
import joblib
import os

class PoseBetaData(Dataset):
    """pose and beta dataset"""

    def __init__(self, pkl_folder_path):
    
          
        pkl_files = os.listdir(pkl_folder_path)
        data = []
        for pkls in pkl_files:
            data.append(joblib.load(pkl_folder_path+'/'+pkls))
            
        keys = []
        for d in data:
            for k in d.keys():
                keys.append(k)
                
        if len(data) != len(keys):
            prin("err")
            
        self.length = len(data[0][keys[0]]['pose'])
        
        self.dataset = defaultdict(list)
        
        for l in range(self.length):
            poses = []
            betas = []
            for n_cam in range(len(data)):
                poses.append(data[n_cam][keys[n_cam]]['pose'][l][3:])
                betas.append(data[n_cam][keys[n_cam]]['betas'][l])
            self.dataset['pose'].append(poses)
            self.dataset['beta'].append(betas)
            
            

    def __len__(self):
        return len(self.length)

    def __getitem__(self, idx):
        
        poses = self.dataset['pose'][idx]
        betas = self.dataset['beta'][idx]
        
        return poses, betas