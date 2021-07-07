from collections import defaultdict
from typing import Iterable, Iterator
from torch.utils.data import Dataset
import joblib
import os
import torch
import argparse
# import pickle
# pickle.HIGHEST_PROTOCOL = 4
import pandas as pd

class PoseBetaData(Dataset):
    def __init__(self, pkl_folder_path):


        pkl_files = os.listdir(pkl_folder_path)

        self.data = pd.read_pickle(pkl_folder_path+'/'+pkl_files[0])
        # self.data = joblib.load(pkl_folder_path+'/'+pkl_files[0])
        print('Pickle file loaded.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        poses = torch.tensor(self.data['pose'][idx],dtype=torch.float32)
        betas = torch.tensor(self.data['betas'][idx],dtype=torch.float32)
        
        return poses[:,3:], betas


            
if __name__ == '__main__':

    image_folder_path = 'ashpickle/'
    # output_path = 'out_test'
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_pickle_path', type=str,default='ashpickle/',
                        help='input pickle path')
    
    
    args = parser.parse_args()

    ds = PoseBetaData(args.image_pickle_path)
    it = iter(ds)
    a,b = next(it)
    print(f'pose size is: {a.shape}, betas size is: {b.shape}')



        




        

