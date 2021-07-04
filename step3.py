from collections import defaultdict
import pandas as pd
import os




def clean_data(pkl_folder_path, output_path, list_keys = ['pose', 'betas'], pikl_protocol = 4):
    
    df = pd.DataFrame()
    pkl_files = os.listdir(pkl_folder_path)
    data = []
    for pkls in pkl_files:
        data.append(joblib.load(pkl_folder_path+'/'+pkls))

    keys = []
    for d in data:
        for k in d.keys():
            keys.append(k)

    section = set(data[0][keys[0]]['frame_ids'])
    
    for idx, i in enumerate(data):
        section = section.intersection(set(i[keys[idx]]['frame_ids']))
        
        
    for frame in section:        
        k = defaultdict(list)
        for ind, d in enumerate(data):
            index = np.where(d[keys[ind]]['frame_ids'] == frame)[0][0]
            for key in list_keys:
                k[key].append(np.array(d[keys[ind]][key][index]))
        df = df.append(k, ignore_index=True)
        
    df.to_pickle(output_path + 'pikel_ASLI.pkl', protocol=pikl_protocol)
    print(f"file save in {output_path}pikel_ASLI.pkl")
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl_folder_path', type=str,default='',
                        help='input pkls floder path')
    
    parser.add_argument('--output_path', type=str,default='',
                        help='output path')
    
    parser.add_argument('--list_keys', type=list,default=['pose', 'betas'],
                        help='list of the key that we want to save')
    
    parser.add_argument('--pikl_protocol', type=int,default=4,
                        help='pikl protocol for saving')


        
    
    
    args = parser.parse_args()

    main(args.pkl_folder_path, args.output_path, args.list_keys, args.pikl_protocol)