import json
import pandas as pd 
import pathlib 
import os
from tqdm import tqdm
import bz2
import bz2
import _pickle as cPickle


DATA_DIR=pathlib.Path('../data')
HANDLES_PATH=pathlib.Path(DATA_DIR / 'handles.txt')

#%%
def load_jsons(datapath):
    tweetsdf = pd.DataFrame()
    for json in tqdm(os.listdir(datapath)):
        if json.endswith('.json'):
            reader = pd.read_json(datapath / json, lines=True)
            reader['handle'] = json.replace('-tweets.json', '')
            tweetsdf = tweetsdf.append(reader)
    tweetsdf.to_pickle(datapath / 'tweetsdf.bz2', compression='infer')

    return None

def get_party_names(tweetsdata, handles_filepath):
    meta = pd.read_csv(handles_filepath, sep='\t')
    meta['handle'] = meta['Handle'].str[1:]
    tweetsdata = pd.merge(tweetsdata, meta, on='handle')
    #tweetsdata.drop(['Handle', 'Name'])
    print(tweetsdata.columns)
    return tweetsdata 

#%%

# %%
