#%%
import pathlib
import twint
import pandas as pd 
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

def load_accounts(file_path):
    accounts = pd.read_csv(file_path, sep='\t', header=0, dtype={
        'Name': str,
        'Handle': str,
        'Party': str}
    )
    accounts.drop_duplicates(inplace=True)
    accounts = accounts.sample(frac=1)

    return accounts

def get_tweets(handle, tweetLimit):
    """
    This function takes in a list of twitter handles to scrape tweets from using TWINT (available at https://github.com/twintproject/twint). .
    Args:
        twitterhandles (list): List of twitter handles (strings) in the conventional twitter format of '@twitterhandle'. 
        tweetLimit (int): Maximum Number of tweets to scrape from each user supplied. 

    Returns:
        list: List of json objects that contains tweets from each handle in argument twitterhandles. 
    """
    json_filename = f'./data/{handle[1:]}-tweets.json'
    f = open(json_filename, 'w+')
    c = twint.Config()
    c.Username = handle[1:]
    c.Limit = tweetLimit
    c.Store_json = True
    c.Output = json_filename
    c.Hide_output = True
    try:
        reply = twint.run.Search(c)
    except ValueError: 
        print(f'Unable to find account {handle}')
    except TypeError:
        print(f'Some other error')
    f.close()
    
    return True

def get_tweets_multip(twitterhandles, tweetLimit, num_pools):
    func = partial(get_tweets, tweetLimit=tweetLimit)
    with Pool(num_pools) as p:
        p.map(func, twitterhandles)

    return True
