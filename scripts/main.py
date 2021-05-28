#%%
import plotting as pt
import pandas as pd
import pathlib
import fetch_tweets as ft
import vectorize as vc
import analysis as an
import models as md
DATA_DIR=pathlib.Path('../data')
HANDLES_PATH=pathlib.Path(DATA_DIR / 'handles.txt')

#%%
def main(download_data=False, dry_run=False):
    if download_data:
        handles_df = ft.load_accounts(HANDLES_PATH)
        print('Attempting to get tweets from the following handles: ')
        print(handles_df['Handle'])
        tweets = ft.get_tweets_multip(handles_df['Handle'], 2000, 128)
        an.load_jsons(DATA_DIR)
        return download_data
    else:
        if dry_run:
            compressed_file = 'SMALL_tweetsdf.bz2'
        else:
            compressed_file = 'tweetsdf.bz2'
        tweetdf = pd.read_pickle(DATA_DIR / compressed_file, compression='bz2')
        tweetdf = an.get_party_names(tweetdf, HANDLES_PATH)
        pt.plot_wordcloud(tweetdf)
        # Vectorize
        cv_binary_vecs = vc.vectorize_reviews_sklearn_countvec_binary(tweetdf)
        cv_vecs = vc.vectorize_reviews_sklearn_countvec(tweetdf)
        tfidf_vecs = vc.vectorize_reviews_sklearn_tfidf(tweetdf)
        # Fit Models
        bernoulliNB_model = md.train_bernoulliNB(cv_binary_vecs, tweetdf['Party'], ['Democrat', 'Republican'], 5)
        multinomialNB_model = md.train_multinomNB(cv_vecs, tweetdf['Party'], ['Democrat', 'Republican'], 5)
        multinomialNB_model_tfidf = md.train_multinomNB(tfidf_vecs, tweetdf['Party'], ['Democrat', 'Republican'], 5)

        mymodels = [bernoulliNB_model, multinomialNB_model, multinomialNB_model_tfidf]
        myvecs = [cv_binary_vecs, cv_vecs, tfidf_vecs]
        return 
    

if __name__ == "__main__":
    tmp = main(download_data=False, dry_run=False)
# %% 

