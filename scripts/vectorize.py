#%%
"""
Author: Shashank Nagaraja
Class: IST 736 Text Mining
References: 
1. https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
   I used the above stack overflow post to figure out how to add the PorterStemmer to CountVectorizer/TfidfVectorizer
2. Applied text analysis with Python (Bilbro & Ojeda 2021)

"""
import glob, os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
# from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
DATA_PATH = Path('data/')
PLOTS_PATH = Path('plots/')
MIN_DF=0.01
#%%

class StemmedCountVectorizer(CountVectorizer):
    '''
    Building my own vectorizer based on sklearn CountVectorizer. This includes stemming using NLTKs PorterStemmer
    '''
    def build_analyzer(self):
        stemmer = PorterStemmer()
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

class StemmedTFIDFVectorizer(TfidfVectorizer):
    '''
    Building my own vectorizer based on sklearn TfidfVectorizer. This includes stemming using NLTKs PorterStemmer
    '''
    def build_analyzer(self):
        stemmer = PorterStemmer()
        analyzer = super(StemmedTFIDFVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
'''
Building the four vectorizers using the hyperparameters envisioned. 
'''
def vectorize_reviews_sklearn_countvec(reviewsdf):
    sklearn_vec = CountVectorizer(encoding='latin-1', binary=False, stop_words='english', min_df=MIN_DF)
    vecs = sklearn_vec.fit_transform(reviewsdf['tweet'])
    #print(sklearn_vec.vocabulary_)
    result = pd.DataFrame(vecs.toarray())
    result.columns = sklearn_vec.get_feature_names()
    return result

def vectorize_reviews_sklearn_countvec_binary(reviewsdf):
    sklearn_vec = CountVectorizer(encoding='latin-1', binary=True, stop_words='english', min_df=MIN_DF)
    vecs = sklearn_vec.fit_transform(reviewsdf['tweet'])
    #print(sklearn_vec.vocabulary_)
    result = pd.DataFrame(vecs.toarray())
    result.columns = sklearn_vec.get_feature_names()
    return result

def vectorize_reviews_sklearn_countvec_stemmed(reviewsdf):
    sklearn_vec = StemmedCountVectorizer(encoding='latin-1', binary=False, stop_words='english', min_df=MIN_DF)
    vecs = sklearn_vec.fit_transform(reviewsdf['tweet'])
    #print(sklearn_vec.vocabulary_)
    result = pd.DataFrame(vecs.toarray())
    result.columns = sklearn_vec.get_feature_names()
    return result

def vectorize_reviews_sklearn_tfidf(reviewsdf):
    sklearn_vec = TfidfVectorizer(encoding='latin-1', use_idf=True, stop_words='english', min_df=MIN_DF)
    vecs = sklearn_vec.fit_transform(reviewsdf['tweet'])
    #print(sklearn_vec.vocabulary_)
    result = pd.DataFrame(vecs.toarray())
    result.columns = sklearn_vec.get_feature_names()
    return result

def vectorize_reviews_sklearn_tfidf_stemmed(reviewsdf):
    sklearn_vec = StemmedTFIDFVectorizer(encoding='latin-1', use_idf=True, stop_words='english', min_df=MIN_DF)
    vecs = sklearn_vec.fit_transform(reviewsdf['tweet'])
    #print(sklearn_vec.vocabulary_)
    result = pd.DataFrame(vecs.toarray())
    result.columns = sklearn_vec.get_feature_names()
    return result

def plot_TFs(TF_Dataframe):
    """Plot Vocabulary Counts. In case of TFIDF, sum all scores for a given term. 

    Args:
        TF_Dataframe (Array): Array output of Vectorizer

    Returns:
        Series: Named series of count/tfidf sums for each term in vocabulary. 
    """
    barplot_series = TF_Dataframe.sum(axis=0)
    barplot_series = barplot_series.sort_values()
    g = sns.barplot(x=barplot_series.index, y=barplot_series.values)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    print(g)
    return barplot_series

def main():
    """
    Main function to run the entire pipeline. Very pythonic :)
    """
    reviews = load_corpus(DATA_PATH)
    vectorized_reviews, vectorizer = vectorize_reviews_sklearn_tfidf_stemmed(reviews)
    resultdf = pd.DataFrame(vectorized_reviews.toarray(), columns=vectorizer.get_feature_names())
    plots = plot_TFs(resultdf)
    print(resultdf)
    output_path = DATA_PATH / 'vectorized.csv'
    resultdf.to_csv(output_path)
    return resultdf

if __name__ == '__main__':
    main()

# %%
