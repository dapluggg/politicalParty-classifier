#%%
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

mystops = ['https', 'amp', 't', 'co']
STOPWORDS.update(mystops)
#print(STOPWORDS)

#%%
def one_big_string(list_of_tweets):
    mystring = ''
    for tweet in tqdm(list_of_tweets):
        mystring += tweet
        mystring += ' '
    return mystring

def plot_wordcloud(tweetsdf):
    dem = tweetsdf.loc[tweetsdf['Party'] == 'Democrat', ]
    republican = tweetsdf.loc[tweetsdf['Party'] == 'Republican', ]

    tweetsdf = one_big_string(tweetsdf['tweet'])
    dem = one_big_string(dem['tweet'])
    republican = one_big_string(republican['tweet'])

    wordcloud = WordCloud(width = 1200, height = 800,
                background_color ='black',
                colormap = 'Blues',
                stopwords = STOPWORDS,
                min_font_size = 10).generate(dem)
  
    # plot the WordCloud image                       
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('../plots/Dem-WordCloud.png')

    wordcloud = WordCloud(width = 1200, height = 800,
                background_color ='black',
                colormap = 'Reds',
                stopwords = STOPWORDS,
                min_font_size = 10).generate(republican)
  
    # plot the WordCloud image                       
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('../plots/Rep-WordCloud.png')

    wordcloud = WordCloud(width = 1200, height = 800,
                background_color ='black',
                stopwords = STOPWORDS,
                min_font_size = 10).generate(tweetsdf)
  
    # plot the WordCloud image                       
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('../plots/DemRep-WordCloud.png')

    return None