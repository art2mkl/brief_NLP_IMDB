#shell : uvicorn main:app --reload

from fastapi import FastAPI
import pandas as pd
import numpy as np
import json

import nltk
nltk.download('stopwords') 
nltk.download('wordnet')

#!pip install wordcloud

#scrap
import requests
from bs4 import BeautifulSoup

# Text preprocessing and modelling
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='talk')

from wordcloud import WordCloud

# Warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Stopwords
stop_words = set(ENGLISH_STOP_WORDS).union(stopwords.words('english'))
stop_words = stop_words.union(['let', 'mayn', 'ought', 'oughtn','shall'])

#SCRAP IMDB
def scraping():   
    """
    ------------------------------------------------------------------------------------------------------
    scrap data on url imdb
    return dataframe
    ------------------------------------------------------------------------------------------------------
    """

    liste_synopsis = []
    titres = []
    summary = []

    for i in range(1, 1001, 50):

        #avancement
        print(f"Scrap de la partie {i} sur {i+49}")

        # connect db
        url = f"https://www.imdb.com/search/title/?title_type=feature&num_votes=5000,&sort=user_rating,desc&start={i}&ref_=adv_nxt"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        soup = soup.find_all(class_='lister-item-content')

        # scrap summary
        summary += [element.find_all("p", class_='text-muted')[-1].text.replace(
            "\n", "").replace("See full summary\xa0»\n')", "") for element in soup]

        # scrap movies href
        liens = [element.find("a")['href'] for element in soup]

        for lien in liens:

            # connect href
            url_2 = f'https://www.imdb.com{lien}?ref_=adv_li_tt'
            response_2 = requests.get(url_2)
            soup_2 = BeautifulSoup(response_2.content, 'html.parser')

            # scrap synopsis
            liste_synopsis.append(soup_2.find(
                class_='inline canwrap').text.replace("\n", "").strip())

            # scrap titles
            titres.append(soup_2.find('h1').text.replace('\xa0', '').split('(')[0])
            
    return pd.DataFrame({'titres': titres, 'synopsis': liste_synopsis, 'resume': summary})

#Launch scrapping
#df = scraping()
#save in csv
#df.to_csv("imdb_1000.csv", index = False)

app = FastAPI()


df = pd.read_csv('imdb_1000.csv')

df2 = df.copy()

#Delete some keywords

#Delete written
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('Written',''))

#delete family
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('family',''))

#delete life
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('life',''))

#delete year
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('year',''))

#delete man
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('man',''))

#delete story
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('story',''))

#delete time
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('time',''))

#delete movie
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('movie',''))

#delete film
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('film',''))

#delete Tamil
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('Tamil',''))

#delete tamil
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('tamil',''))

#delete come
df['synopsis'] = df['synopsis'].apply(lambda x: x.replace('come',''))

#Preprocessing

def preprocess_text(document):
    """
    -------------------------------------------------------------------------
    - keep only alpha character group by 3, select somes tag word, remove stop-words, lemmatize words, Preprocess document into normalised tokens.
    - return array of final words 
    -------------------------------------------------------------------------
    """

    # Tokenise words into alphabetic tokens with minimum length of 3
    tokeniser = RegexpTokenizer(r'[A-Za-z]{3,}')
    tokens = tokeniser.tokenize(document)
    
    # Tag words with POS tag
    #pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}
    pos_map = {'N': 'n'}
    
    pos_tags = pos_tag(tokens)
    
    # Lowercase and lemmatise 
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(t.lower(), pos=pos_map.get(p[0], 'v')) for t, p in pos_tags]
    
    # Remove stopwords
    keywords= [lemma for lemma in lemmas if lemma not in stop_words]
    return keywords

#split Dataset
X_train, X_test = train_test_split(df, test_size=0.2, 
                                   random_state=1)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

#Model creation¶

def describe_topics(lda, feature_names, top_n_words=5, show_weight=False):
    """
    ------------------------------------------------------------------------------------------------------
    show main words of each topics from lda model
    return none
    ------------------------------------------------------------------------------------------------------
    """

    normalised_weights = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    
    for i, weights in enumerate(normalised_weights):  
        print(f"********** Topic {i+1} **********")
        
        if show_weight:
            feature_weights = [*zip(np.round(weights, 4), feature_names)]
            feature_weights.sort(reverse=True)
            print(feature_weights[:top_n_words], '\n')
        
        else:
            top_words = [feature_names[i] for i in weights.argsort()[:-top_n_words-1:-1]]
            print(top_words, '\n')

#Define number of Contents
#Topics creation fon n_components = n

#**************************************
n_components = 3
#**************************************

pipe = Pipeline([('vectoriser', CountVectorizer(analyzer=preprocess_text, min_df=5)),
                 ('lda', LatentDirichletAllocation(n_components=n_components, topic_word_prior = 0.9, doc_topic_prior = 0.1, learning_method='batch', random_state=0))])

pipe.fit(X_train['synopsis'])

# Inspect topics
feature_names = pipe['vectoriser'].get_feature_names()
describe_topics(pipe['lda'], feature_names, top_n_words=20)

#Transform Train & Viz
pd.options.display.max_colwidth = 50
train = pd.DataFrame(X_train)
columns = ['topic'+str(i+1) for i in range(n_components)]
train[columns] = pipe.transform(X_train['synopsis'])
#train.head()

#Create df with Top 1 to 3

train = train.assign(top1=np.nan, prob1=np.nan, top2=np.nan, 
                     prob2=np.nan, top3=np.nan, prob3=np.nan)

top_liste = [f'top{i}' for i in range(1,n_components+1)]
top_prob = [f'prob{i}' for i in range(1,n_components+1)]
last_topic = f'topic{n_components}'

for record in train.index:
    top = train.loc[record, 'topic1':last_topic].astype(float).nlargest(n_components)
    train.loc[record, top_liste] = top.index
    train.loc[record, top_prob] = top.values
#train.drop(['synopsis'], axis=1).head()






  
@app.get('/line/')
async def root():
    data = json.loads(df2.to_json())
    return data

@app.get('/lda/')
async def root():
    data2 = json.loads(train.to_json())
    return data2

@app.post('/add/')
async def add_line(data2):
    return data2

@app.get('/viztopics/')
async def viztopics():

    df3 = train[['titres','top1']]
    df_compare=pd.DataFrame()
    df4 = df3.sample(len(df3))

    for i in range(1,n_components+1):
        df_compare[f'topic{i}'] = list(df4['titres'][df4['top1'] == f'topic{i}'].head(10))

    data3 = json.loads(df_compare.to_json())
    return data3

@app.post('/choice/')
async def add_line(data_1, data_2, data_3):
    return [data_1, data_2, data_3]

