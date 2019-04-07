from tika import parser

raw = parser.from_file("Resume.pdf")
print(raw['content'])

import nltk
import re 
import string
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from sklearn.preprocessing import normalize
import numpy as np 
from scipy.spatial import distance

wordnet_lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(pos_tag):
    
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ
    
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB
    
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN

    

def tokenize(text):
    text = text.lower()
    pattern=r'\w[\w\'_.@-]*\w'
    tokens = nltk.regexp_tokenize(text, pattern)
    tokens=[token.strip(string.punctuation) for token in tokens]
    tokens=[token.strip() for token in tokens if token.strip()!='']
    #print(len(tokens))                   
    #print(tokens)
    tagged_tokens= nltk.pos_tag(tokens)
    stop_words = stopwords.words('english')
    stop_words+=["they'll", "can't"]    
    lemmatized_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for (word, tag) in tagged_tokens if word not in stop_words and word not in string.punctuation]
    token_count=nltk.FreqDist(lemmatized_words)
    return token_count

# Q2

def find_similar_doc(doc_id, docs):
    
#     print(doc_id)
#     print(docs)
    
    docs_tokens={idx:tokenize(doc) for idx,doc in enumerate(docs)}
#     print(docs_tokens)
    
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
    
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    
    df=np.where(tf>0,1,0)
    
    idf=np.log(np.divide(len(docs),np.sum(df, axis=0)))+1
    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=normalize(tf*smoothed_idf)
    
    tf_idf=normalize(tf*idf)
    
    similarity = 1-distance.squareform(distance.pdist(tf_idf, 'cosine'))
#     print(similarity)
   
    best_matching_doc_id = np.argsort(similarity)[:,::-1][int(doc_id),1:2]
#     print(best_matching_doc_id)
    
    return best_matching_doc_id, similarity


data=pd.read_csv("resume.csv", header=0)
#     print(data.head(52))
    resume_id=15
    x,y=find_similar_doc(doc_id, data["Description"].values.tolist())
#     print(x,y)
    print(x)
    print(data["Description"].iloc[doc_id])
    print(data["Description"].iloc[x])
