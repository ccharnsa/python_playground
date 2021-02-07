from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import pandas as pd
n_features = 1000
n_topics = 4
    
df = pd.read_csv("/Users/owner/Documents/GMU/Spring 2017/AIT 690/Project/src/Cluster4.csv")

data = data['Message'].tolist()

tf_vectorizer = CountVectorizer(analyzer = "word",
                                stop_words = 'english',
                                token_pattern = r'\b[a-zA-Z]{3,}\b', 
                                min_df = 5)
tf = tf_vectorizer.fit_transform(data)

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=10,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
lda.fit(tf)
pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)


tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)