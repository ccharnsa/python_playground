#import pandas as pd

from collections import Counter
import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

#df = pd.DataFrame({"headline": ["Lenovo teases what might be the first true √É¬¢√¢¬Ç¬¨√ã¬úall screen√É¬¢√¢¬Ç¬¨√¢¬Ñ¬¢ phone", "Lenovo might be first to release a smartphone with an all-screen front √É¬¢√¢¬Ç¬¨ without the iPhone X-style notch", "Intel Finally Unveils 10nm Cannon Lake Processors", "Lenovo teases a true all-screen phone", "Meet Smartisan R1, the world√É¬¢√¢¬Ç¬¨√¢¬Ñ¬¢s first phone with 1TB storage"], "link": ["xxx", "yyy", "zzz", "aaa", "bbb"]})

def preprocess_text_new(text, ps):
    '''
    Lowercase, tokenises, removes stop words and lemmatize's using word net. Returns a string of space seperated tokens.
    '''
    words = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = word_tokenize(words)

    stops = set(stopwords.words("english"))
    result = []
    for word in words:
        if word not in stops:
            continue

        stemmed = ps.stem(word)
        if len(stemmed) > 1:
            result.append(stemmed)

    return " ".join(result)

text="Lenovo teases what might be the first true Lenovo"


ngram_vectorizer = CountVectorizer(analyzer='word', tokenizer=word_tokenize, ngram_range=(1, 1), min_df=1)

X = ngram_vectorizer.fit_transform(text.split('\n'))

# Vocabulary
vocab = list(ngram_vectorizer.get_feature_names())

# Column-wise sum of the X matrix.
# It's some crazy numpy syntax that looks horribly unpythonic
# For details, see http://stackoverflow.com/questions/3337301/numpy-matrix-to-array
# and http://stackoverflow.com/questions/13567345/how-to-calculate-the-sum-of-all-columns-of-a-2d-numpy-array-efficiently
counts = X.sum(axis=0).A1

freq_distribution = Counter(dict(zip(vocab, counts)))
print(freq_distribution.most_common(10))

