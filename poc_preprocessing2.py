import os
import pandas as pd
import json
import csv
import sys
import csv
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from collections import Counter

def clean_text(raw_text):

    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)
    #
    # 2. Remove RT and http
    # letters_only = re.sub(r'RT', "", letters_only)
    #
    # 3. Convert to lower case
    words = letters_only.lower().split()
    #
    # words = re.sub('^(.*rt)'," ", words)
    # 4. Convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Return the result
    return (" ".join(meaningful_words))

text="HEllo python this is the good example 576473 É¬¢√¢¬Ç hello"

text=clean_text(text)
freq_distribution = Counter(text.split()).most_common()
print(freq_distribution)