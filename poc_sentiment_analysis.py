from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
# nltk.download('vader_lexicon')
# pip install praw
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
# sns.set(style='darkgrid', context='talk', palette='Dark2')

# import praw
def write_file(file_name, df):
	df.to_csv("output/"+file_name, encoding='utf-8', index=False)


# reddit = praw.Reddit(client_id='KgFBUmRPl1whtw',
#                      client_secret='gmOJmuS3iDRwrUnhmn0CYtpnGGY',
#                      user_agent='LearnDataSci')

# headlines = set()
# for submission in reddit.subreddit('politics').new(limit=None):
#     headlines.add(submission.title)
#     display.clear_output()
#     # print(len(headlines))


lenovo_df = pd.read_csv("input/lenovo_inbound_20180615.csv", encoding="utf-8")

print(len(lenovo_df))
# headlines = lenovo["Headline"].tolist()


sia = SIA()
results = []

for i, row in lenovo_df.iterrows():
	line = row["Headline"]
	url = row["URL"]
	sentiment = row["Sentiment"]

	pol_score = sia.polarity_scores(line)
	pol_score['headline'] = line
	pol_score['url'] = url
	pol_score['sentiment'] = sentiment
	results.append(pol_score)

# pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)


df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
# print(df.head())

df2 = df[['url', 'headline', 'sentiment', 'label']]

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].headline)[:5], width=200)

print("\nNeutral headlines:\n")
pprint(list(df[df['label'] == 0].headline)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].headline)[:5], width=200)


df.loc[df['label'] == 0, 'label'] = 'Neutral'
df.loc[df['label'] == 1, 'label'] = 'Positive'
df.loc[df['label'] == -1, 'label'] = 'Negative'
df = df[['url', 'headline', 'sentiment', 'label']]
write_file("poc_sentiment.csv",df)