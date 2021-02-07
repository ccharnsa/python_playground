# https://stackoverflow.com/questions/40325980/how-is-the-vader-compound-polarity-score-calculated-in-python-nltk
# https://stackoverflow.com/questions/51514208/edit-vader-lexicon-txt-in-nltk-for-python-to-add-words-related-to-my-domain
# https://github.com/cjhutto/vaderSentiment

# Improve
# https://stackoverflow.com/questions/45296897/is-there-a-way-to-improve-performance-of-nltk-sentiment-vader-sentiment-analyser

# Dict
# https://www.kdnuggets.com/2012/05/provalis-sentiment-analysis-financial-political-general-dictionary.html

import nltk
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# from nltk import tokenize
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def get_sentiment(paragraph):
	sentence_list = nltk.tokenize.sent_tokenize(paragraph)
	paragraphSentiments = 0.0
	for sentence in sentence_list:
		vs = Analyzer.polarity_scores(sentence)
		print("{:-<69} {}".format(sentence, str(vs["compound"])))
		paragraphSentiments += vs["compound"]

	print("")
	print("Average sentiment: \t" + str(round(paragraphSentiments / len(sentence_list), 4)))
	print("############")

	cp = round(paragraphSentiments / len(sentence_list), 4)

	# tokenized_sentence = nltk.word_tokenize(sentence)
	# pos_word_list=[]
	# neu_word_list=[]
	# neg_word_list=[]

	# for word in tokenized_sentence:
	#     if (Analyzer.polarity_scores(word)['compound']) >= 0.05:
	#         pos_word_list.append(word)
	#     elif (Analyzer.polarity_scores(word)['compound']) <= -0.05:
	#         neg_word_list.append(word)
	#     else:
	#         neu_word_list.append(word)                

	# print('Positive:',pos_word_list)
	# print('Neutral:',neu_word_list)
	# print('Negative:',neg_word_list) 
	# score = Analyzer.polarity_scores(sentence)
	# print('Scores:', score)
	# cp = score['compound']
	# print("############")

	pred_sentiment = ""
	if cp > 0.2:
		pred_sentiment = "Positive"
	elif cp < -0.2:
		pred_sentiment = "Negative"
	else:
		pred_sentiment = "Neutral"
	return pred_sentiment

########################################################################################################################

additional_lexicon = {
	'headset': 1.0
}

# add new dict
Analyzer = SentimentIntensityAnalyzer()
Analyzer.lexicon.update(additional_lexicon)

# remove
# .lexicon.pop('no')

paragraph = '''
Today there is good and bad news about the company new Z5 just release
'''

print(get_sentiment(paragraph))