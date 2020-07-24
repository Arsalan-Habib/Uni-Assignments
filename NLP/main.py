import pandas as pd
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist

# loading the tweets csv as a pd dataframe
df = pd.read_csv('./tweets.csv', encoding='unicode_escape', header=None)

# saving all the tweets as a list.
tweets_list = list(df[1])

# Tokenizing the tweets
tweets_words_list = word_tokenize("".join(tweets_list))

# Converting the text to lowercase
tweets_words_list = [w.lower() for w in tweets_words_list]

# creating a stop-words-list
stop_words = set(stopwords.words('english'))

# Creating a list of punctuations to remove
punct = [x for x in string.punctuation if x not in '!?.']

unique_words = set(tweets_words_list)

# A list of slang words to remove from the text.
slangs = set(['hi', 'dm', 'zarye', 'ada', 'maslay', 'kejye', 'ga', 'kar', 'pe', 'se', 'ka', 'kerne', 'rehnumaye', 'raha', 'kejiye', 'maujood', 'gaye', 'hoga', 'kerna', 'ilzaam', 'jaye', 'mein', 'atwaar', 'kiye', 'rahe', 'shukriya', 'karwany', 'elaqoon', 'liye', 'parha', 'sakty', 'tafseelat', 'kiya', 'pr', 'leye', 'izafay', 'sakein', 'aamawam', 'letay', 'apko', 'ap', 'parishani', 'shamil', 'hain'
              'humein', 'bhejain', 'ki', 'hai', 'lay', 'bahtar', 'saky', 'sharafat', 'karna', 'nahi', 'han', 'kijiye', 'rahnumai', 'farmaye', 'ker', 'yeh', 'sey', 'karnay', 'kay', 'apkay', 'sath', 'takay', 'haasil', 'apni', 'hongy', 'kijyey', 'nahe', 'jana', 'ky', 'hasil', 'aur', 'jald', 'hein', 'kitnay', 'tak', 'ke', 'rahnumai', 'gaya', 'hum', 'humay', 'apne', 'rehnumai', 'apna', 'masla'])

# creating a regex pattern to match symbols and emoticons
pattern = r'\bu\+\w+'
# Creating a list of all the emojis to remove from the text
emoticons = re.findall(pattern, ' '.join(unique_words))

# Creating a new list by filtering out the old text.
filtered_words_list = [
    w for w in tweets_words_list if w not in slangs and w not in emoticons and w not in stop_words and w not in punct and not w.isnumeric() and not w[0].isdigit() and not w.startswith('//') and not w.startswith('http')]
# print(len(filtered_words_list), filtered_words_list)

# Initializing a porter stemmer object
ps = PorterStemmer()

# creating a new list with all the stemmed words
stemmed_words_list = [ps.stem(w) for w in filtered_words_list]
# print(stemmed_words_list)

# Creating a freq dist object from the stemmed words list but without punctuations or usernames.
f_dist = FreqDist([w for w in stemmed_words_list if w.isalnum()])

# Plotting the 20 most common words
f_dist.plot(20, title='Zong Tweets Freq Distribution')

print(f_dist.most_common(20))
