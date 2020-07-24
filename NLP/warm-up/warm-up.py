import nltk
import itertools

# importing the inaugural corpus
from nltk.corpus import inaugural

# importing stopwords
from nltk.corpus import stopwords

# importing Frequency distribution
from nltk.probability import FreqDist

words = inaugural.words('1993-Clinton.txt')
speech = inaugural.raw('1993-Clinton.txt')
stop_words = set(stopwords.words('english'))

# listing all the documents in the inaugural corpus
# print(inaugural.fileids())

# the number of words in clintons 1993 speech
print('Total number of words in the given text is:',
      len(inaugural.words('1993-Clinton.txt')))

# unique words in the speech
unique_words = sorted(set(inaugural.words('1993-Clinton.txt')))

print('Number of unique words in the given text is:', len(unique_words))


# function that returns the avg length of words in a word list.
def avg_word_length(word_list):
    word_lengths = [len(w) for w in word_list]
    total_length = 0

    for w in word_lengths:
        total_length += w

    avg_length = total_length/len(words)
    return avg_length


print('The average length of a word in the given text is:', avg_word_length(words))


# creating a list of all the lower case words in the speech
lower_case_words = [word for word in words if word.islower()]

# creating a frequency distribution object of the lower case words
fdist = FreqDist(lower_case_words)

# prints out the words and their frequency in descending order.
# print(fdist.most_common())


# Created a function that displays the frequency distribution along with the ranks by descending order of frequency. The first integer represents the rank, the string is the word itself and the last integer is the frequency of the word.
def ranked_freq_dist(dist):

    # Using the 10 most common words for better readabilty, If you want a freqiency distribution of all the words remove the 10.
    f_list = dist.most_common(10)
    f_list2 = f_list[1:]
    rank = 1
    ranked_list = []

    for x, y in itertools.zip_longest(f_list, f_list2, fillvalue=(0, 0)):

        rank_tup = tuple(['Rank:'+str(rank)])
        new_element = tuple([rank_tup+x])

        if x[1] > y[1]:
            ranked_list += new_element
            rank += 1

        elif x[1] == y[1]:
            ranked_list += new_element

    return ranked_list


print('Ranked frequency distribution in descending order of frequency',
      ranked_freq_dist(fdist))


# Removing stop words and punctuations from the words list.
filtered_words = [
    w for w in words if not w in stop_words and w.isalnum()]

# Creating a FreqDist object from the filtered list.
filtered_fdist = FreqDist(filtered_words)

# Plotting the top 10 words. I didnt plot the top 50 as the graph gets very hard to read on a small screen.
#filtered_fdist.plot(10, title='Frequency Distribution')

# Checking the number of occurences of the words 'America' and 'world'.
print('Occurences of \'America\':', filtered_fdist.pop(
    'America'), '\nOccurences of \'world\':', filtered_fdist.pop('world'))
