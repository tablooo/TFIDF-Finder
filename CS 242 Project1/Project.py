import math
import pandas as pd
import numpy as np
import re


def tokenization(file_name):                                 #remove all characters such as new lines and non alphaneumeric characters

    file = open(file_name, 'r',encoding="utf8")
    book = file.read()
    pattern1 = r'\n'
    book = re.sub(pattern1, ' ', book)
    pattern = r'[^A-Za-z0-9\s]'
    book = re.sub(pattern, '', book)
    book = re.sub(r'\d+', '', book)
    book = book.lower()                                       # turn the words to lowercase, then splitit into a list
    list = book.split(' ')

    dict = count_words(list)                                 #submit that list into a new function that will turn it into a dictionary

    return dict



def count_words(list):                                      #count words counts frequency of each word and puts it in a dictionary

    word_count = {}

    for word in list:
        if word: 
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    return word_count


def makedata(book1,book2,book3,book4):                              #create the dataframe for all the data based on the books

    df = pd.DataFrame({'Evolution in Modern Thought': book1, 'The Confessions of Saint Augustine': book2, 
                'The Imitation of Christ': book3, 'The Story of the Living Machine':book4}).fillna(0).astype(int)

    with open('stopwords.txt', 'r') as file:
        stopwords = [line.strip() for line in file]

    df = df[~df.index.isin(stopwords)]                      #remove stop words

    word_total = df.sum(axis=1)


    df_sorted = df.loc[word_total.sort_values(ascending=False).index]       #sort it based on most used words

    return df_sorted  
def TF(df):                            #calculate TF by getting sum of all words, then dividing it by the amont of time a word was used

    words = df.sum()

    tf = df.div(words, axis=1)

    return tf

def IDF(df):                            #IDF is found by finding how many times a word is in a book, doing 4 divided by that number
                                        # getting the log of that number. This returns a sing column df that just has the IDF of each word
    count = 0
    words = []
    for index, row in df.iterrows():
        for column_name, cell_value in row.iteritems():
            if cell_value > 0:
                count +=1
        words.append(math.log((4/count)))
    
        count = 0
    df['IDF'] = words
    IDF = df['IDF']

    return IDF

def TDIDF(tf, idf):

    result_df = tf.mul(idf, axis=0)          #multiply tf by idf to calculate TDIDF
    return result_df


#code that runs everything, inputing txt file names and also used for testing.

book1 = tokenization('Evolution in Modern Thought.txt')
book2 = tokenization('The Confessions of Saint Augustine.txt')
book3 = tokenization('The Imitation of Christ.txt')
book4 = tokenization('The Story of the Living Machine.txt')
data = makedata(book1,book2,book3,book4)
#print(data)
tf = TF(data)
#print(TF(data))
idf = IDF(data)
tdif = TDIDF(tf,idf)
remove_unique_words = tdif[~(tdif == 0).all(axis=1)]
tdif['Sum'] = tdif.sum(axis=1)
sorted_tdif = tdif.sort_values(by='Sum', ascending=False) #find tdif sorted
#print(sorted_tdif)

print(remove_unique_words.head(40))
