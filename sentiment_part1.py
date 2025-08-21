import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

"""COLLECTING DATA"""
train_df = pd.read_csv('labeledTrainData.tsv', sep='\t')
print(train_df.isnull().sum())
train_df = train_df.drop(columns=['id'])

value_counts = train_df['sentiment'].value_counts()
print(value_counts) # data is balanced

test_df = pd.read_csv('testData.tsv', sep='\t')
test_df = test_df.drop(columns=['id'])

# plt.figure(figsize=(8, 6))
# plt.bar(value_counts.index, value_counts.values)
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.title('Sentiment Distribution')
# plt.xticks(rotation=0)
# plt.show()


"""PRE PROCESSING"""
def pre_processing(df):
    
    # Lowercasing and removing special characters
    df['review'] = df['review'].str.lower().apply(lambda x: re.sub(r'[^a-z]', ' ', x))
    """ In above line, After converting to lowercase using str.lower(), it applies a lambda function to each row in review using the .apply()
    method. The lambda function uses re.sub(r'[^a-z]', ' ', x) to replace any character that is not a lowercase letter (a-z) with a space. 
    This regular expression ensures that numbers and special characters are removed."""

    #tokenization
    def tokenization(text):
        tokens = re.split(r'\W+', text)
        return [t for t in tokens if t] #removes empty tokens

    df['tokenized'] = df['review'].apply(lambda x: tokenization(x))

    #stopwords removal
    stop_words = nltk.corpus.stopwords.words('english')
    df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])

    # #stemming
    # ps = nltk.PorterStemmer()
    # def stemming(text):
    #     text = [ps.stem(word) for word in text]
    #     return text

    # df['stemmed'] = df['tokenized'].apply(lambda x: stemming(x))

    #lemmatization (more accurate but requires PoS tagging first)
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # Lemmatization with POS tagging applied to the dataset
    lemmatizer = WordNetLemmatizer()
    def lemmatize_tokens(tokens):
        tagged = pos_tag(tokens)
        return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]

    df['lemmatized'] = df['tokenized'].apply(lemmatize_tokens)
    print(df[['review', 'tokenized', 'lemmatized']].head())