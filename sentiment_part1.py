import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

"""COLLECTING DATA"""
train_df = pd.read_csv('labeledTrainData.tsv', sep='\t')
print(train_df.isnull().sum())
train_df = train_df.drop(columns=['id'])

value_counts = train_df['sentiment'].value_counts()
print(value_counts) # data is balanced
# print(train_df.shape)
# plt.figure(figsize=(8, 6))
# plt.bar(value_counts.index, value_counts.values)
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.title('Sentiment Distribution')
# plt.xticks(rotation=0)
# plt.show()

test_df = pd.read_csv('testData.tsv', sep='\t')
test_df = test_df.drop(columns=['id'])

sample_submission = pd.read_csv('sampleSubmission.csv')


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
    #print(df[['review', 'tokenized', 'lemmatized']].head())
    return df
train_df = pre_processing(train_df)
test_df = pre_processing(test_df)



"""FEATURE EXTRACTION """

# Join lemmatized tokens back into strings
train_df['lemmatized_text'] = train_df['lemmatized'].apply(lambda tokens: ' '.join(tokens))

vectorizer = TfidfVectorizer(max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)
x_train = vectorizer.fit_transform(train_df['lemmatized_text'])
y_train = train_df['sentiment']
#stopwords removal done again for robustness (just in case there are any issues or missed stop words previously)
#also in TfidfVectorizer, stopwords removal are specific to vectorization



"""MODEL SELECTION"""

def fnc_classification_all_model(x,y):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    from sklearn.metrics import confusion_matrix,classification_report

    b=BernoulliNB()
    m=MultinomialNB()
    s=SVC()
    k=KNeighborsClassifier()
    D=DecisionTreeClassifier()
    R=RandomForestClassifier()
    Log=LogisticRegression()
    XGB=XGBClassifier()
    G=GradientBoostingClassifier()

    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x,y,random_state=42)


    algos=[b,m,s,k,D,R,Log,XGB,G]
    algo_names=['BernoulliNB','MultinomialNB','SVC','KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier','LogisticRegression','XGBClassifier','GradientBoostingClassifier']

    accuracy_scored=[]
    precision_scored=[]
    recall_scored=[]
    f1_scored=[]


    for item in algos:
        print(item)

        predict=item.fit(x_train_split,y_train_split).predict(x_val_split)

        accuracy_scored.append(accuracy_score(y_val_split,predict))
        precision_scored.append(precision_score(y_val_split,predict,average='macro'))
        recall_scored.append(recall_score(y_val_split,predict,average='macro'))
        f1_scored.append(f1_score(y_val_split,predict,average='macro'))

    result=pd.DataFrame(columns=['accuracy_score','f1_score','recall_score','precision_score'],index=algo_names)
    result['accuracy_score']=accuracy_scored
    result['f1_score']=f1_scored
    result['recall_score']=recall_scored
    result['precision_score']=precision_scored

    return result.sort_values('accuracy_score',ascending=False)

print(fnc_classification_all_model(x_train,y_train))


"""MODEL TRAINING"""

from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)
test_df['lemmatized_text'] = test_df['lemmatized'].apply(lambda tokens: ' '.join(tokens))
vectorizer = TfidfVectorizer(max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)
x_test = vectorizer.fit_transform(test_df['lemmatized_text'])
y_pred = model.predict(x_test)

submission=pd.DataFrame()
submission["id"]=sample_submission["id"]
submission["sentiment"]=y_pred.astype(int)
submission.head()