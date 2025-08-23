import pandas as pd
import numpy as np
import re
import nltk
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')


"""DATA COLLECTION"""
unlabelled_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', quoting=3)
unlabelled_df = pd.DataFrame(unlabelled_train)
unlabelled_df = unlabelled_df.drop(columns=['id'])

labeled_df = pd.read_csv("labeledTrainData.tsv", sep="\t")

test_df = pd.read_csv('testData.tsv', sep='\t')

"""PRE PROCESSING"""
def preprocessing(df): #without lemmetization for word2vec

    # Lowercasing and removing special characters
    df['review'] = df['review'].str.lower().apply(lambda x: re.sub(r'[^a-z]', ' ', x))

    #tokenization
    def tokenization(text):
        tokens = re.split(r'\W+', text)
        return [t for t in tokens if t] #removes empty tokens

    df['tokenized'] = df['review'].apply(lambda x: tokenization(x))

    #stopwords removal
    stop_words = nltk.corpus.stopwords.words('english')
    df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
    return df


unlabelled_df = preprocessing(unlabelled_df)
print(unlabelled_df.head())


"""FEATURE EXTRACTION USING Word2Vec"""
w2v_model = Word2Vec(
    sentences=unlabelled_df['tokenized'],
    vector_size=300,   # embedding dimension
    window=5,          # context window
    min_count=5,       # ignore rare words
    workers=4          # number of CPU threads
)

w2v_model.save("word2vec_unlabeled.model")
# print("Most similar to 'movie':")
# print(w2v_model.wv.most_similar("movie", topn=5))


def review_to_vector(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)       
    #Word2Vec gives vectors for words.But classifier needs a single vector per review.Reviews have variable length (10 words, 200 words, 1000 words).
    #So we need a way to turn all those word vectors into one fixed-size vector.(No matter how long the review, the average is always a 300-dim vector (if embeddings are 300-dim).)

labeled_df = preprocessing(labeled_df)
X = np.array([review_to_vector(tokens, w2v_model) for tokens in labeled_df["tokenized"].values]) 
y = labeled_df["sentiment"].values


"""MODEL SELECTION"""

def fnc_classification_all_model(x,y):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
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
    s=LinearSVC()
    k=KNeighborsClassifier()
    D=DecisionTreeClassifier()
    R=RandomForestClassifier()
    Log=LogisticRegression()
    XGB=XGBClassifier()
    G=GradientBoostingClassifier()

    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x,y,random_state=42)


    algos=[b,s,k,D,R,Log,XGB,G]
    algo_names=['BernoulliNB','LinearSVC','KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier','LogisticRegression','XGBClassifier','GradientBoostingClassifier']

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

print(fnc_classification_all_model(X,y))


"""MODEL TRAINING"""
from sklearn.svm import LinearSVC


model = LinearSVC()
model.fit(X, y)


"""PREDICTION"""
test_df = preprocessing(test_df) 
X_test = np.array([review_to_vector(tokens, w2v_model) for tokens in test_df['tokenized'].values])

y_pred = model.predict(X_test)

# submission = pd.DataFrame({
#     "id": test_df["id"],
#     "sentiment": y_pred})
# submission.to_csv("submission.csv", index=False)