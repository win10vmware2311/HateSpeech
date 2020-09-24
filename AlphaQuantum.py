import pandas as pd
import joblib
import re
import sklearn
from IPython.display import display
from sklearn import svm
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le = preprocessing.LabelEncoder()
from sklearn.model_selection import learning_curve
#from sklearn.externals import joblib

def clean_tweets(df):
    df['Tweets']=df['Tweets'].str.lower()
    df['Tweets']=df['Tweets'].apply(lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
    return df
 
# preprocess data
df=pd.read_csv('/Users/Desktop/CompiledTweets2.csv',encoding='ISO-8859-1')
df.columns=['Labels','Tweets']
# del df['Id'], df['Date'], df['Query'], df['User']
df['Labels']=df['Labels'].map({1:'Negative',0:'Positive'})
df = df.sample(frac=0.1).reset_index(drop=True)
df=clean_tweets(df)

# preparing training, test data
X=df['Tweets'].to_list()
y=df['Labels'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=10)
Vectorizer = TfidfVectorizer(max_df=0.9,ngram_range=(1, 2))
TfIdf=Vectorizer.fit(X_train)
X_train=TfIdf.transform(X_train)

# training the model
model =sklearn.svm.LinearSVC(C=0.1)
model.fit(X_train,y_train)

# evaluation
X_test=TfIdf.transform(X_test)
y_pred=model.predict(X_test)
y_test=le.transform(y_test)
print(classification_report(y_test, y_pred))