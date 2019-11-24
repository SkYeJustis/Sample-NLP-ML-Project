import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

"""
Reference:
    (Method) https://medium.com/@chrisfotache/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0
    (Data) https://www.kaggle.com/c/gendered-pronoun-resolution/data 
"""

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

def tokenize_and_stem(text):
    # Tokenize by sentence, then by word
    tokens = [word for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]

    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]

    # Stem the filtered_tokens
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(token) for token in filtered_tokens]

    return stems

def get_gender(word):
    if word.lower() in ['she', 'her', 'hers']:
        return 1
    else:
        return 0

def get_word_count(text):
    return len(text.split(' '))


if __name__ == '__main__':
    data = pd.read_csv('files/test_stage_1.tsv', delimiter='\t')

    print(data.head())
    print(data.columns)

    print(data['Pronoun'].str.lower().unique())

    # ['her' 'his' 'she' 'him' 'he']
    data['Gender'] = data['Pronoun'].apply(get_gender)
    data['TotalWords'] = data['Text'].apply(get_word_count)

    X = data[['Text', 'TotalWords']]
    Y = data['Gender']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


    classifier = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('colext', TextSelector('Text')),
                ('tfidf', TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem,
                                       ngram_range=(1, 3))),
            ])),
            ('words', Pipeline([
                ('wordext', NumberSelector('TotalWords')),
                ('wscaler', StandardScaler()),
            ])),
        ])),
        ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)),
    ])

    print('Fitting classifier...')
    classifier.fit(X_train, y_train)
    print('Making predictions...')
    preds = classifier.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))