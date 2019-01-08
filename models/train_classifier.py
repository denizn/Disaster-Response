import sys
import pandas as pd
from sqlalchemy import create_engine

import re

import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support,classification_report,f1_score
from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet');

import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Table', engine, index_col='id')
    X = df['message']
    y = df.drop(['message', 'original', 'genre'], axis=1)
    return X, y, y.columns
    
def tokenize(text):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]"," ", text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens]
    return tokens

def build_model():
    TFIDF = TfidfVectorizer(stop_words=None, tokenizer = tokenize, lowercase=True)
    clf = MultiOutputClassifier(LinearSVC(C=0.5), n_jobs=-1)
    pipeline = Pipeline([
        ('tfidf', TFIDF),
        ('clf', clf)])
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    yhat_test = model.predict(X_test)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, yhat_test, average='micro')
    print('Precision: {}\nRecall: {}\nFscore: {}'.format(precision,recall,fscore))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()