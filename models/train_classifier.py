import sys
import nltk
nltk.download('wordnet')
nltk.download('punkt')
import pandas as pd
from sqlalchemy.engine import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    INPUT
    database_filepath - path of database with cleaned data
    
    OUTPUT
    X - array of messages to be used as covariable
    Y - array of aim variables 
    column_names - list of column names for the aim variables
    
    This function splits the dataset in aim and covariables
    '''
    dbfile = 'sqlite:///' + database_filepath 
    engine = create_engine(dbfile)
    df = pd.read_sql(database_filepath.replace(".db", "").replace("data/", ""), engine)
    X = df.message.values
    Y = df.drop(columns=['id','message']).values.astype(str)
    column_names = df.drop(columns=['id','message']).columns
    return X, Y, column_names


def tokenize(text):
    '''
    INPUT
    text - text to be tokenized
    
    OUTPUT
    clean_tokens - tokenized text
    
    This function tokenizes and lemmatizes the text input
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT
    None
    
    OUTPUT
    pipeline - machine learning pipeline
    
    This function vectorizes and transforms (TFID) the covariable and trains a multi-output Random Forest model 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__max_features': ('auto',5,10,20,30),
        'clf__estimator__max_leaf_nodes': (None, 5,10,20)
    }
    
    pipeline = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - machine learning model
    X_test - co-variable of the test dataset
    Y_test - aim variable of the test dataset
    category_names - names of the aim variables
    
    OUTPUT
    None
    
    This function uses the machine learning model to predict based on the co-variable of the test dataset and compares this with the aim variable of the test dataset 
    '''
    predicted = model.predict(X_test)
    y_true = pd.DataFrame(Y_test,columns=category_names)
    y_pred = pd.DataFrame(predicted,columns=category_names)
    
    for colnam in y_true.columns:
        print("Test of " + colnam + ":")
        print(classification_report(y_true[colnam], y_pred[colnam]))


def save_model(model, model_filepath):
    '''
    INPUT
    model - machine learning model
    model_filepath - path to the file the model should be stored at
    
    OUTPUT
    None
    
    This function stores the machine learning model at the given filepath
    '''
    outfile = open(model_filepath,'wb')
    pickle.dump(model,outfile)
    outfile.close()


def main():
    '''
    INPUT
    None
    
    OUTPUT
    None
    
    This function runs the creation and storage of a machine learning model
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
