import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - filepath of the csv-file with the messages
    categories_filepath - filepath of the csv-file with the categories
    
    OUTPUT
    df - combined dataframe of messages and categories
    
    This function creates a combined dataframe of messages and categories
    '''
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)  
    df = pd.merge(messages,categories,on='id',how='outer')
    
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(";",expand=True)
    row = list(categories.iloc[0])
    category_colnames = [x[:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace categories column in df with new category columns.        
    df = df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories.reset_index(drop=True)], axis=1)
    return df


def clean_data(df):
    '''
    INPUT
    df - combined dataframe of messages and categories
    
    OUTPUT
    df - cleaned dataframe
    
    This function removes duplicates of the dataframe
    '''
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    INPUT
    df - dataframe
    database_filename - filename where the dataframe should be stored
    
    OUTPUT
    None
    
    This function stores the dataframe as sqlite file
    '''
    dbfile = 'sqlite:///' + database_filename 
    engine = create_engine(dbfile)
    df.to_sql(database_filename.replace(".db", "").replace("data/", ""), engine, index=False)
    


def main():
    '''
    INPUT
    None
    
    OUTPUT
    None
    
    This function runs the creation and storage of a dataset which can be used for the training of machine learning algorithms
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()