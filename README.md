# Disaster Response Pipeline Project

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `cd app/`
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Motivation for the project

The aim of this analysis as part of the Udacity Data Science Nanodegree was to enable users to use a web interface to categorize text in the context of disasters. 


## Summary of the results

It was possible to predict the categories using a Random Forest algorithm.

## Libraries used

The following python libraries are used in this project:

- sys
- pandas
- sqlalchemy
- nltk
- numpy
- sklearn
- pickle
- json
- plotly
- flask
- joblib


## Files in the repository

- **./data/disaster_categories.csv**: Categories as the aim variables
- **./data/disaster_messages.csv**: Messages written during disasters as the co variable
- **./data/process_data.py**: Code to clean and combine the csv files and export them as db file
- **./data/DisasterResponse.db**: Result of the *process_data.py* file
- **./models/train_classifier.py**: Code to create the ML model based on *DisasterResponse.db*
- **./models/classifier.pkl**: Result of the *train_classifier.py* file
- **./app/run.py**: Code of the flask app to create the web interface
- **./app/templates/master.html**: HTML component of the flask app
- **./app/templates/go.html**: HTML component of the flask app


## Acknowledgements

- The [Udacity Data Science Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)


