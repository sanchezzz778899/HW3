from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from connector.pg_connector import get_data
from conf.conf import logging
from sklearn.linear_model import LogisticRegression
from conf.conf import settings
import pickle
from util.util import save_model, load_model

settings.load_file(path="/path/to/file/.toml")

logging.info("Extracting dataset")
df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')
logging.info("DF extracted")

def split(df): 

    logging.info('Defining x and y')
    # variables
    X = df.iloc[:, :-1] 
    y = df['target']
    logging.info('Splitting data')
# Split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, #independent variables
                                                    y, #dependent variable
                                                    random_state = 3
                                                   )
    return X_train, X_test, y_train, y_test

def train_log_regression(X_train, y_train):
    # Initialize the model
    clf = LogisticRegression(max_depth = 3,
                                random_state = 3
                                )
    logging.info('Training the model')
    # Train the model
    clf.fit(X_train, y_train)

    save_model(dir='model/conf/log_regression.pkl', model=clf)

    return clf 

def predict(values, path_to_model):
    clf = load_model(path_to_model)
    return clf.predict(values)

df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')
X_train, X_test, y_train, y_test= split(df)
clf = split(X_train, y_train)

logging.info(f'Accuracy is {clf.score(X_test, y_test)}')

responce = predict(X_test)
clf = load_model('model/conf/log_regression.pkl')

logging.info(f'Prediction is is {clf.predict(X_test)}')
