import pandas as pd 
from conf.conf import logging

def get_data(link: str) -> pd.DataFrame:
    logging.info("DF extracted")
    df = pd.read_csv('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')
    logging.info("DF extracted")
    return df