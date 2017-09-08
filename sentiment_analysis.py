import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

@staticmethod
def __init__(self):
    pass

@staticmethod
def event():
    pass

def render(wf_module, table):
    if table == None:
        return None

    column = wf_module.get_param_string('textcolumn')
    if column == '' or column == None:
        wf_module.set_ready(notify=False)
        return None

    if column not in table.columns:
        wf_module.set_error("No column {} exists.".format(column))
        return None

    data = table[column]
    sentiment = []

    sid = SentimentIntensityAnalyzer()

    for text in data:
        score = sid.polarity_scores(text)
        sentiment.append(score['compound'])

    # add column to existing table
    table = table.assign(sentiment=pd.Series(sentiment))
    return table
