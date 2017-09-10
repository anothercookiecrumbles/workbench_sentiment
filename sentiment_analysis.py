from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd

class SentimentAnalyser:

    @staticmethod
    def __init__(self):
        pass

    @staticmethod
    def event():
        pass

    @staticmethod
    def render(wf_module, table):

        column = wf_module.get_param_column('column_name')
        if column == '' or column == None:
            wf_module.set_ready(notify=False)
            return None

        if column not in table.columns:
            wf_module.set_error("No column {} exists.".format(column))
            return None

        all_texts = table[column]
        sentiment = []

        sid = SentimentIntensityAnalyzer()

        for text in all_texts:
            if type(text) != str:
                text = str(text)
            score = sid.polarity_scores(text)
            sentiment.append(score['compound'])

        # add column to existing table
        table = table.assign(sentiment=pd.Series(np.asarray(sentiment, dtype=np.float32)))
        return table
