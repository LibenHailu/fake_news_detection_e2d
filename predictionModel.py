# preprocessing
import timeit
from nltk.corpus import wordnet
import _pickle as pickle
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.data.path.append('./nltk_data')

start = timeit.default_timer()
with open("./model_file/model.pkl", 'rb') as f:
    pipeline = pickle.load(f)
    stop = timeit.default_timer()
    print('=> Pickle Loaded in: ', stop - start)


# loading vectorizer
# TODO add timer
vectorizer = pickle.load(open('./model_file/vectorizer.pkl', 'rb'))


class PredictionModel:
    output = {}
    # lemmatizier changes worlds to their root word studing,studied,study -> study
    lemmatizer = WordNetLemmatizer()
    # removes stopwords a, an, the etc
    stopwords = stopwords.words('english')

    # constructor
    def __init__(self, text):
        self.output['original'] = text

    def predict(self):

        self.preprocess()

        test_news = {"text": [self.output['preprocessed']]}
        new_def_test = pd.DataFrame(test_news)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorizer.transform(new_x_test)

        self.output['prediction'] = 'Fake' if pipeline.predict(new_xv_test)[
            0] == 1 else 'Real'

        return self.output['prediction']

    # Helper methods
    def preprocess(self):
        text = self.output['original'].lower()
        row = re.sub('[^a-zA-Z]', ' ', text)
        token = row.split()
        news = [self.lemmatizer.lemmatize(word)
                for word in token if not word in self.stopwords]

        self.output['preprocessed'] = ' '.join(news)
