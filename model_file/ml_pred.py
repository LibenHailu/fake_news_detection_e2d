from operator import mod
import re

from numpy import vectorize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stopwords = stopwords.words('english')

nltk.download('wordnet')


def clean_data(text):
    text = text.lower()
    row = re.sub('[^a-zA-Z]', ' ', text)
    token = row.split()
    news = [lemmatizer.lemmatize(word)
            for word in token if not word in stopwords]
    clean_news = ' '.join(news)

    return clean_news
    # return news


def predict_news(model, text):
    voc = clean_data(text).split()
    vectorizer = TfidfVectorizer(
        stop_words='english', max_features=50000, lowercase=False, ngram_range=(1, 2))
    vectorizer.fit(voc)
    vec_val = vectorizer.transform(voc)

    pred = model.predict(vec_val)

    return 'GENIUNE NEWS' if pred == 1 else 'FAKE NEWS'
