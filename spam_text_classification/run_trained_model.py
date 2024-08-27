from sklearn.preprocessing import LabelEncoder

import time
import numpy as np
import pandas as pd
import string
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')


def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features


def preprocess_text(message):
    translator = str.maketrans('', '', string.punctuation)
    text = message.lower().translate(translator)

    tokens = nltk.word_tokenize(text)

    stop_words = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = nltk.PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def predict(query_input, model, dictionary, label_encoder):
    tokens = preprocess_text(query_input)
    features = create_features(tokens, dictionary).reshape(1, -1)
    prediction = model.predict(features)
    prediction_class = label_encoder.inverse_transform(prediction)[0]

    return prediction_class


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def main():
    dictionary = pd.read_csv(
        'spam_text_classification/csv_files/dictionary.csv', header=None).iloc[0].tolist()

    labels = pd.read_csv(
        'spam_text_classification/csv_files/labels.csv', header=None).iloc[:, 0].to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    print(f'\nClass {y[0]} is {label_encoder.classes_[0]}')
    print(f'Class {y[1]} is {label_encoder.classes_[1]}\n')

    trained_model = load_model('spam_text_classification/trained_model.pkl')

    # Use model to predict message is spam or ham
    query_input = 'Congatulations! you just win a prize for 1000th customer of FA website'  # spam
    # ham: 'I am actually thinking a way of doing something useful'
    print(f"Message: {query_input}")
    model_output = predict(
        query_input, trained_model, dictionary, label_encoder)
    print(f'Prediction: {model_output}')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Runtime: {(time.time()-start_time):.5f}s\n")
