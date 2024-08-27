from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import time
import numpy as np
import pandas as pd
import string
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def train_model(x_train, y_train):
    MODEL = GaussianNB()

    print('Start training...')
    start_time = time.time()
    trained_model = MODEL.fit(x_train, y_train)
    print('Training completed!')
    print(f"Training time: {(time.time()-start_time):.5f}s\n")

    return trained_model


def split_data(x, y):
    VAL_SIZE = 0.2
    TEST_SIZE = 0.1
    SEED = 0

    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=VAL_SIZE,
                                                      shuffle=True,
                                                      random_state=SEED)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=TEST_SIZE,
                                                        shuffle=True,
                                                        random_state=SEED)

    return x_train, x_val, x_test, y_train, y_val, y_test


def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features


def create_dictionary(messages):
    dictionary = []
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)

    return dictionary


def preprocess_text(message):
    translator = str.maketrans('', '', string.punctuation)
    text = message.lower().translate(translator)

    tokens = nltk.word_tokenize(text)

    stop_words = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = nltk.PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def main():
    dataset_path = 'spam_text_classification/csv_files/2cls_spam_text_cls.csv'
    dataset = pd.read_csv(dataset_path)
    messages = dataset['Message'].values.tolist()
    labels = dataset['Category'].values.tolist()

    messages = [preprocess_text(message) for message in messages]

    # dictionary is a string list
    dictionary = create_dictionary(messages)
    pd.DataFrame([dictionary.copy()]).to_csv(
        "spam_text_classification/csv_files/dictionary.csv", header=None, index=None)

    x = np.array([create_features(tokens, dictionary) for tokens in messages])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)         # y is a np.array
    print(f'\nClasses: {label_encoder.classes_}\n')
    pd.DataFrame(label_encoder.classes_).to_csv(
        "spam_text_classification/csv_files/labels.csv", header=None, index=None)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

    # Train and val model
    trained_model = train_model(x_train, y_train)
    val_accuracy = accuracy_score(y_val, trained_model.predict(x_val))
    test_accuracy = accuracy_score(y_test, trained_model.predict(x_test))
    print(f'Val accuracy: {val_accuracy:.5f}')
    print(f'Test accuracy: {test_accuracy:.5f}')
    save_model(trained_model, 'spam_text_classification/trained_model.pkl')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Runtime: {(time.time()-start_time):.5f}s\n")
