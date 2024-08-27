# Spam Text Classification Project

This project implements a spam text classification model using a Naive Bayes classifier. The model is trained on a dataset of text messages labeled as spam or not spam. This project includes 2 files "train_model.py" and "run_model.py". "train_model.py" is file that trains, evaluates, tests model and then saves traind model to "trained_model.pkl".  "run_model.py" is file that runs trained model.

## --------------------------------------------------------------

# Training Model

## Table of Contents
- Installation
- Usage
- Project Structure
- Model Training
- Evaluation
- Saving the Model

## Installation

1. Clone the repository:
    ```bash
    https://github.com/Honghai2102/email_classification_project.git
    cd spam_text_classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your dataset in the root directory of the project. The dataset should be a CSV file named `2cls_spam_text_cls.csv` with two columns: `Message` and `Category`.

2. Run the `train_model.py` script to train the model:
    ```bash
    python train_model.py
    ```

## Project Structure

- `train_model.py`: Main script for training and evaluating the model.
- `2cls_spam_text_cls.csv`: Dataset file containing text messages and their labels.
- `dictionary.csv`: File containing the dictionary of unique tokens.
- `labels.csv`: File containing the label classes.
- `trained_model.pkl`: File where the trained model is saved.

## Model Training

The model training process involves the following steps:

1. **Preprocessing Text**: 
    - Convert text to lowercase.
    - Remove punctuation.
    - Tokenize text.
    - Remove stopwords.
    - Apply stemming.

2. **Creating Dictionary**: 
    - Create a dictionary of unique tokens from the preprocessed text.

3. **Feature Extraction**: 
    - Convert tokens into feature vectors based on the dictionary.

4. **Splitting Data**: 
    - Split the data into training, validation, and test sets.

5. **Training the Model**: 
    - Train a Naive Bayes classifier on the training data.

## Evaluation

The model is evaluated on both the validation and test sets. The accuracy scores are printed after training.

## Saving the Model

The trained model is saved to a file named `trained_model.pkl` using the `pickle` module.

## --------------------------------------------------------------


# Running Trained Model

## Table of Contents
- Installation
- Usage
- Project Structure
- Prediction
- Acknowledgements

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Chow05/Spam_Text_Classification_Project.git
    cd spam_text_classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the following files in the root directory:
    - `dictionary.csv`: File containing the dictionary of unique tokens.
    - `labels.csv`: File containing the label classes.
    - `trained_model.pkl`: File where the trained model is saved.

2. Run the `run_model.py` script to predict whether a message is spam or not:
    ```bash
    python run_model.py
    ```

## Project Structure

- `run_model.py`: Main script for loading the model and predicting the class of a given text message.
![pic/query_example.png](pic/query_example.png)
- `dictionary.csv`: File containing the dictionary of unique tokens.
- `labels.csv`: File containing the label classes.
- `trained_model.pkl`: File where the trained model is saved.

## Prediction

The prediction process involves the following steps:

1. **Preprocessing Text**: 
    - Convert text to lowercase.
    - Remove punctuation.
    - Tokenize text.
    - Remove stopwords.
    - Apply stemming.

2. **Feature Extraction**: 
    - Convert tokens into feature vectors based on the dictionary.

3. **Loading the Model**: 
    - Load the pre-trained Naive Bayes classifier from the `trained_model.pkl` file.

4. **Predicting the Class**: 
    - Use the model to predict whether the input message is spam or not.

## Acknowledgements

This project uses the following libraries:
- `scikit-learn`
- `pandas`
- `numpy`
- `nltk`

Special thanks to the open-source community for providing these tools.
