import re
import nltk
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# CNN libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import os
from numpy import load
from PIL import Image

import warnings

warnings.filterwarnings("ignore")


nltk.download("all")
nltk.download("words")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

tqdm.pandas()


# Run model on 5-fold CV
def run_model(model, x_train, x_test, y_train, y_test, kf_splits=5):

    best_model = None
    best_train_acc = 0
    best_val_acc = 0

    train_acc_list, val_acc_list = [], []

    kf = KFold(n_splits=kf_splits)

    for train, validation in tqdm(kf.split(x_train)):
        model.fit(x_train[train], y_train[train])

        y_val_pred = model.predict(x_train[validation])
        validation_acc = accuracy_score(y_train[validation], y_val_pred)
        val_acc_list.append(validation_acc)

        y_train_pred = model.predict(x_train[train])
        train_acc = accuracy_score(y_train[train], y_train_pred)
        train_acc_list.append(train_acc)

        if train_acc > best_train_acc and validation_acc > best_val_acc:
            best_train_acc = train_acc
            best_val_acc = validation_acc
            best_model = model

    print("Train classification report")
    print(classification_report(best_model.predict(x_train), y_train))
    print()

    print("Test classification report")
    print(classification_report(best_model.predict(x_test), y_test))

    return best_model, train_acc_list, val_acc_list


if __name__ == "__main__":
    df = pd.read_csv("All Text-Image Cleaned data.csv")
    df.head()

    # Load Tfidf vectorizer
    tfidfvectorizer = pickle.load(open("tfidfvec.pkl", "rb"))

    # Mapping categories to numbers
    Label_List = ["food", "sports", "travel", "technology", "science"]
    df["target"] = df["subreddit"].progress_apply(lambda x: Label_List.index(x))

    # Separate text and target
    X = df["reddit_submissions_title"]
    y = df["target"]

    # Get text features using the saved vectorizer
    threshold = 10
    print("Rare word threshold", threshold)
    X_vect = tfidfvectorizer.transform(X)
    vocab = tfidfvectorizer.get_feature_names_out()
    print("Vocab size : ", len(vocab))

    # Randomly sample training indices
    train_indexes = np.random.choice(len(df), int(0.8 * len(df)), replace=False)
    print(train_indexes, len(train_indexes))

    # Test indices
    test_indexes = np.arange(0, len(df)).tolist()
    for i in range(len(df)):
        if i in train_indexes:
            test_indexes.remove(i)

        len(test_indexes)

    # Separate text data into train and test sets
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in train_indexes:
        x_train.append(X_vect[i])
        y_train.append(y[i])

    for i in test_indexes:
        x_test.append(X_vect[i])
        y_test.append(y[i])

    # Get training and test image files
    train_fn = df["IMG_NAME"].iloc[train_indexes]
    test_fn = df["IMG_NAME"].iloc[test_indexes]

    train_files = train_fn.values
    test_files = test_fn.values
    print(train_files, len(train_files))
    print(test_files, len(test_files))

    # Separate image data into train and test sets
    Train_source = "Merged"
    label_List = ["food", "sports", "travel", "technology", "science"]
    x_train_img = []
    y_train_img = []

    x_test_img = []
    y_test_img = []

    X_img = []
    y_img = []
    Dimension = 32

    print("Processing Train Files")
    for f in tqdm(train_files):
        f += ".jpg"
        im = Image.open(Train_source + "/" + f)
        new_im = im.resize((Dimension, Dimension))
        rgb = np.array(new_im.convert("RGB"))
        x_train_img.append(rgb)
        label = []
        label.append(Label_List.index(f.split("_")[0]))
        label = np.array(label)
        y_train_img.append(label)

    for f in tqdm(test_files):
        f += ".jpg"
        im = Image.open(Train_source + "/" + f)
        new_im = im.resize((Dimension, Dimension))
        rgb = np.array(new_im.convert("RGB"))
        x_test_img.append(rgb)
        label = []
        label.append(Label_List.index(f.split("_")[0]))
        label = np.array(label)
        y_test_img.append(label)

    # Converting list to numpy array
    x_train_img = np.array(x_train_img)
    x_test_img = np.array(x_test_img)
    y_train_img = np.array(y_train_img)
    y_test_img = np.array(y_test_img)
    print(x_train_img.shape, y_train_img.shape)

    # Define input and output to the CNN model
    num_classes = 5
    input_shape = (Dimension, Dimension, 3)

    # Image normalization
    x_train_img = x_train_img.astype("float32") / 255
    x_test_img = x_test_img.astype("float32") / 255
    print("orig x_train shape:", x_train_img.shape)

    # convert class vectors to binary class matrices
    y_train_img = keras.utils.to_categorical(y_train_img, num_classes)
    y_test_img = keras.utils.to_categorical(y_test_img, num_classes)

    # Image model load
    img_model = keras.models.load_model("subreddit.model")

    # Text model load
    text_model = pickle.load(open("best_lr.sav", "rb"))

    # Getting trained CNN models predictions on train data
    img_y_train_pred = img_model.predict(x_train_img)

    # Getting trained LR predictions on train data
    text_y_train_pred = []
    for i in range(len(x_train)):
        text_y_train_pred.append(text_model.predict_proba(x_train[i]))
    text_y_train_pred = np.array(text_y_train_pred)
    print(img_y_train_pred.shape, text_y_train_pred.shape)

    # Removing the unecessary dimension
    text_y_train_pred = np.squeeze(text_y_train_pred)
    print(text_y_train_pred.shape)

    # Getting image predictions on test data
    img_y_test_pred = img_model.predict(x_test_img)

    # Text test probability predictions
    text_y_test_pred = []

    for i in range(len(x_test)):
        text_y_test_pred.append(text_model.predict_proba(x_test[i]))

    text_y_test_pred = np.squeeze(text_y_test_pred)
    print(text_y_test_pred.shape)

    # Combining image and text features
    combined_features_train = np.concatenate(
        (img_y_train_pred, text_y_train_pred), axis=1
    )
    combined_features_test = np.concatenate((img_y_test_pred, text_y_test_pred), axis=1)

    print(combined_features_train.shape, combined_features_test.shape)

    # Running 3rd LR model on combined features
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    lr_clf = LogisticRegression()
    best_lr_clf, train_acc_list, val_acc_list = run_model(
        lr_clf, combined_features_train, combined_features_test, y_train, y_test
    )

    # Print accuracy
    print(
        "Train accuracy score",
        accuracy_score(best_lr_clf.predict(combined_features_train), y_train),
    )
    print(
        "Test accuracy score",
        accuracy_score(best_lr_clf.predict(combined_features_test), y_test),
    )

    # Trained model coefficients
    print("Trained LR models coefficients")
    print(best_lr_clf.coef_)

    # Actual test data
    df_test = pd.read_csv("Test Cleaned data.csv")
    df_test["target"] = df_test["subreddit"].progress_apply(
        lambda x: Label_List.index(x)
    )

    # Getting text features on actual test data
    test_X = df_test["reddit_submissions_title"]
    test_y = df_test["target"]
    test_X_vect = tfidfvectorizer.transform(test_X)
    vocab = tfidfvectorizer.get_feature_names_out()
    print("Vocab size : ", len(vocab))

    # Preparing test data for images
    final_test_img_files = df_test["IMG_NAME"].values
    Train_source = "images19"

    final_x_test = []
    final_y_test = []

    for f in tqdm(final_test_img_files):
        f += ".jpg"
        im = Image.open(Train_source + "/" + f)
        new_im = im.resize((Dimension, Dimension))
        rgb = np.array(new_im.convert("RGB"))
        final_x_test.append(rgb)
        label = []
        label.append(Label_List.index(f.split("_")[0]))
        label = np.array(label)
        final_y_test.append(label)

    final_x_test = np.array(final_x_test)
    final_y_test = np.array(final_y_test)
    print(final_x_test.shape)
    print(final_y_test.shape)

    # Normalizing image features
    final_x_test = final_x_test.astype("float32") / 255

    # Converting test labels to OHE
    final_y_test = keras.utils.to_categorical(final_y_test, num_classes)

    # Getting image predictions on test data
    final_img_y_test_pred = img_model.predict(final_x_test)

    # Getting text predictions on test data
    final_text_y_test_pred = text_model.predict_proba(test_X_vect)

    print(final_img_y_test_pred.shape)
    print(final_text_y_test_pred.shape)

    # Combined features on actual test data
    final_combined_features_test = np.concatenate(
        (final_img_y_test_pred, final_text_y_test_pred), axis=1
    )
    print(final_combined_features_test.shape)

    # Print accuracy
    print(
        "Train accuracy score",
        accuracy_score(best_lr_clf.predict(combined_features_train), y_train),
    )
    print(
        "Test accuracy score",
        accuracy_score(best_lr_clf.predict(final_combined_features_test), test_y),
    )
