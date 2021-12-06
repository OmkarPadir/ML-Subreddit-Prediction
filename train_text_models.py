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

import warnings

warnings.filterwarnings("ignore")

nltk.download("all")
nltk.download("words")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

tqdm.pandas()


def remove_emojis(data):
    emoj = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", data)


def clean_text(text):
    # Remove URL
    text = re.sub(r"^https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)

    # Only English alphabets and numbers
    # text = re.sub(r'[^A-Za-z0-9_-]',' ', text)
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenize text
    tokenized_text = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    cleaned_text_tokens = [token for token in tokenized_text if token not in stop_words]

    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(token) for token in cleaned_text_tokens
    ]  # noun
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, "v") for token in lemmatized_tokens
    ]  # verb

    cleaned_text = " ".join(lemmatized_tokens)
    return cleaned_text


# Function to convert text into OHE vectors
def convertToVec(X, vectorizer_type="tfidf", ngram_tuple=(1, 1), min_df=1):
    vectorizer = None
    if vectorizer_type == "count":
        vectorizer = CountVectorizer(ngram_range=ngram_tuple, min_df=min_df)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_tuple, min_df=min_df)

    X_vect = vectorizer.fit_transform(X)
    return vectorizer, X_vect


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

    # Load csv file
    df = pd.read_csv("All_Titles_20_21.csv")
    df = df[["subreddit", "reddit_submissions_title"]]
    df.head()

    # Remove emojis
    df["reddit_submissions_title"] = df["reddit_submissions_title"].progress_apply(
        lambda x: remove_emojis(x)
    )

    # Clean text
    df["reddit_submissions_title"] = df["reddit_submissions_title"].progress_apply(
        lambda x: clean_text(x)
    )

    # Remove other languages
    print("Before", len(df))
    df = df[df["reddit_submissions_title"].map(lambda x: x.isascii())]
    print("After", len(df))

    # Removed rows with empty title
    print(
        "Before removing", len(df.loc[df["reddit_submissions_title"].str.strip() == ""])
    )
    df.drop(
        df.loc[df["reddit_submissions_title"].str.strip() == ""].index, inplace=True
    )
    print(
        "After removing", len(df.loc[df["reddit_submissions_title"].str.strip() == ""])
    )

    # Drop duplicates
    print("Dataset size before dropping duplicates", len(df))
    df.drop_duplicates(subset=["reddit_submissions_title"], keep="first", inplace=True)
    print("Dataset size after dropping duplicates", len(df))

    # Reset Index
    df.reset_index(drop=True, inplace=True)

    # Save cleaned data csv
    df.to_csv("All Cleaned data.csv", index=False)

    # Separate features and labels
    X = df["reddit_submissions_title"]
    y = df["subreddit"]
    print(len(X), len(y))

    # Convert y from categories to numbers
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(y.shape)

    # Use CountVectorizer - LR and tune word frequency
    rare_threshold = list(range(1, 11))
    train_mean_acc, val_mean_acc = [], []
    train_std_acc, val_std_acc = [], []

    for threshold in rare_threshold:
        print("Rare word threshold", threshold)
        countvectorizer, X_vect = convertToVec(X, "count", min_df=threshold)
        vocab = countvectorizer.get_feature_names_out()
        print("Vocab size : ", len(vocab))

        x_train, x_test, y_train, y_test = train_test_split(
            X_vect, y, test_size=0.20, random_state=42
        )

        # Logistic regression
        lr_clf = LogisticRegression()
        best_lr_clf, train_acc_list, val_acc_list = run_model(
            lr_clf, x_train, x_test, y_train, y_test
        )

        # Mean scores
        train_mean_acc.append(np.array(train_acc_list).mean())
        train_std_acc.append(np.array(train_acc_list).std())
        val_mean_acc.append(np.array(val_acc_list).mean())
        val_std_acc.append(np.array(val_acc_list).std())

        # Print accuracy
        print(
            "Train accuracy score",
            accuracy_score(best_lr_clf.predict(x_train), y_train),
        )
        print(
            "Test accuracy score", accuracy_score(best_lr_clf.predict(x_test), y_test)
        )

    plt.errorbar(
        rare_threshold, val_mean_acc, yerr=val_std_acc, label="Validation data"
    )
    plt.errorbar(
        rare_threshold, train_mean_acc, yerr=train_std_acc, label="Training data"
    )
    plt.xlabel("Word Frequency")
    plt.ylabel("Accuracy")
    plt.title("Errorbar of accuracy for varying word frequency threshold")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        "Errorbar of frequency of words on LR.png", facecolor="white", transparent=False
    )
    plt.show()

    # Use CountVectorizer LR and tune C
    # Small C values more regularization
    # Using threshold value 10 to reduce overfitting

    C_range = [0.01, 0.1, 1]
    train_mean_acc, val_mean_acc = [], []
    train_std_acc, val_std_acc = [], []
    threshold = 10

    for C in C_range:
        print("C value", C)
        countvectorizer, X_vect = convertToVec(X, "count", min_df=threshold)
        # vocab = countvectorizer.get_feature_names_out()
        # print("Vocab size : ",len(vocab))

        x_train, x_test, y_train, y_test = train_test_split(
            X_vect, y, test_size=0.20, random_state=42
        )

        # Logistic regression
        lr_clf = LogisticRegression(C=C)
        best_lr_clf, train_acc_list, val_acc_list = run_model(
            lr_clf, x_train, x_test, y_train, y_test
        )

        # Mean scores
        train_mean_acc.append(np.array(train_acc_list).mean())
        train_std_acc.append(np.array(train_acc_list).std())
        val_mean_acc.append(np.array(val_acc_list).mean())
        val_std_acc.append(np.array(val_acc_list).std())

        # Print accuracy
        print(
            "Train accuracy score",
            accuracy_score(best_lr_clf.predict(x_train), y_train),
        )
        print(
            "Test accuracy score", accuracy_score(best_lr_clf.predict(x_test), y_test)
        )

    plt.errorbar(C_range, val_mean_acc, yerr=val_std_acc, label="Validation data")
    plt.errorbar(C_range, train_mean_acc, yerr=train_std_acc, label="Training data")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("Errorbar of accuracy for varying penalty parameter C")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        "Errorbar of penalty param C on LR.png", facecolor="white", transparent=False
    )
    plt.show()

    # Use CountVectorizer - KNN and tune K range
    # Frequency threshold = 10

    K_range = [3, 5, 7]
    train_mean_acc, val_mean_acc = [], []
    train_std_acc, val_std_acc = [], []

    for K in K_range:
        print("K value", K)
        countvectorizer, X_vect = convertToVec(X, "count", min_df=10)
        # vocab = countvectorizer.get_feature_names_out()
        # print("Vocab size : ",len(vocab))

        x_train, x_test, y_train, y_test = train_test_split(
            X_vect, y, test_size=0.20, random_state=42
        )

        # Logistic regression
        knn_clf = KNeighborsClassifier(n_neighbors=K)
        best_knn_clf, train_acc_list, val_acc_list = run_model(
            knn_clf, x_train, x_test, y_train, y_test
        )

        # Mean scores
        train_mean_acc.append(np.array(train_acc_list).mean())
        train_std_acc.append(np.array(train_acc_list).std())
        val_mean_acc.append(np.array(val_acc_list).mean())
        val_std_acc.append(np.array(val_acc_list).std())

        # Print accuracy
        print(
            "Train accuracy score",
            accuracy_score(best_knn_clf.predict(x_train), y_train),
        )
        print(
            "Test accuracy score", accuracy_score(best_knn_clf.predict(x_test), y_test)
        )

    plt.errorbar(K_range, val_mean_acc, yerr=val_std_acc, label="Validation data")
    plt.errorbar(K_range, train_mean_acc, yerr=train_std_acc, label="Training data")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Errorbar of accuracy for varying k")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        "Errorbar of varying k in KNN - CountVec.png",
        facecolor="white",
        transparent=False,
    )
    plt.show()

    # Use TfIdfVectorizer - LR and tune word frequency
    rare_threshold = list(range(1, 11))
    train_mean_acc, val_mean_acc = [], []
    train_std_acc, val_std_acc = [], []

    for threshold in rare_threshold:
        print("Rare word threshold", threshold)
        tfidfvectorizer, X_vect = convertToVec(X, "tfidf", min_df=threshold)
        vocab = tfidfvectorizer.get_feature_names_out()
        print("Vocab size : ", len(vocab))

        x_train, x_test, y_train, y_test = train_test_split(
            X_vect, y, test_size=0.20, random_state=42
        )

        # Logistic regression
        lr_clf = LogisticRegression()
        best_lr_clf, train_acc_list, val_acc_list = run_model(
            lr_clf, x_train, x_test, y_train, y_test
        )

        # Mean scores
        train_mean_acc.append(np.array(train_acc_list).mean())
        train_std_acc.append(np.array(train_acc_list).std())
        val_mean_acc.append(np.array(val_acc_list).mean())
        val_std_acc.append(np.array(val_acc_list).std())

        # Print accuracy
        print(
            "Train accuracy score",
            accuracy_score(best_lr_clf.predict(x_train), y_train),
        )
        print(
            "Test accuracy score", accuracy_score(best_lr_clf.predict(x_test), y_test)
        )

    plt.errorbar(
        rare_threshold, val_mean_acc, yerr=val_std_acc, label="Validation data"
    )
    plt.errorbar(
        rare_threshold, train_mean_acc, yerr=train_std_acc, label="Training data"
    )
    plt.xlabel("Word Frequency")
    plt.ylabel("Accuracy")
    plt.title("Errorbar of accuracy for varying word frequency threshold")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        "Errorbar of frequency of words on LR - TfIdfVec.png",
        facecolor="white",
        transparent=False,
    )
    plt.show()

    # Save the best LR classifier
    filename = "best_lr.sav"
    pickle.dump(best_lr_clf, open(filename, "wb"))

    # Use TfidfVectorizer - KNN and tune K range
    # Frequency threshold = 10

    K_range = [3, 5, 7]
    train_mean_acc, val_mean_acc = [], []
    train_std_acc, val_std_acc = [], []

    for K in K_range:
        print("K value", K)
        tfidfvectorizer, X_vect = convertToVec(X, "tfidf", min_df=10)
        # vocab = countvectorizer.get_feature_names_out()
        # print("Vocab size : ",len(vocab))

        x_train, x_test, y_train, y_test = train_test_split(
            X_vect, y, test_size=0.20, random_state=42
        )

        # Logistic regression
        knn_clf = KNeighborsClassifier(n_neighbors=K)
        best_knn_clf, train_acc_list, val_acc_list = run_model(
            knn_clf, x_train, x_test, y_train, y_test
        )

        # Mean scores
        train_mean_acc.append(np.array(train_acc_list).mean())
        train_std_acc.append(np.array(train_acc_list).std())
        val_mean_acc.append(np.array(val_acc_list).mean())
        val_std_acc.append(np.array(val_acc_list).std())

        # Print accuracy
        print(
            "Train accuracy score",
            accuracy_score(best_knn_clf.predict(x_train), y_train),
        )
        print(
            "Test accuracy score", accuracy_score(best_knn_clf.predict(x_test), y_test)
        )

    plt.errorbar(K_range, val_mean_acc, yerr=val_std_acc, label="Validation data")
    plt.errorbar(K_range, train_mean_acc, yerr=train_std_acc, label="Training data")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Errorbar of accuracy for varying k")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        "Errorbar of varying k in KNN - TfidfVec.png",
        facecolor="white",
        transparent=False,
    )
    plt.show()
