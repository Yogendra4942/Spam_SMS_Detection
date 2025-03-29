Spam SMS Detection

📌 Overview

This project is a Spam SMS Detection system that classifies messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and Machine Learning. The model is trained using the Multinomial Naïve Bayes algorithm with TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

🚀 Features

Preprocesses SMS text by tokenization, stopword removal, and text cleaning.

Converts text data into numerical features using TF-IDF Vectorization.

Uses Multinomial Naïve Bayes for classification.

Evaluates performance using accuracy score and classification report.

Saves model evaluation results to results.txt.

📂 Dataset

The dataset used is spam.csv, which contains SMS messages labeled as ham (not spam) or spam. The dataset is preprocessed before training.