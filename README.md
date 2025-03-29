# Spam SMS Detection

A machine learning project to classify SMS messages as spam or non-spam for the GrowthLink internship.

## Task Objectives
- Develop a model to identify spam SMS messages using the SMS Spam Collection Dataset.
- Preprocess text data and apply a classification algorithm.
- Evaluate the model’s performance.

## Steps to Run the Project
1. Clone this repository: `git clone <your-repo-url>`
2. Set up a virtual environment: `python -m venv spam_sms_env`
3. Activate it: `spam_sms_env\Scripts\activate` (Windows) or `source spam_sms_env/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place `spam.csv` in the project folder.
6. Run the script: `python spam_detection.py`

## Results
- Model accuracy and classification report are saved in `results.txt`.
- The Naive Bayes model effectively distinguishes spam from ham messages.

## Dataset
- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)