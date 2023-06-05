# This class contains all the procedures for data preprocessing requires for entire dataset!!

import re
import string
import nltk
import pandas as pd
nltk.download('punkt')
from nltk.tokenize import word_tokenize


class Data_Preprocessing():
    def __init__(self, ):
        super().__init__()

    def preprocess_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www.\S+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def filter_comments(self, comment):
        # Tokenize comment into words
        words = [word for word in word_tokenize(comment) if word.isalpha()]
        # Filter out comments that are too short
        if len(words) < 5 and len(words) > 128:
            return None
        return comment

    def save_file_csv(self, x, file_name):
        folder_name = "./Dataset/Saved_Dataset/"
        file_name = folder_name + file_name
        x.to_csv(file_name, index=False, encoding="utf-8")

    def Concat_chatbot_dialogues(self, df, path):
        chatbot_dialogues = pd.read_csv(path)
        reddit_data = pd.concat([df, chatbot_dialogues], axis=0)
        reddit_data = reddit_data.reset_index(drop=True)
        return reddit_data