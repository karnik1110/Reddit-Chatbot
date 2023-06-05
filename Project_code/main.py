# Author: Group 3
# Date: May 22, 2023
# Description: This program is a Data 255: Deep Learning tech project.

import pandas as pd
from Functions import project_functions as pfnc
import gdown
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    file_names = ['reddit_answers.csv', 'reddit_questions.csv', 'chatbot_dialouges.csv']
    folder_path = './Dataset/'

    # Check if all files exist
    all_exist = all(os.path.exists(os.path.join(folder_path, file_name)) for file_name in file_names)

    if all_exist:
        print("All dataset files already exist.")
    else:
        # Create the missing files
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                if file_name == 'reddit_answers.csv':
                    file_id = '1KgHrKdXQ8b53ZpbyeNdbUcYNPyNYwwxK'

                elif file_name == 'reddit_questions.csv':
                    file_id = '1C8TSqRwnGO48BuRSjqSHOwFfe4SsenlY'

                elif file_name == 'chatbot_dialouges.csv':
                    file_id = '1AlB8HN9WaKMcai6HNiktk0Yj8rOgpwFx'

                # Download the file from Google Drive
                gdown.download(f'https://drive.google.com/uc?id={file_id}', file_path, quiet=False)

                print(f"File '{file_name}' Downloaded successfully.")

    # Load the dataset
    reddit_answer = pd.read_csv("./Dataset/reddit_answers.csv", delimiter=";")
    reddit_question = pd.read_csv("./Dataset/reddit_questions.csv", delimiter=";")

    # Data Cleaning
    cleaned_df = pfnc.data_cleaning(reddit_answer, reddit_question)

    # Data Preprocessing
    pfnc.data_preprocessing(cleaned_df)

    # Data Exploration
    pfnc.data_exploration(cleaned_df)

    # Load Cleaned dataframe:
    reddit_df = pd.read_csv("./Dataset/Saved_Dataset/reddit_df.csv")

    # Display the input
    print("Hello! This is the Reddit Assist Code. We have implemented four Deep Learning Models. Here are These:\n 1. LSTM \n 2. BERT2BERT \n 3. T5 \n 4. Hybrid Custom Model \n\n Please select the model you want to train.. !")

    # Prompt for a numeric input
    selected_model = int(input("Mention 1,2,3 or 4: "))

    if selected_model == 1:
        # LSTM
        pfnc.data_modeling(reddit_df, "LSTM")

    elif selected_model == 2:
        # BERT2BERT
        pfnc.data_modeling(reddit_df, "BERT2BERT")

    elif selected_model == 3:
        # T5
        pfnc.data_modeling(reddit_df, "T5")

    elif selected_model == 4:
        # T5
        pfnc.data_modeling(reddit_df, "Hybrid Custom Model")

    else:
        print("Please select valid model number!")

