from Functions import Train_Models as fnc
from Classes.data_cleaning import Data_Cleaning
from Classes.data_preprocessing import Data_Preprocessing
from Classes.data_exploration import Data_Exploration

def data_cleaning(answers, questions):

    print('\033[1m' + '\n Starting Data Cleaning' + '\033[0m')
    cleaning = Data_Cleaning()
    answers = cleaning.rename_columns(answers, 'q_id', 'id')

    # merge Question and Answers dataframe:
    merged_df = cleaning.merge_dataset(questions, answers, 'id')
    print(merged_df.head())

    # Rename columns:
    merged_df = cleaning.rename_columns(merged_df, 'text_x', 'question')
    merged_df = cleaning.rename_columns(merged_df, 'text_y', 'answer')

    # drop columns:
    columns_list = ['votes_x', 'timestamp', 'datetime', 'Unnamed: 0']
    merged_df = cleaning.drop_columns(merged_df, columns_list)

    # dataframe describe:
    print(merged_df.describe())

    # treat null values in votes column
    merged_df = cleaning.clean_votes(merged_df)

    # Type casting to string
    merged_df = cleaning.type_casting_str(merged_df, 'question')
    merged_df = cleaning.type_casting_str(merged_df, 'answer')

    print(merged_df.describe())
    return merged_df

def data_preprocessing(cleaned_df):

    print('\033[1m' + '\n Starting Data Preprocessing' + '\033[0m')
    preprocessing = Data_Preprocessing()
    # remove stop words and punctuations
    cleaned_df['question'] = cleaned_df['question'].apply(preprocessing.preprocess_text)
    cleaned_df['answer'] = cleaned_df['answer'].apply(preprocessing.preprocess_text)

    # filter out comments having less than 4 characters
    cleaned_df['answer'] = cleaned_df['answer'].apply(preprocessing.filter_comments)

    # only taking two columns:
    filtered_df = cleaned_df[['question', 'answer']]
    filtered_df.columns = ['question', 'answer']
    filtered_df.reset_index(drop=True, inplace=True)

    # Concat chatbot dialogues
    reddit_df = preprocessing.Concat_chatbot_dialogues(filtered_df, "./Dataset/chatbot_dialouges.csv")

    # number of null values in answers
    reddit_df = reddit_df.dropna()
    null_count = reddit_df['answer'].isnull().sum()
    print("Number of null values in answer column: ", null_count)

    # get sample of the data
    print('\033[1m' + '\n Taking Sample Data' + '\033[0m')
    reddit_df = reddit_df.sample(n=2000)
    print(reddit_df.shape, "\n")

    # saved the cleaned dataframe:
    preprocessing.save_file_csv(reddit_df, "reddit_df.csv")
    print("Checkpoint Saved \n\n")

def data_exploration(reddit_df):
    obj_viz = Data_Exploration()

    # Display First Visualization
    obj_viz.viz_1(reddit_df)

    # Display First Visualization
    obj_viz.viz_2(reddit_df)

    # Display First Visualization
    obj_viz.viz_3(reddit_df)

def data_modeling(reddit_df, model_to_train):

    if model_to_train == 'LSTM':
        # LSTM
        print('\033[1m' + '\n Running LSTM' + '\033[0m')
        fnc.train_LSTM(reddit_df)

    elif model_to_train == 'BERT2BERT':
        # BERT2BERT
        print('\033[1m' + '\n Running BERT2BERT' + '\033[0m')
        fnc.train_BERT2BERT(reddit_df)

    elif model_to_train == 'T5':
        # T5
        print('\033[1m' + '\n Running T5' + '\033[0m')
        fnc.train_T5(reddit_df)

    elif model_to_train == 'Hybrid Custom Model':
        # Hybrid Custom Model
        print('\033[1m' + '\n Running Hybrid Custom Model' + '\033[0m')
        fnc.train_HybridModel(reddit_df)

    else:
        print("Please Select Valid Model!" + "\n\n")




