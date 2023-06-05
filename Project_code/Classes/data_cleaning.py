# This class contains all the data cleaning methods for the entire dataset!!

class Data_Cleaning():
    def __init__(self, ):
        super().__init__()

    def rename_columns(self, df, old_name, new_name):
        df = df.rename(columns={old_name: new_name})
        return df

    def merge_dataset(self, questions, answers, col):
        df = questions.merge(answers, how="left", on=col)
        return df

    def drop_columns(self, df, columns_list):
        df = df.drop(columns_list, axis=1)
        return df

    def filter_comments_by_votes(self, df, vote_threshold):
        filtered_comments = []
        for vote in df["votes_y"]:
            if vote >= vote_threshold:
                filtered_comments.append(vote)
        return filtered_comments

    def type_casting_str(self, df, col):
        df[col] = df[col].astype('str')
        return df

    # def check_profanity(self, text):
    #     return predict([text])[0] == 0  # Return True if the text is clean, False if it's profane

    def clean_votes(self, df):
        df['votes_y'] = df['votes_y'].fillna(0)
        idx = df.groupby('question')[
            'votes_y'].idxmax()  # Find the index of the row with max votes per question
        df = df.loc[idx]
        df.reset_index(inplace=True, drop=True)
        return df
