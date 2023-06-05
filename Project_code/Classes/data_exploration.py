import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Data_Exploration():

    def viz_1(self, reddit_df):
        # Distribution of answer counts in the whole dataset
        num_answers = reddit_df.groupby('question')['answer'].apply(
            lambda x: len(x.str.split('[SEP]'))).reset_index(name='num_answers')

        # Group by the number of answers and count the number of unique questions for each group
        num_answers_count = num_answers['num_answers'].value_counts().reset_index()
        num_answers_count.columns = ['num_answers', 'num_questions']
        # Set the figure size
        plt.subplots(figsize=(8, 6))
        # Create a plot of the number of questions for each group of answer numbers
        sns.histplot(data=num_answers_count, x='num_answers', weights='num_questions', hue='num_answers',
                     palette='deep', bins=30)

        # Set the plot title and axis labels
        plt.title('Distribution of Unique Answer Numbers')
        plt.xlabel('Number of Answers')
        plt.ylabel('Number of Questions')

        # Show the plot
        plt.show()

    def viz_2(self, reddit_df):
        # Create a new column with the length of each comment
        reddit_df['comment_length'] = reddit_df['answer'].apply(len)

        # Create a histogram of comment lengths
        sns.histplot(data=reddit_df, x='comment_length', color='orange', fill=False)

        # Set the plot title and axis labels
        plt.title('Distribution of Comment Lengths')
        plt.xlabel('Comment Length')
        plt.ylabel('Frequency')

        # Show the plot
        plt.show()

    def viz_3(self, reddit_df):
        # Combine all the questions into a single string
        all_questions = ' '.join(reddit_df['question'])

        # Create a word cloud
        wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=1000,
                              colormap='magma').generate(all_questions)

        # Plot the word cloud
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        # Show the plot
        plt.show()