from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input, BatchNormalization, Activation, Add
from tensorflow.keras import regularizers
from torch import nn
import numpy as np
from torch.utils.data import RandomSampler
from torch.utils.data import TensorDataset
import os
from transformers import BertGenerationEncoder, BertGenerationDecoder
from transformers import EncoderDecoderModel
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class LSTM_Class():

    def Preprocessing_LSTM(self, df):
        df = df.fillna('')
        return df

    def Train_Test_Split(self, df):
        train, val = train_test_split(df, test_size=0.2)
        return train, val

    def plot_loss_curve(self, train_loss, val_loss):
        # Plotting
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('\033[1m' + '\n BERT2BERT Configuration' + '\033[0m')
        plt.show()

    def predict_answer(self, model, question, tokenizer, max_len):
        # Tokenize the question
        tokenized_question = tokenizer.texts_to_sequences([question])

        # Pad the sequences
        padded_question = pad_sequences(tokenized_question, maxlen=max_len)

        # Generate prediction
        prediction = model.predict(padded_question)

        # Get the highest probability token
        predicted_token = np.argmax(prediction, axis=-1)

        # Convert tokens to words
        answer = ' '.join(tokenizer.sequences_to_texts(predicted_token))

        return answer

    # Model
    def residual_block(self, x, units):
        y = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
        y = Dropout(0.5)(y)
        y = BatchNormalization()(y)
        y = Add()([x, y])
        y = Activation('relu')(y)
        return y

    # Learning rate schedule
    def lr_schedule(self, epoch):
        if epoch < 10:
            return 0.001
        elif epoch < 20:
            return 0.0005
        else:
            return 0.0001


class BERT2BERT_Class():

    def Tokenize_Data_BERT2BERT(self, df, tokenizer):
        input_ids = tokenizer(list(df["question"]), add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=64).input_ids
        labels = tokenizer(list(df["answer"]), return_tensors="pt", padding=True, truncation=True, max_length=64).input_ids
        return input_ids, labels

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("Using CPU for training")

        return device

    def tensor_dataset(self, input_ids, labels, n_train, n_val, device):
        train_dataset = TensorDataset(input_ids[:n_train].to(device), labels[:n_train].to(device))
        val_dataset = TensorDataset(input_ids[n_train:n_train + n_val].to(device), labels[n_train:n_train + n_val].to(device))
        test_dataset = TensorDataset(input_ids[n_train + n_val:].to(device), labels[n_train + n_val:].to(device))
        return train_dataset, val_dataset, test_dataset

    def dataLoader(self,train_dataset, val_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, val_loader, test_loader

    def plot_loss_curve(self, train_losses, val_losses):
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('\033[1m' + '\n BERT2BERT Configuration' + '\033[0m')
        plt.legend()
        plt.show()

    def generate_answer(self, model_dir, n_epochs, device, tokenizer, bert2bert, question):
        # Load the saved model checkpoint
        model_checkpoint = os.path.join(model_dir, "epoch_{}.pt".format(n_epochs - 1))
        bert2bert.load_state_dict(torch.load(model_checkpoint, map_location=device))

        # Set the model to evaluation mode
        bert2bert.eval()

        # Tokenize the input question
        inputs = tokenizer.encode(question, return_tensors='pt').to(device)

        # Generate the answer using the model
        outputs = bert2bert.generate(inputs, max_length=20, num_beams=4, early_stopping=True)

        # Decode the generated answer and return it
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def BERT2BERT_Architecture(self):
        encoder1 = BertGenerationEncoder.from_pretrained("bert-large-uncased")
        # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
        decoder1 = BertGenerationDecoder.from_pretrained(
            "bert-large-uncased", add_cross_attention=True, is_decoder=True)
        bert2bert1 = EncoderDecoderModel(encoder=encoder1, decoder=decoder1).to('cpu')
        print('\033[1m' + '\n BERT2BERT Architecture' + '\033[0m')
        print(bert2bert1)

        print('\033[1m' + '\n BERT2BERT Configuration' + '\033[0m')
        print(bert2bert1.config)
        return bert2bert1

    def total_learnable_parameters(self, bert2bert1):
        total_trainable_params = sum(p.numel() for p in bert2bert1.parameters() if p.requires_grad)
        print('\033[1m' + '\n BERT2BERT Architecture' + '\033[0m', total_trainable_params)


class T5_Class():

    def Train_Test_Split(self, df):
        train, val = train_test_split(df, test_size=0.2, random_state=42)
        return train, val

    def tokenize_dataset(self, df, train_df, val_df, tokenizer):
        # Assuming you have your data in a pandas DataFrame called df, and you've split it into train_df and val_df
        train_dataset = RedditDataset(train_df, tokenizer, source_max_length=128, target_max_length=128)
        val_dataset = RedditDataset(val_df, tokenizer, source_max_length=128, target_max_length=128)
        return train_dataset, val_dataset

    def dataloader(self, train_dataset, val_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=16)
        val_dataloader = DataLoader(val_dataset, batch_size=16)
        return train_dataloader, val_dataloader

    def plot_loss_curve(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

class RedditDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_length, target_max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = str(row['question'])  # Ensure question is a string
        answer = str(row['answer'])  # Ensure answer is a string
        source_encoding = self.tokenizer(question, truncation=True,
                                         max_length=self.source_max_length, padding='max_length', return_tensors='pt')
        target_encoding = self.tokenizer(answer, truncation=True,
                                         max_length=self.target_max_length, padding='max_length', return_tensors='pt')

        labels = target_encoding['input_ids']
        labels[labels == 0] = -100  # We set padding tokens to -100, so they don't affect the loss

        return dict(
            question=row['question'],
            answer=row['answer'],
            input_ids=source_encoding['input_ids'].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )


class HybridModel_Class():

    def preprocess_data(self, data, tokenizer):
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in data:
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    def split_train_test(self, df):
        train, test = train_test_split(df, test_size=0.1)
        return train, test

    def preprocess_train_data(self, train, tokenizer):
        # Apply the function on your questions and answers columns
        questions_input_ids, questions_attention_masks = self.preprocess_data(train['question'], tokenizer)
        answers_input_ids, answers_attention_masks = self.preprocess_data(train['answer'], tokenizer)
        return questions_input_ids, questions_attention_masks, answers_input_ids, answers_attention_masks

    def preprocess_test_data(self, test, tokenizer):
        # Apply the function on your questions and answers columns
        questions_input_ids_test, questions_attention_masks_test = self.preprocess_data(test['question'], tokenizer)
        answers_input_ids_test, answers_attention_masks_test = self.preprocess_data(test['answer'], tokenizer)
        return questions_input_ids_test, questions_attention_masks_test, answers_input_ids_test, answers_attention_masks_test

    def tensor_dataset(self):
        # Combine the training inputs into a TensorDataset.
        train_dataset = TensorDataset(self.questions_input_ids, self.questions_attention_masks,
                                      self.answers_input_ids, self.answers_attention_masks)
        test_dataset = TensorDataset(self.questions_input_ids_test, self.questions_attention_masks_test,
                                     self.answers_input_ids_test, self.answers_attention_masks_test)
        return train_dataset, test_dataset

    def dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,  # The training samples.
            sampler=RandomSampler(self.train_dataset),  # Select batches randomly
            batch_size=16  # Trains with this batch size.
        )
        val_dataloader = DataLoader(
            self.test_dataset,  # The training samples.
            sampler=RandomSampler(self.test_dataset),  # Select batches randomly
            batch_size=16  # Trains with this batch size.
        )
        return train_dataloader, val_dataloader

class TransformerBlock(nn.Module):
    def __init__(self, k, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(k, num_heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.feed_forward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k),
        )

    def forward(self, value, key, query):
        attention_output, _ = self.attention(query, key, value)
        x = self.norm1(attention_output + query)
        forward_output = self.feed_forward(x)
        return self.norm2(forward_output + x)

class CustomModel(nn.Module):
    def __init__(self, num_tokens, k, num_heads, num_transformer_blocks):
        super(CustomModel, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, k)
        self.position_embedding = nn.Embedding(512, k)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(k, num_heads) for _ in range(num_transformer_blocks)])
        self.to_probs = nn.Linear(k, num_tokens)

    def forward(self, x):
        batch_size, seq_len = x.size()
        tokens = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        position_embeddings = self.position_embedding(positions)
        x = tokens + position_embeddings

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, x, x)

        x = self.to_probs(x)
        return x