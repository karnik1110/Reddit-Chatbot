import warnings
import torch.optim as optim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from torch import nn
from tensorflow.keras.models import Model
from Classes.data_modeling import LSTM_Class, BERT2BERT_Class, T5_Class, HybridModel_Class, CustomModel
from transformers import BertTokenizer
from transformers import BertGenerationEncoder, BertGenerationDecoder
from transformers import EncoderDecoderModel
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from transformers import get_linear_schedule_with_warmup

def train_LSTM(reddit_df):
    tokenizer = Tokenizer()
    model_LSTM = LSTM_Class()
    reddit_df = model_LSTM.Preprocessing_LSTM(reddit_df)

    train, val = model_LSTM.Train_Test_Split(reddit_df)
    tokenizer.fit_on_texts(train['question'].tolist() + train['answer'].tolist())

    # Sequences
    train_X = tokenizer.texts_to_sequences(train['question'])
    train_Y = tokenizer.texts_to_sequences(train['answer'])
    val_X = tokenizer.texts_to_sequences(val['question'])
    val_Y = tokenizer.texts_to_sequences(val['answer'])

    # Padding
    max_len = 100
    train_X = pad_sequences(train_X, maxlen=max_len)
    train_Y = pad_sequences(train_Y, maxlen=max_len)
    val_X = pad_sequences(val_X, maxlen=max_len)
    val_Y = pad_sequences(val_Y, maxlen=max_len)

    max_len = 100
    input_layer = Input(shape=(max_len,))
    model = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=256, input_length=max_len)(input_layer)
    model = LSTM(256, return_sequences=True)(model)  # Add this line
    model = model_LSTM.residual_block(model, 128)
    model = TimeDistributed(Dense(len(tokenizer.word_index) + 1, activation='softmax'))(model)

    model = Model(input_layer, model)
    model.summary()
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lr_callback = LearningRateScheduler(model_LSTM.lr_schedule)

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
    # Training
    history = model.fit(train_X, np.expand_dims(train_Y, -1), validation_data=(val_X, np.expand_dims(val_Y, -1)),batch_size=64, epochs=10, callbacks=[lr_callback, checkpoint_callback])
    model_LSTM.plot_loss_curve(history.history['loss'], history.history['val_loss'])

def train_BERT2BERT(reddit_df):
    # Display a warning message
    warnings.warn("This code requires higher GPU's operation. Do you want to continue?")
    # User confirmation
    user_input = input("Do you want to continue? (y/n): ")

    # Check user response
    if user_input.lower() == "y":
        model_Bert2Bert = BERT2BERT_Class()
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

        # BERT2BERT Architecture and configuration
        bert2bert1 = model_Bert2Bert.BERT2BERT_Architecture()

        # Total Learnable Parameters
        model_Bert2Bert.total_learnable_parameters(bert2bert1)

        # get input_ids and labels
        input_ids, labels = model_Bert2Bert.Tokenize_Data_BERT2BERT(reddit_df, tokenizer)

        # Create encoder and decoder for BERT2BERT
        encoder = BertGenerationEncoder.from_pretrained("bert-large-uncased")
        decoder = BertGenerationDecoder.from_pretrained("bert-large-uncased", add_cross_attention=True, is_decoder=True)
        bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        # get device
        device = model_Bert2Bert.get_device()

        # Divide data into training, validation, and test sets
        n_samples = len(input_ids)
        n_train = int(n_samples * 0.8)
        n_val = int(n_samples * 0.1)
        n_test = n_samples - n_train - n_val

        # Use Tensor dataset to get train, test and validation dataset
        train_dataset, val_dataset, test_dataset = model_Bert2Bert.tensor_dataset(input_ids, labels, n_train, n_val,
                                                                                  device)

        # DataLoader to loading the datasets created by tensor
        train_loader, val_loader, test_loader = model_Bert2Bert.dataLoader(train_dataset, val_dataset, test_dataset)

        bert2bert.to(device)

        # Create empty lists to store training and validation losses
        train_losses = []
        val_losses = []

        # Set gradient accumulation steps
        gradient_accumulation_steps = 2

        # Create optimizer and scheduler
        optimizer = AdamW(bert2bert.parameters(), lr=1e-5)

        # Set scaler for mixed precision training
        if torch.cuda.is_available():
            scaler = GradScaler()
        else:
            scaler = None

        # Train model on training set
        n_epochs = 5

        for epoch in range(n_epochs):
            # Training loop
            train_loss = 0.0
            correct = 0
            total = 0
            bert2bert.train()
            for i, (X, Y) in enumerate(train_loader):
                X = X.to(device)
                Y = Y.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = bert2bert(input_ids=X, decoder_input_ids=Y, labels=Y)
                    loss = outputs.loss
                    loss = loss / gradient_accumulation_steps  # divide loss by accumulation steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % gradient_accumulation_steps == 0:  # accumulate gradients
                    # Update parameters
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                predicted_labels = torch.argmax(outputs.logits, dim=-1)
                total += len(Y)
                correct += (predicted_labels == Y).sum().item()
                train_loss += loss.item()

            # Validation loop
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            bert2bert.eval()
            with torch.no_grad():
                for X, Y in val_loader:
                    X = X.to(device)
                    Y = Y.to(device)
                    with autocast():
                        outputs = bert2bert(input_ids=X, decoder_input_ids=Y, labels=Y)
                        loss = outputs.loss
                    val_loss += loss.item()
                    predicted_labels = torch.argmax(outputs.logits, dim=-1)
                    total_val += len(Y)
                    correct_val += (predicted_labels == Y).sum().item()
                    val_loss += loss.item()

            train_loss /= len(train_loader) * gradient_accumulation_steps  # divide by total number of batches
            train_acc = correct / total

            val_loss /= len(val_loader)
            val_acc = correct_val / total_val
            print(
                f"Epoch {epoch + 1} | Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}| Training Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f}")

            # Append training and validation losses to lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        torch.cuda.empty_cache()
        model_Bert2Bert.plot_loss_curve(train_losses, val_losses)
    else:
        print("You press No.. \n Stopping the BERT2BERT Model Training")

def train_T5(reddit_df):
    # Display a warning message
    warnings.warn("This code requires higher GPU's operation. Do you want to continue?")
    # User confirmation
    user_input = input("Do you want to continue? (y/n): ")

    # Check user response
    if user_input.lower() == "y":
        model_T5 = T5_Class()
        # Split the dataset:
        train, val = model_T5.Train_Test_Split(reddit_df)
        print(f'Training samples: {len(train)}')
        print(f'Validation samples: {len(val)}')

        # define tokenizer
        tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)

        # Tokenize dataset
        train_dataset, val_dataset = model_T5.tokenize_dataset(reddit_df, train, val, tokenizer)

        # Dataloader
        train_dataloader, val_dataloader = model_T5.dataloader(train_dataset, val_dataset)

        # train the model
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=1e-4)

        total_steps = len(train_dataloader) * 20  # Number of training epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        train_losses = []
        val_losses = []
        for epoch in range(20):  # Number of training epochs
            total_loss = 0
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()

                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)

            # Validation
            model.eval()
            total_val_loss = 0
            for batch in val_dataloader:
                with torch.no_grad():
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    # Forward pass
                    outputs = model(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    labels=batch['labels'])
                    loss = outputs.loss
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            print(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        model.save_pretrained('t5_reddit_chatbot')

        # plot loss curve
        model_T5.plot_loss_curve(train_losses, val_losses)

    else:
        print("You press No.. \n Stopping the T5 Model Training")


def train_HybridModel(reddit_df):
    # Display a warning message
    warnings.warn("This code requires higher GPU's operation. Do you want to continue?")
    # User confirmation
    user_input = input("Do you want to continue? (y/n): ")

    model_hybridModel = HybridModel_Class()

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # split dataset to train and test
    train, val = model_hybridModel.split_train_test(reddit_df)

    #  preprocess train and test dataset
    questions_input_ids, questions_attention_masks, answers_input_ids, answers_attention_masks = model_hybridModel.preprocess_train_data(train, tokenizer)
    questions_input_ids_test, questions_attention_masks_test, answers_input_ids_test, answers_attention_masks_test = model_hybridModel.preprocess_test_data(train, tokenizer)

    # tensor data
    train_dataset, test_dataset = model_hybridModel.tensor_dataset()

    # dataloader
    train_dataloader, val_dataloader = model_hybridModel.dataloader(train_dataset, test_dataset)

    # train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_customModel = CustomModel()

    # Initialize the model
    k = 64
    num_heads = 2
    num_transformer_blocks = 1
    num_tokens = tokenizer.vocab_size

    # Check user response
    if user_input.lower() == "y":
        model = model_customModel.CustomModel(num_tokens, k, num_heads, num_transformer_blocks).to(device)

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Define the loss function
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        # Number of training epochs
        epochs = 5

        # Store loss and accuracy data
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []

        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_output_ids = batch[1].to(device)

                model.zero_grad()
                outputs = model(b_input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), b_output_ids.view(-1))
                total_train_loss += loss.item()

                # Get the predictions
                _, preds = torch.max(outputs, dim=2)
                correct_predictions += (preds == b_output_ids).sum().item()
                total_predictions += b_output_ids.ne(tokenizer.pad_token_id).sum().item()

                # Perform a backward pass to calculate the gradients
                loss.backward()

                # Update parameters and take a step using the computed gradient
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            train_accuracy = correct_predictions / total_predictions
            train_loss.append(avg_train_loss)
            train_acc.append(train_accuracy)

            # Validation
            model.eval()
            total_val_loss = 0
            correct_predictions = 0
            total_predictions = 0
            for batch in val_dataloader:
                b_input_ids = batch[0].to(device)
                b_output_ids = batch[1].to(device)
                with torch.no_grad():
                    outputs = model(b_input_ids)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), b_output_ids.view(-1))
                total_val_loss += loss.item()

                # Get the predictions
                _, preds = torch.max(outputs, dim=2)
                correct_predictions += (preds == b_output_ids).sum().item()
                total_predictions += b_output_ids.ne(tokenizer.pad_token_id).sum().item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = correct_predictions / total_predictions
            val_loss.append(avg_val_loss)
            val_acc.append(val_accuracy)

            print(
                f'Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.3f}, Train Acc: {train_accuracy:.3f}, Val Loss: {avg_val_loss:.3f}, Val Acc: {val_accuracy:.3f}')

        else:
            print("You press No.. \n Stopping the Hybrid Custom Model Model Training")

