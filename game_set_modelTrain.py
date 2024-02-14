"""
Created on Wed Feb 14 13:43:29 2024

@author: chandu11feb
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

NUM_PLAYERS = 3
NUM_SETS = 20
class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Predictor class.

        Parameters:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            output_size (int): The number of output features.
        """
        super(Predictor, self).__init__()
        # LSTM layer with input_size as input dimension and hidden_size as output dimension
        self.lstm = nn.LSTM(input_size, hidden_size)
        # Fully connected layer to map hidden_size to output_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Define the forward pass of the Predictor class.
        Parameters:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output prediction.
        """
        # Pass the input through the LSTM layer
        x, _ = self.lstm(x)
        # Take the last hidden state as the output
        x = x[-1]
        # Pass the last hidden state through the fully connected layer
        x = self.fc(x)
        return x

def prepare_data1(game_data):
    """
    Prepare data for training a model.

    Args:
    - game_data (list): A list of lists representing sequences of game data.

    Returns:
    - X (list): A list of one-hot encoded sequences, where each sequence represents the input data.
    - y (list): A list of tensors representing the target data.

    This function takes in game data, which is a list of sequences, where each sequence represents
    a series of events in a game. It prepares the data by converting each sequence into input-output
    pairs, where the input is a one-hot encoded representation of the events up to the second-to-last event,
    and the output is the last event in the sequence.

    The function returns two lists, X and y, where X contains the input sequences as one-hot encoded tensors,
    and y contains the target events as tensors.
    """
    X = []
    y = []
    for game in game_data:
        game_X = []
        game_y = []
        for i in range(len(game) - 1):
            game_X.append(game[i])
            game_y.append(game[i+1])
        # One-hot encode sequences
        X.append(torch.nn.functional.one_hot(torch.tensor(game_X), num_classes=3))
        # X.append(torch.tensor(game_X))  # Alternative: Uncomment this line for non-one-hot encoded input
        y.append(torch.tensor(game_y))
    # print(X)
    # print(y)
    return X, y

def train_model_linear(model, X_train, y_train, epochs=10, lr=0.001):
    """
    Function to train a linear model using game sequences data.

    Parameters:
        model (torch.nn.Module): The linear model to train.
        X_train (list): List of game sequences.
        y_train (list): List of labels indicating the winner of each game sequence.
        epochs (int): Number of training epochs (default is 10).
        lr (float): Learning rate for the optimizer (default is 0.001).

    Returns:
        None

    Note:
        This function trains the model using a CrossEntropyLoss criterion and Adam optimizer.
        It iterates over each epoch and within each epoch over each game sequence and its corresponding label.
        For each game sequence, it computes the loss, performs backpropagation, and updates the model parameters.
        It prints the average loss per epoch along with the current time.
    """
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss criterion
    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam optimizer with specified learning rate
    print("Training Started :",time.ctime())
    for epoch in range(epochs):  # Iterate over epochs
        total_loss = 0
        for game_sequence, next_winner in zip(X_train, y_train):  # Iterate over game sequences and labels
            for i in range(1, len(game_sequence)):  # Iterate over sets with context
                current_winner = torch.tensor(game_sequence[:i])  # Extract the current sequence
                one_hot_input = torch.nn.functional.one_hot(current_winner.long(), num_classes=3)  # Convert to one-hot encoding
                outputs = model(one_hot_input.float())  # Forward pass through the model
                loss = criterion(outputs.view(-1), next_winner[i-1])  # Compute the loss by comparing with actual winner.
                total_loss += loss.item()  # Accumulate the total loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(X_train)}, Time : {time.ctime()}')

def train_model_allCombination(model, X_train, y_train, epochs=10, lr=0.001):
    """
    Train a neural network model using all combinations of game sequences.

    Args:
        model (torch.nn.Module): The neural network model to train.
        X_train (list of lists): List of game sequences.
        y_train (list of lists): List of corresponding winners for each game sequence.
        epochs (int): Number of training epochs (default is 10).
        lr (float): Learning rate for the optimizer (default is 0.001).

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()  # Define the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Define the optimizer
    for epoch in range(epochs):  # Iterate over epochs
        total_loss = 0  # Initialize total loss for this epoch
        for game_sequence, next_winner in zip(X_train, y_train):  # Iterate over game sequences and their corresponding winners
            for i in range(0, len(game_sequence)):  # Iterate over sets with context
                for j in range(i + 1, len(game_sequence)):  # Iterate over pairs of players in the game sequence
                    current_winner = torch.tensor(game_sequence[i:j])  # Extract the current sequence
                    one_hot_input = torch.nn.functional.one_hot(current_winner.long(), num_classes=3)  # Convert current winner to one-hot encoding
                    outputs = model(one_hot_input.float())  # Get model predictions
                    loss = criterion(outputs.view(-1), next_winner[j - 1])  # Calculate the loss by comparing with actual winner.
                    total_loss += loss.item()  # Accumulate the total loss for this epoch
                    loss.backward()  # Backpropagate the loss
                    optimizer.step()  # Update model parameters
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(X_train)} , Time : {time.ctime()}')  # Print epoch information


def predict_winner2(model, initial_winner):
    # Initialize the sequence with the initial winner
    sequence = [int(initial_winner)]  # Ensure integer type
    correct_predictions = 0

    # Iterate through each set
    for i in range(NUM_SETS):
        # Print the current sequence
        print("sequence ", sequence)

        # Convert sequence into one-hot encoding for input to the model
        input_data = torch.nn.functional.one_hot(torch.tensor(sequence).long(), num_classes=3)
        print("input_data  ", input_data)

        # Pass input data through the model to get predictions
        with torch.no_grad():
            output = model(input_data.float())
        print(output)

        # Get the predicted winner for the current set
        predicted_winner = torch.argmax(output).item()
        print(f'Predicted winner for set {i + 1}: Player {predicted_winner}')

        # Prompt user to input the actual winner for the set
        true_winner = int(input("Enter the actual winner (0 for A, 1 for B, 2 for C): "))

        # Check if the predicted winner matches the actual winner
        if predicted_winner == true_winner:
            correct_predictions += 1

        # Add the true winner to the sequence for the next iteration
        sequence.append(int(true_winner))  # Ensure integer type

    # Calculate and print the accuracy for the game
    accuracy = correct_predictions / NUM_SETS
    print(f'Accuracy for this game: {accuracy * 100} %')


dataset_file = "game_dataset.txt"
with open(dataset_file, 'r') as file:
    game_data = [[int(winner) for winner in line.strip()] for line in file]
print(game_data)

X_train, y_train = prepare_data1(game_data)


# model_path = None
model_path = "set_predictor_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_path is None:
    print("Creating New Predictor Model.")
    input_size = 1
    hidden_size = 128
    output_size = 3
    model = Predictor(input_size, hidden_size, output_size)
    model.to(device)
    train_model_linear(model, X_train, y_train)
    model_path = "set_predictor_model.pth"
    torch.save(model, model_path)
else:
    print("Loading",model_path,"Model for Traning New Data.")
    model = torch.load(model_path, map_location=device)
    model.eval()
    train_model_linear(model, X_train, y_train)
    torch.save(model, model_path)
print("Traning Completed and Model",model_path," is Saved.")


