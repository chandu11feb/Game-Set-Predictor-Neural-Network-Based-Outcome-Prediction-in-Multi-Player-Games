"""
Created on Wed Feb 14 13:43:29 2024

@author: chandu11feb
"""

import torch
import torch.nn as nn

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
def predict_winner2(model, initial_winner):
    # Initialize the sequence with the initial winner
    sequence = [int(initial_winner)]  # Ensure integer type
    correct_predictions = 0

    # Iterate through each set
    for i in range(NUM_SETS):
        # Print the current sequence
        # print("sequence ", sequence)

        # Convert sequence into one-hot encoding for input to the model
        input_data = torch.nn.functional.one_hot(torch.tensor(sequence).long(), num_classes=3)
        # print("input_data  ", input_data)

        # Pass input data through the model to get predictions
        with torch.no_grad():
            output = model(input_data.float())
        # print(output)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "set_predictor_model.pth"
model = torch.load(model_path, map_location=device)
model.eval()


initial_winner = int(input("Enter the winner of the first set (0 for A, 1 for B, 2 for C): "))
predict_winner2(model, initial_winner)




