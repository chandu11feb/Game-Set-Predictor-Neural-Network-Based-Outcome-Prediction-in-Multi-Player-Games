# Game Set Predictor: Neural Network-Based Outcome Prediction in Multi-Player Games

### Model
The program utilizes a neural network architecture, specifically a **Long Short-Term Memory (LSTM) network**, to learn patterns from historical game data and make predictions about future game outcomes.

### Overview
The Game Set Predictor is a Python program that utilizes a neural network model to predict the winner of subsequent sets in a series of games based on historical data. It employs a Long Short-Term Memory (LSTM) network architecture to learn patterns from past game outcomes and make predictions about future game results.

## How it Works
### Training Data
The game_dataset.txt file contains a collection of past games, where each line represents a single game and comprises a sequence of integers indicating the winner of each set (0 for player A, 1 for player B, 2 for player C). This data is used to train the model to learn patterns and relationships between past winners and future outcomes.
### Neural Network Architecture
The model employed in this program is a Predictor class inheriting from PyTorch's nn.Module. It possesses the following structure:

**Input Layer:** Accepts a one-hot encoded representation of the current game sequence, with each element indicating the winner of a previous set.

**LSTM Layer:** This Long Short-Term Memory layer processes the sequential data, capturing long-term dependencies and context from past winners.

**Hidden Layer:** The LSTM layer's output is fed into a fully connected hidden layer for dimensionality reduction and feature extraction.

**Output Layer:** The final layer maps the extracted features to a probability distribution over the possible winners (player A, player B, or player C) for the upcoming set.
### Algorithm Used
The program employs the Adam optimizer and the cross-entropy loss function during the training phase. The Adam optimizer is used to update the model parameters, while the cross-entropy loss function measures the difference between predicted and actual winners of sets.
### Data Preparation
* The prepare_data1 function iterates through each game in the dataset, creating sequences of past winners (input) and the subsequent winner (output).
* One-hot encoding is applied to the input sequences to represent each winner as a binary vector.
### Training
* The train_model_linear function trains the model using the Adam optimizer and cross-entropy loss.
* It iterates through each game, considering all possible combinations of past winners from the first set to the penultimate set.
* For each combination, the one-hot encoded sequence is fed to the model, and the predicted winner is compared to the actual winner using cross-entropy loss.
* The loss is backpropagated, and the model parameters are updated to minimize the loss function.
### Prediction
* The predict_winner2 function takes the winner of the first set as input.
* It iterates through all sets after the first, feeding the sequence of winners (including the initial winner) to the model to predict the winner for the current set.
* The user is prompted to enter the actual winner, and the prediction accuracy is computed at the end.



### Output ML Model
After training, the ML model is saved to a file named "set_predictor_model.pth" using PyTorch's torch.save() function. This model can be loaded and used for making predictions without the need for retraining.

### Inputs Expected
The model expects input sequences representing the winners of previous sets in a game. Each input sequence is converted into a one-hot encoded tensor before being fed into the neural network for prediction.

### Output
* The model outputs a probability distribution over the possible winners for the upcoming set.
* The predicted winner is the player with the highest probability.

### Predicting Algorithm
The prediction algorithm iteratively predicts the winner of subsequent sets based on an initial winner provided by the user. It utilizes the trained neural network model to make predictions, and the user provides the actual winner for each set. The accuracy of the predictions is calculated based on the user-provided true winners.

### Getting Started
1. Ensure you have Python and PyTorch installed.
2. Clone the repository.
3. Prepare your game dataset and store it in a file named "game_dataset.txt".
4. Run the program using python game_set_predictor.py.
5. Follow the instructions to input the initial winner and provide the actual winners for subsequent sets.
### License
This project is licensed under the MIT License.

