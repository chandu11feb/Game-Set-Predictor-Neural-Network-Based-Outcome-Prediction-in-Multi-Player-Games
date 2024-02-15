"""
Created on Wed Feb 14 13:43:29 2024

@author: chandu11feb
"""

import torch
import torch.nn as nn
from datetime import datetime



NUM_PLAYERS = 3
NUM_SETS = 70


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
def predict_winner2(model, game_results_list):
    # Initialize the sequence with the initial winner
    sequence = [int(game_results_list[0])]  # Ensure integer type
    correct_predictions = 0
    predictedList =[]
    winningStreak = 0
    maxWinStreak =0
    maxLoseStreak =0
    losingStreak = 0
    winLoseList = []

    # Iterate through each set
    for i in range(1,len(game_results_list)):
        # Print the current sequence
        # print("sequence ", sequence)

        # Convert sequence into one-hot encoding for input to the model
        input_data = torch.nn.functional.one_hot(torch.tensor(sequence).long().to(device), num_classes=3)
        # print("input_data  ", input_data)

        # Pass input data through the model to get predictions
        with torch.no_grad():
            output = model(input_data.float())
        # print(output)

        # Get the predicted winner for the current set
        predicted_winner = torch.argmax(output).item()
        predictedList.append(str(predicted_winner))
        # print(f'Predicted winner for set {i + 1}: Player {predicted_winner}')

        # Prompt user to input the actual winner for the set
        true_winner = int(game_results_list[i])

        # Check if the predicted winner matches the actual winner
        if predicted_winner == true_winner:
            correct_predictions += 1
            winLoseList.append("W")
            winningStreak +=1
            losingStreak =0
            if maxWinStreak < winningStreak:
                maxWinStreak = winningStreak

        else:
            winLoseList.append("L")
            winningStreak =0
            losingStreak +=1
            if maxLoseStreak < losingStreak:
                maxLoseStreak = losingStreak

        # Add the true winner to the sequence for the next iteration
        sequence.append(int(true_winner))  # Ensure integer type

    # Calculate and print the accuracy for the game
    accuracy = correct_predictions / (NUM_SETS-1)
    accuracy = accuracy*100

    # print(f'Accuracy for this game: {accuracy } %')
    return accuracy , "".join(predictedList) , maxWinStreak , maxLoseStreak , correct_predictions , NUM_SETS-1-correct_predictions , "".join(winLoseList)

def generate_number_from_time():
    now = datetime.now()
    formatted_time = now.strftime("%H%M%S")
    return int(formatted_time)

def append_to_file(file_name, content):
    with open(file_name, 'a') as file:
        file.write(content + '\n')

def ModelValidator(validFile , storeGameExcel , storeModelResultsExcel , model ,modelPath ):
    accuracyList = []
    predictedList = []
    winStreakList = []
    loseStreakList = []
    winsCountList = []
    loseCountList = []
    winLoseList = []


    with open(validFile, 'r') as file:
        game_data = [[int(winner) for winner in line.strip()] for line in file]
    for game in game_data:
        accuracy, predictedListStr, maxWinStreak, maxLoseStreak, winCount, LoseCount , winLoseListStr = predict_winner2(model , game)
        accuracyList.append(accuracy)
        predictedList.append(predictedListStr)
        winStreakList.append(maxWinStreak)
        loseStreakList.append(maxLoseStreak)
        winsCountList.append(winCount)
        loseCountList.append(LoseCount)
        winLoseList.append(winLoseListStr)


    avgAccuracy = sum(accuracyList)/len(accuracyList)
    avgWinCount = sum(winsCountList)/len(winsCountList)
    avgLoseCount = sum(loseCountList)/len(loseCountList)
    avgWinSteak = sum(winStreakList)/len(winStreakList)
    avgLoseStreak =sum(loseStreakList)/len(loseStreakList)

    uniqueNumber =  generate_number_from_time()


    storeModelResultsStr = f" {uniqueNumber}  {modelPath}  {avgAccuracy}  {avgWinCount}          {avgLoseCount}          {avgWinSteak}          {avgLoseStreak} "
    print("serialNo    ModelName                  gameNo    Accuracy             PredictedList                                                         PredHealth  winCount  loseCount  winStreak loseStreak  winLossList")
    for i in range(len(accuracyList)):
        sequenceHealth = None
        storeGameStr = f" {uniqueNumber}  {modelPath}    {i+1}     {accuracyList[i]}  {predictedList[i]}  {sequenceHealth}        {winsCountList[i]}        {loseCountList[i]}         {winStreakList[i]}         {loseStreakList[i]}           {winLoseList[i]} "
        print(storeGameStr)
        append_to_file(storeGameExcel,storeGameStr)
    print("serialNo   ModelName                     AverageAccuracy   AvgWinCount  AvgLoseCount  AvgWinStreak  AvgLoseStreak")
    print(storeModelResultsStr)
    append_to_file(storeModelResultsExcel, storeModelResultsStr)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "set_predictor_model_128te.pth"
model = torch.load(model_path, map_location=device)
model.eval()


storeGameExcel = "storeGameExcel.txt"
storeModelResultsExcel = "storeModelResultsExcel.txt"
validationDataFile = "validation_dataset.txt"

ModelValidator(validationDataFile,storeGameExcel,storeModelResultsExcel,model,model_path)




