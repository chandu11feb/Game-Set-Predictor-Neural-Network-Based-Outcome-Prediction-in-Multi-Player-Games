import random
import shutil
import time

def read_file_to_list(filename):

    lines = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    print("Dataset Trained ",filename," Consists Data of Total",len(lines),"games")
    return lines

def copy_and_paste_for_dataset_backup(source_file, destination_file):
    shutil.copyfile(source_file, destination_file)
    print(f"File '{source_file}' copied to '{destination_file}'")

def clear_file(filename):
    with open(filename, 'w') as file:
        pass
    print(f"File '{filename}' cleared")

def append_files(source_file, target_file):

  with open(source_file, "r") as source:
    data = source.read()

  with open(target_file, "a") as target:
    target.write(data+'\n')

filenameDataset = 'datasetTrained.txt'
DatasetFilelines = read_file_to_list(filenameDataset)
def write_string_to_file(filename="newGamesDataset.txt",DatasetFilelines=[], content=""):


    with open(filename, 'a') as file:
        if content not in DatasetFilelines:
            file.write(content+'\n')
def cards_sum(lis):
    total=0
    for i in lis:
        last_digit=int(i[-1])
        total+=last_digit
    if total>9:
        return total%10
    return total

def create_cards_deck(deckSize=8):
    cards_deck=[]
    clubs=["♣️A01","♣️️02","♣️️03","♣️04","♣️️05","♣️06","♣️️07","♣️️08","♣️️09","♣️️10","♣️️J10","♣️️Q10","♣️️K10"]
    diamonds=["♦️A01","♦️02","♦️03","♦️04","♦️05","♦️06","♦️07","♦️08","♦️09","♦️10","♦️J10","♦️Q10","♦️K10"]
    hearts=["❤️A01","❤️02","❤️03","❤️️04","❤️️05","❤️06","❤️07","❤️️08","❤️09","❤️️10","❤️️J10","❤️️Q10","❤️️K10"]
    spades=["♠️A01","♠️️02","♠️️03","♠️04","♠️️05","♠️️06","♠️️07","♠️️08","♠️️09","♠️️10","♠️️J10","♠️️Q10","♠️️K10"]
    for _ in range(deckSize):
        cards_deck.extend(clubs)
        random.shuffle(cards_deck)
        cards_deck.extend(diamonds)
        random.shuffle(cards_deck)
        cards_deck.extend(hearts)
        random.shuffle(cards_deck)
        cards_deck.extend(spades)
        random.shuffle(cards_deck)
    # print(cards_deck)
    # print("Cards deck length",len(cards_deck))
    for _ in range(3):
        random.shuffle(cards_deck)
    return cards_deck

def remove_face_card_number(card_deck_list):
    face_card=card_deck_list.pop(0)
    face_card_number=int(face_card[-2:])
    if face_card_number==10:
        card_face = face_card[-3]
        if card_face =="J":
            face_card_number = 11
        elif card_face =="Q":
            face_card_number = 12
        elif card_face == "K":
            face_card_number = 13
        else:
            face_card_number = 10
    card_deck_list=card_deck_list[face_card_number:]
    return card_deck_list

def banker_draw_card(banker_hand, player_third_card):
    """
      This function takes the Banker's hand value and the Player's third card value as input and returns True if the Banker should draw a card, False otherwise.

      Args:
          banker_hand: An integer representing the Banker's hand value (0-9).
          player_third_card: An integer representing the Player's third card value (0-9).

      Returns:
          A boolean value indicating whether the Banker should draw a card (True) or not (False).
      """

    # Banker always draws with 0, 1, or 2
    if banker_hand <= 2:
        return True

    # Banker draws with 3 based on player's third card
    if banker_hand == 3:
        return player_third_card not in [8, 9, 10]

    # Banker draws with 4 based on player's third card
    if banker_hand == 4:
        return player_third_card not in [0,1, 8, 9, 10]

    # Banker draws with 5 based on player's third card
    if banker_hand == 5:
        return player_third_card in [4, 5, 6, 7]

    # Banker draws with 6 based on player's third card
    if banker_hand == 6:
        return player_third_card in [6, 7]

    # Banker never draws with 7
    if banker_hand == 7:
        return False

    # Invalid hand values
    raise ValueError("Invalid Banker hand value or Player third card value.")


def start_game_and_get_results_string(cards_deck):
    results_str = ""
    game_count =0
    deckSize = len(cards_deck)//52
    if deckSize==7:
        game_count_limit = 70
    else:
        game_count_limit = (deckSize)*10
    while (game_count < game_count_limit):
        game_count += 1
        Banker = []
        Player = []
        x = cards_deck.pop(0)
        Player.append(x)
        x = cards_deck.pop(0)
        Banker.append(x)
        x = cards_deck.pop(0)
        Player.append(x)
        x = cards_deck.pop(0)
        Banker.append(x)
        if cards_sum(Player) <= 7 and cards_sum(Banker) <= 7:
            player_third_card = 0
            if cards_sum(Player) <= 5:
                player_third_card = cards_deck.pop(0)
                Player.append(player_third_card)
                player_third_card = int(player_third_card[-2:])
            Banker_sump = cards_sum(Banker)

            if banker_draw_card(Banker_sump,player_third_card):
                if cards_sum(Banker) <= 5:
                    x = cards_deck.pop(0)
                    Banker.append(x)

        Banker_sum = cards_sum(Banker)
        Player_sum = cards_sum(Player)

        if Banker_sum > Player_sum:
            results_str+="0"
        elif Banker_sum < Player_sum:
            results_str+="1"
        else:
            results_str+="2"
    return results_str

newGamesDatasetFile = "newGamesDataset.txt"
currentFileList = read_file_to_list(newGamesDatasetFile)
def cards_pipeline(cards_deck):
    for i in range(len(cards_deck)):
        reshapedDeck = cards_deck[i:]+cards_deck[:i]
        result_str = start_game_and_get_results_string(remove_face_card_number(reshapedDeck))
        if result_str not in currentFileList:
            currentFileList.append(result_str)
            write_string_to_file(content=result_str)
    cards_deck.reverse()
    for i in range(len(cards_deck)):
        reshapedDeck = cards_deck[i:]+cards_deck[:i]
        result_str = start_game_and_get_results_string(remove_face_card_number(reshapedDeck))
        if result_str not in currentFileList:
            currentFileList.append(result_str)
            write_string_to_file(content=result_str)

populateNumber = 9999
datasetTrainedFile = "datasetTrained.txt"
datasetTrainedBackupFile = "datasetTrainedBackup.txt"
newGamesDatasetFile = "newGamesDataset.txt"
deckSize = 8

newDataTrained = False

if newDataTrained:
    copy_and_paste_for_dataset_backup(datasetTrainedFile, datasetTrainedBackupFile)
    append_files(newGamesDatasetFile, datasetTrainedFile)
    copy_and_paste_for_dataset_backup(datasetTrainedFile, datasetTrainedBackupFile)
    clear_file(newGamesDatasetFile)

print(f'Before populating data : {time.ctime()}')
for _ in range(populateNumber):
    cards_pipeline(create_cards_deck(deckSize))
print("Data collected of",len(currentFileList),"games")
print(f'Before populating data : {time.ctime()}')

