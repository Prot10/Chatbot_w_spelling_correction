import random
import json
import time
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


# se possibile utilizza la gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# legge il file di allenamento
with open('Files/intents.json', 'r') as json_data:
    intents = json.load(json_data)


FILE = "Files/model_data.pth"
data = torch.load(FILE)


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


# per far sembrare che il bot scriva nella print
def simulate_typing(message):
    for char in message:
        print(char, end='', flush=True)
        time.sleep(0.05)
    print('')


bot_name = "Sorridi"
simulate_typing("Chiedimi ciò che ti serve sapere! Per piacere cerca di essere specifico nelle domande.")
simulate_typing("Per chiudere la chat digita 'esci'.")


while True:
    sentence = input("Io: ")
    if sentence == "esci":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.5:
        for intent in intents['data']:
            if tag == intent["tag"]:
                print(f"{bot_name}: ", end="")
                simulate_typing(random.choice(intent['responses']))
    else:
        print(f"{bot_name}: ", end="")
        simulate_typing("Scusami non ho capito... o forse l'argomento non rientra tra le mie competenze.")
