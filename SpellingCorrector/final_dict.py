import json
import string
import os


# get the current path
current_dir = os.getcwd()
# get the path of the father folder
parent_dir = os.path.dirname(current_dir)
# set the wd to the father's folder
os.chdir(parent_dir)


# open both json files
with open(f"{os.getcwd()}/Files/word_counts.json", "r") as f:
    data_dict = json.load(f)
with open(f"{os.getcwd()}/Files/sorridi_website.json", "r") as f:
    data_website = json.load(f)


# keep only words that compare more than 10 times and start with a letter of the english alphabet
filtered_dict = {word: count for word, count in data_dict.items() if count >= 10 and word[0].lower() in string.ascii_lowercase}


# multiply for 10.000 the frequency of each word
modified_website = {word: count * 10000 for word, count in data_website.items()}


# join the two dictionaries
for word, count in modified_website.items():
    filtered_dict[word] = filtered_dict.get(word, 0) + count


# sort the dictionaries in alphabetical order
sorted_data = dict(sorted(filtered_dict.items()))


# save the result in a new json file
with open(f"{os.getcwd()}/Files/final_data.json", "w") as f:
    json.dump(sorted_data, f, indent=4)
