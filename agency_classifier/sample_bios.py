import pickle
import random
import pandas as pd

file_name = "../dataset/BIOS.pkl"
output_path = "../dataset/BIOS_sampled_2.csv"
MAX_COUNT = 50
with open(file_name, "rb") as f:
    # Load the contents of the file
    contents = pickle.load(f)
    random.shuffle(contents)

# Iterate over the contents line by line
saved_data = {
    "name": [],
    "gender": [],
    "raw_title": [],
    "raw_bio": [],
    "title": [],
    "bio": [],
}
male_count = 0
female_count = 0

for line in contents:
    if line["gender"] == "M":
        if male_count == MAX_COUNT:
            continue
        male_count += 1
    else:
        if female_count == MAX_COUNT:
            continue
        female_count += 1

    saved_data["name"].append(
        line["name"][0] + " " + line["name"][2]
        if line["name"][1] == ""
        else " ".join(line["name"])
    )
    saved_data["gender"].append(line["gender"])
    saved_data["raw_title"].append(line["raw_title"])
    saved_data["title"].append(line["title"])
    saved_data["raw_bio"].append(line["raw"])
    saved_data["bio"].append(line["bio"])

data = pd.DataFrame(saved_data)
data.to_csv(output_path, index=False)
