import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from argparse import ArgumentParser

def load_from_bios(dataset_path, is_balanced):
    print("Loading data from {}".format(dataset_path))
    temp_df = pd.read_csv(dataset_path)

    # Communal: 0, Agentic: 1
    text, label = [], []

    for i in range(len(temp_df)):
        # Adding raw_bios can cause imbalance in dataset distribution (mostly agentic)
        if not is_balanced:
            text.append(temp_df.loc[i, "raw_bio"])
            label.append(
                0 if temp_df.loc[i, "chatgpt_eval"].lower() == "communal" else 1
            )
        text.append(temp_df.loc[i, "communal_bio"])
        label.append(0)
        text.append(temp_df.loc[i, "agentic_bio"])
        label.append(1)

    df = pd.DataFrame({"text": text, "label": label})
    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def split_data(df, output_path):
    print("Splitting data from {}".format(output_path))
    # Split dataframe into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Stats
    print("Train distribution split: {}".format(Counter(train_df["label"].tolist())))
    print("Val distribution split: {}".format(Counter(val_df["label"].tolist())))
    print("Test distribution split: {}".format(Counter(test_df["label"].tolist())))

    train_path = output_path + "train.csv"
    val_path = output_path + "val.csv"
    test_path = output_path + "test.csv"
    directory = os.path.dirname(train_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    # Configuration
    parser = ArgumentParser()
    parser.add_argument('-if', '--input_file', default='./agency_bios/BIOS_sampled_preprocessed.csv', required=False)
    parser.add_argument('-of', '--output_folder', default='./agency_dataset/', required=False)
    parser.add_argument('-ib', '--is_balanced', default=False, required=False)
    args = parser.parse_args()

    df = load_from_bios(args.input_file, args.is_balanced)
    split_data(df, args.output_folder)
