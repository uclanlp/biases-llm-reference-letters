import os
import pandas as pd
import json
import ast
import re
from agency_generation_util import generate_response_fn
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import time

if __name__ == "__main__":
    # Configuration
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset_path', default='./agency_bios/', required=False) # './dataset/'
    args = parser.parse_args()

    dataset = os.path.join(args.dataset_path, 'BIOS_sampled.csv')
    df = pd.read_csv(dataset)
    df['raw_gen_data'] = None
    for i in tqdm(range(len(df))):
        df['raw_gen_data'][i] = generate_response_fn(df['raw_bio'][i])

    df["agentic_gen"] = None
    df["communal_gen"] = None
    for i in tqdm(range(len(df))):
        print(i)
        d = df['raw_gen_data'][i]
        data = d.split('{"agentic":')
        data = list(filter(lambda a: a != '', data))[-1]
        data = data.split('"communal":')
        data = list(filter(lambda a: a not in  ['', ',','.',';','?','!','/'], data))
        df["agentic_gen"][i], df["communal_gen"][i] = data[0], data[1]

    df.to_csv(os.path.join(args.dataset_path, 'BIOS_sampled_preprocessed.csv'))