"""This file aims to produce necessary files needed for rec letters after swapping."""

import sys
import re
import random
import pandas as pd
from constants import LAST_NAMES, F_FIRST_NAMES, M_FIRST_NAMES
from generation_util import *
from tqdm import tqdm
import argparse

def choose_occupation(df, occu):
    return df[df["occupation"] == occu]

def f_to_m(string):
    string = string.lower()
    new_string = (
        string.replace(" she ", " he ")
        .replace(" her ", " his ")
        .replace(" hers ", " his ")
        .replace(" she's ", " he's ")
    )
    return new_string

def m_to_f(string):
    string = string.lower()
    new_string = (
        string.replace(" he ", " she ")
        .replace(" his ", " her ")
        .replace(" him ", " her ")
        .replace(" he's ", " she's ")
    )
    return new_string

def name_swap(df, gender, n):
    """
    df: the input dataframe. e.g., pd.read_csv('dataset/processed_career_life_df_f.csv')
    gender: gender identity of the input dataframe

    return: n*2 (two genders)*len(df) sized dataframe
    """

    res = []
    # print(f"current length: {len(df)}")
    candidate_name_ls = F_FIRST_NAMES if gender == "f" else M_FIRST_NAMES
    opposite_name_ls = M_FIRST_NAMES if gender == "f" else F_FIRST_NAMES
    for i, row in tqdm(df.iterrows(), ascii=True):
        first_name, last_name, original_info = (
            row["first_name"],
            row["last_name"],
            row["info"],
        )
        if not isinstance(last_name, str):
            last_name = ""
        if not isinstance(first_name, str):
            first_name = ""
        # First generate the same side of gender
        # Do not keep the original name to avoid pulling out info from wiki
        count = 0
        while count < n:
            replace_first_name = random.choice(candidate_name_ls)
            replace_last_name = random.choice(LAST_NAMES)
            if first_name != "":
                first_pattern = re.compile(first_name, re.IGNORECASE)
                info = first_pattern.sub(replace_first_name, original_info)
            else:
                info = original_info
            if last_name != "":
                last_pattern = re.compile(last_name, re.IGNORECASE)
                info = last_pattern.sub(replace_last_name, info)
            res.append(
                {
                    "first_name": replace_first_name,
                    "last_name": replace_last_name,
                    "gender": gender,
                    "career_sec": row["career_sec"],
                    "personal_sec": row["personal_sec"],
                    "info": info,
                    "seed_first_name": first_name,
                    "seed_last_name": last_name,
                }
            )
            count += 1

        # Then generate the opposite side of gender
        count = 0
        while count < n:
            replace_first_name = random.choice(opposite_name_ls)
            replace_last_name = random.choice(LAST_NAMES)
        
            if first_name != "":
                first_pattern = re.compile(first_name, re.IGNORECASE)
                info = first_pattern.sub(replace_first_name, original_info)
            else:
                info = original_info
            if last_name != "":
                last_pattern = re.compile(last_name, re.IGNORECASE)
                info = last_pattern.sub(replace_last_name, info)
            if gender == "f":
                info = f_to_m(info)
            else:
                info = m_to_f(info)
            new_gender = [item for item in ["f", "m"] if item != gender][0]
            res.append(
                {
                    "first_name": replace_first_name,
                    "last_name": replace_last_name,
                    "gender": new_gender,
                    "career_sec": row["career_sec"],
                    "personal_sec": row["personal_sec"],
                    "info": info,
                    "seed_first_name": first_name,
                    "seed_last_name": last_name,
                }
            )
            count += 1
    return pd.DataFrame(res)


def preprocess_dataset(occu, gend, n=-1):
    """
    :return: Ask users to choose an occupation, and then generate datasets with
    name swapping.
    """
    if gend == "f":
        df_f = pd.read_csv("./sampled_bios/processed_career_life_2_para_df_f.csv")
        df_occu = df_f[df_f["occupation"] == occu]
        if n > 0:
            df_occu = df_occu.head(n)
        swamped_df = name_swap(df_occu, "f", 1)
        swamped_df["occupation"] = occu
        swamped_df.to_csv(
            "./preprocessed_bios/df_f_{}_2_para.csv".format(occu), index=False
        )
    else:
        df_m = pd.read_csv("./sampled_bios/processed_career_life_2_para_df_m.csv")
        df_occu = df_m[df_m["occupation"] == occu]
        if n > 0:
            df_occu = df_occu.head(n)
        swamped_df = name_swap(df_occu, "m", 1)
        swamped_df["occupation"] = occu
        swamped_df.to_csv(
            "./preprocessed_bios/df_m_{}_2_para.csv".format(occu), index=False
        )
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chatgpt", help="Model type.")
    parser.add_argument(
        "--n",
        default=-1,
        help="Number of samples for each occupation for each gender.",
    )
    args = parser.parse_args()
    print(args)

    for occupation in ['acting', 'chefs', 'artists', 'dancers', 'comedians', 'models', 'musicians', 'podcasters', 'writers', 'sports']:
        # Generate name swapping data.
        for gend in ["m", "f"]:
            preprocess_dataset(occupation, gend, n=args.n)

if __name__ == "__main__":
    main()
