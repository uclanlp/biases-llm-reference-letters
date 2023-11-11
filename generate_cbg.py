import sys
import pandas as pd
from generation_util import *
import random
import os
from tqdm import tqdm
import argparse


def chatgpt_gen(occu, gend, output_folder):
    if gend == "f":
        csv_file = f"./biography_dataset/preprocessed_bios/df_f_{occu}_2_para.csv"
    else:
        csv_file = f"./biography_dataset/preprocessed_bios/df_m_{occu}_2_para.csv"
    file_name = csv_file.split('/')[-1].split('.')[0] + '_chatgpt.csv'

    if not os.path.exists(csv_file):
        raise Exception(f"Occupation {occu} for ChatGPT has not been generated yet!")
    
    if occu == "acting":
        real_occupation = "actor"
    else:
        real_occupation = occu.rstrip("s")

    df = pd.read_csv(csv_file)
    if "info" not in list(df.columns) or "first_name" not in list(df.columns):
        raise Exception("info and name must be in df's columns.")
    df["chatgpt_gen"] = -1

    for i, row in tqdm(df.iterrows(), ascii=True):
        pronoun = "him" if row["gender"] == "m" else "her"
        generated_response = generate_response_rec_chatgpt(
            {
                "occupation": real_occupation,
                "name": "{} {}".format(row["first_name"], row["last_name"]),
                "pronoun": pronoun,
                "info": row["info"],
            }
        )
        generated_response = generated_response.replace("\n", "<return>")
        df["chatgpt_gen"][i] = generated_response
    df.to_csv(output_folder + '/' + file_name)
    return


def model_gen(occu, gend, model_type, output_folder):
    if gend == "f":
        csv_file = f"./biography_dataset/preprocessed_bios/df_f_{occu}_2_para.csv"
    else:
        csv_file = f"./biography_dataset/preprocessed_bios/df_m_{occu}_2_para.csv"
    file_name = csv_file.split('/')[-1].split('.')[0] + '_{}.csv'.format(model_type)

    if not os.path.exists(csv_file):
        raise Exception(f"Occupation {occu} for Model {model_type} has not been generated yet!")
    
    if occu == "acting":
        real_occupation = "actor"
    else:
        real_occupation = occu.rstrip("s")

    if model_type == "alpaca":
        tokenizer = LlamaTokenizer.from_pretrained(
            "chavinlo/alpaca-native", model_max_length=1024
        )
        model = LlamaForCausalLM.from_pretrained("chavinlo/alpaca-native")

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.half().to(device)

    elif model_type == "vicuna":
        tokenizer = LlamaTokenizer.from_pretrained("/local/elaine1wan/vicuna")
        model = LlamaForCausalLM.from_pretrained("/local/elaine1wan/vicuna")

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.half().to(device)

    elif model_type == "stablelm":
        tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b")
        model = AutoModelForCausalLM.from_pretrained(
            "StabilityAI/stablelm-tuned-alpha-7b"
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.half().to(device)
        # model.to(device)

    elif model_type == "falcoln":
        model = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.half().to(device)
    else:
        raise NotImplementedError

    df = pd.read_csv(csv_file)
    if "info" not in list(df.columns) or "first_name" not in list(df.columns):
        raise Exception("info and name must be in df's columns.")
    df["{}_gen".format(model_type)] = -1
    print("Total generations: {}".format(len(df)))

    write_amount = 0
    for i, row in tqdm(df.iterrows(), ascii=True):
        # print(i)
        # if i < 3470: continue
        pronoun = "him" if row["gender"] == "m" else "her"
        if model_type == "alpaca":
            generated_response = generate_response_rec_alpaca(
                {
                    "occupation": real_occupation,
                    "name": "{} {}".format(row["first_name"], row["last_name"]),
                    "pronoun": pronoun,
                    "info": row["info"],
                },
                model,
                tokenizer,
                device,
            )
        elif model_type == "vicuna":
            generated_response = generate_response_rec_vicuna(
                {
                    "occupation": real_occupation,
                    "name": "{} {}".format(row["first_name"], row["last_name"]),
                    "pronoun": pronoun,
                    "info": row["info"],
                },
                model,
                tokenizer,
                device,
            )
        elif model_type == "stablelm":
            generated_response = generate_response_rec_stablelm(
                {
                    "occupation": real_occupation,
                    "name": "{} {}".format(row["first_name"], row["last_name"]),
                    "pronoun": pronoun,
                    "info": row["info"],
                },
                model,
                tokenizer,
                device,
            )
        elif model_type == "falcoln":
            generated_response = generate_response_rec_falcon(
                {
                    "occupation": real_occupation,
                    "name": "{} {}".format(row["first_name"], row["last_name"]),
                    "pronoun": pronoun,
                    "info": row["info"],
                },
                model,
                tokenizer,
                device,
            )
        generated_response = generated_response.replace("\n", "<return>")
        df["{}_gen".format(model_type)][i] = generated_response
        write_amount += 1
    print("Number of generated samples: {}".format(write_amount))
    df.to_csv(output_folder + '/' + file_name)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chatgpt", help="Model type.")
    parser.add_argument(
        "--n",
        default=-1,
        help="Number of samples for each occupation for each gender.",
    )
    parser.add_argument('-of', '--output_folder', default='./generated_letters/chatgpt/cbg')
    args = parser.parse_args()
    print(args)

    for occupation in ['acting', 'chefs', 'artists', 'dancers', 'comedians', 'models', 'musicians', 'podcasters', 'writers', 'sports']:
        for gend in ["m", "f"]:
            if args.model == "chatgpt":
                chatgpt_gen(occupation, gend, args.output_folder)
            else:
                model_gen(occupation, gend, args.model, args.output_folder)

if __name__ == "__main__":
    main()
