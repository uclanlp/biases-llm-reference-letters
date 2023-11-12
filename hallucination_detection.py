import re
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("if", "--input_file", type=str, default="./generated_letters/chatgpt/cbg/all_2_para_w_chatgpt-eval.csv")
    parser.add_argument("-ml", "--max_length", type=int, default=256)
    parser.add_argument("-m", "--model_type", type=str, default="chatgpt")
    args = parser.parse_args()

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
            # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
            # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    device_ids = [i for i in range(4)]
    model = nn.DataParallel(model, device_ids)

    df = pd.read_csv(args.input_file)
    cols = list(df.columns)[1:]
    for i in range(len(cols)):
        # 'per_pos' -> 'per_pos_1'
        if 'per_' in cols[i] or 'con_' in cols[i]:
            cols[i] = cols[i] + '_1'
    df = df[[cols]]
    df['hallucination'] = ''
    df['contradiction'] = ''

    for i in tqdm(range(len(df)), ascii=True):
        premise = df['info'][i]
        hypotheses = re.split(r"\.|\?|\!",df['{}_gen'.format(args.model_type)][i].replace('<return>', ''))
        l = len(hypotheses)
        for j in range(len(hypotheses)):
            hypothesis = hypotheses[j]
            tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis.format(df['first_name'][i]),
                                                            max_length=args.max_length,
                                                            return_token_type_ids=True, truncation=True)

            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(device)
            # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(device)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(device)

            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=None)
            # Note:
            # "id2label": {
            #     "0": "entailment",
            #     "1": "neutral",
            #     "2": "contradiction"
            # },

            predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist() 

            m = max(predicted_probability)
            if (m == predicted_probability[1]) or (m == predicted_probability[2]):
                df['hallucination'][i] = df['hallucination'][i] + hypothesis + '. '
                if (m == predicted_probability[2]):
                    df['contradiction'][i] = df['contradiction'][i] + hypothesis + '. '
        
    df.to_csv(args.input_file.replace('.csv', '_hallucination.csv'))
                


