import spacy
from spacy.matcher import Matcher
from collections import Counter
from operator import itemgetter
import pandas as pd
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-df', '--data_folder', default='./generated_letters/chatgpt/cbg')
    parser.add_argument('-m', '--model_type', default='chatgpt')
    parser.add_argument('-t', '--task', type=str, default="merge", help="either merge or filter")
    args = parser.parse_args()

    if args.task == "merge":
        dfs = []
        for occupation in ['acting', 'chefs', 'artists', 'dancers', 'comedians', 'models', 'musicians', 'podcasters', 'writers', 'sports']:
            dfs.append(pd.read_csv(args.data_folder + '/' + 'df_f_{}_2_para_w_{}.csv'.format(occupation, args.model)))
        rec_letters = pd.concat(dfs)
        rec_letters.to_csv(args.data_folder + '/' + 'all_2_para_w_{}.csv'.format(args.model))

    elif args.task == "filter":
        # Calculate Success Rate and Keeping Only Successful Generations
        rec_letters = pd.read_csv(args.data_folder + '/' + 'all_2_para_w_{}.csv'.format(args.model))
        success = 0
        ct = 0
        idxs = []
        for i in range(len(rec_letters)):
            text = rec_letters["{}_gen".format(args.model)][i]
            ct += 1
            if (str(text) != 'nan') and (" recommend " in text):
                sentences = text.split(".")
                if (len(sentences) == 1) and (len(sentences[0]) >= 800):
                    continue
                else:
                    success += 1
                    idxs.append(i)
        print('\n Success Rate: ', success / ct)
        rec_letters = rec_letters.loc[idxs]
        rec_letters.to_csv(args.data_folder + '/' + 'all_2_para_w_{}_success.csv'.format(args.model), index=False)
