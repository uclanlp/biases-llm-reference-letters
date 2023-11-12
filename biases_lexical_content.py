import spacy
from spacy.matcher import Matcher
from collections import Counter
from operator import itemgetter
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
from argparse import ArgumentParser

def calculate_dict(female_array, male_array):
    counter_f_h = Counter(female_array)
    counter_m_h = Counter(male_array)
    # make sure there is no key lookup error
    for key in set(counter_f_h) - set(counter_m_h):
        counter_m_h[key] = 0
    for key in set(counter_m_h) - set(counter_f_h):
        counter_f_h[key] = 0
    return counter_f_h, counter_m_h

def odds_ratio(f_dict, m_dict, topk=50, threshold=20):
    very_small_value = 0.00001
    if len(f_dict.keys()) != len(m_dict.keys()):
        raise Exception('The category for analyzing the male and female should be the same!')
    else:
        odds_ratio = {}
        total_num_f = sum(f_dict.values())
        total_num_m = sum(m_dict.values())
        for key in f_dict.keys():
            m_num = m_dict[key]
            f_num = f_dict[key]
            non_f_num = total_num_f - f_num
            non_m_num = total_num_m - m_num
            if f_num >= threshold and m_num >= threshold:
                # we only consider the events where there are at least {thresohld} occurences for both gender
                odds_ratio[key] = round((m_num / f_num) / (non_m_num / non_f_num), 2)
            else:
                continue
        return dict(sorted(odds_ratio.items(), key=itemgetter(1), reverse=True)[:topk]), dict(
            sorted(odds_ratio.items(), key=itemgetter(1))[:topk])

class Word_Extraction:
    def __init__(self, word_types=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        patterns = []

        for word_type in word_types:
            if word_type == 'noun':
                patterns.append([{'POS':'NOUN'}])
            elif word_type == 'adj':
                patterns.append([{'POS':'ADJ'}])
            elif word_type == 'verb':
                patterns.append([{"POS": "VERB"}])
        self.matcher.add("demo", patterns)

    def extract_word(self, doc):
        doc = self.nlp(doc)
        matches = self.matcher(doc)
        vocab = []
        for match_id, start, end in matches:
            string_id = self.nlp.vocab.strings[match_id]  # Get string representation
            span = doc[start:end]  # The matched span
            vocab.append(span.text)
        return vocab

if __name__ == '__main__':
    """
    Arguments:
    file_name: Directory of the input file.
    - For analzing original CBG letters, pass in './generated_letters/{model_type}/cbg/all_2_para_w_{model_type}.csv'
    - For analzing filtered (only successful generations) CBG letters, pass in './generated_letters/{model_type}/cbg/all_2_para_w_{model_type}_success.csv'
    - For analzing hallucinated parts of CBG letters, pass in './generated_letters/{model_type}/cbg/all_2_para_w_{model_type}-eval_hallucination.csv'
    model_type: Model used to generated the letters.
    - Options: chatgpt, alpaca, vicuna, stablelm
    """
    parser = ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, default="./generated_letters/chatgpt/cbg/all_2_para_w_chatgpt.csv")
    parser.add_argument('-m', '--model_type', default='chatgpt', required=False)
    parser.add_argument('-t', '--threshold', default=10, required=False)
    parser.add_argument('-s', '--save_output', action='store_true')
    parser.add_argument('-o', '--output_folder', default=None, required=False)
    args = parser.parse_args()
    rec_letters = pd.read_csv(args.file_name)
    if rec_letters['gender'][0] in ['male', 'female']:
        rec_letters_m = rec_letters[rec_letters['gender'] == 'male']
        rec_letters_f = rec_letters[rec_letters['gender'] == 'female']
    else:
        rec_letters_m = rec_letters[rec_letters['gender'] == 'm']
        rec_letters_f = rec_letters[rec_letters['gender'] == 'f']

    INPUT = "{}_gen".format(args.model_type)

    # generated letters
    rec_letters_m = rec_letters_m[INPUT].tolist()
    rec_letters_f = rec_letters_f[INPUT].tolist()

    noun_f, noun_m = [], []
    adj_f, adj_m = [], []
    len_f, len_m = [], []

    noun_extract = Word_Extraction(['noun'])
    adj_extract = Word_Extraction(['adj'])
    ability_m, standout_m, ability_f, standout_f = 0, 0, 0, 0
    masculine_m, feminine_m, masculine_f, feminine_f = 0, 0, 0, 0
    for i in tqdm(range(len(rec_letters_f)), ascii=True):
        noun_vocab_f = noun_extract.extract_word(rec_letters_f[i])
        # For normal analysis
        for v in noun_vocab_f:
            v = v.split()[0].replace('<return>', '').replace('<return', '').strip(',./?').lower()
            noun_f.append(v)
        
        adj_vocab_f = adj_extract.extract_word(rec_letters_f[i])
        for v in adj_vocab_f:
            v = v.split()[0].replace('<return>', '').replace('<return', '').strip(',./?').lower()
            adj_f.append(v)


    for i in tqdm(range(len(rec_letters_m)), ascii=True):
        noun_vocab_m = noun_extract.extract_word(rec_letters_m[i])
        # For normal analysis
        for v in noun_vocab_m:
            v = v.split()[0].replace('<return>', '').replace('<return', '').strip(',./?').lower()
            noun_m.append(v)
        
        adj_vocab_m = adj_extract.extract_word(rec_letters_m[i])
        for v in adj_vocab_m:
            v = v.split()[0].replace('<return>', '').replace('<return', '').strip(',./?').lower()
            adj_m.append(v)

    # For normal analysis
    noun_counter_f, noun_counter_m = calculate_dict(noun_f, noun_m)
    noun_res_m, noun_res_f = odds_ratio(noun_counter_f, noun_counter_m, threshold=args.threshold)
    print('\n noun male', noun_res_m.keys(), '\n noun female', noun_res_f.keys())
    adj_counter_f, adj_counter_m = calculate_dict(adj_f, adj_m)
    adj_res_m, adj_res_f = odds_ratio(adj_counter_f, adj_counter_m, threshold=args.threshold)
    print('\n adj male', adj_res_m.keys(), '\n adj female', adj_res_f.keys())

    if args.save_output:
        output = {}
        output['noun_male'] = list(noun_res_m.keys())
        output['noun_female'] = list(noun_res_f.keys())
        output['adj_male'] = list(adj_res_m.keys())
        output['adj_female'] = list(adj_res_f.keys())
        df = pd.DataFrame.from_dict(output)
        df.to_csv(args.output_folder + '/' + '{}_lexical_content.csv'.formate(args.model_type))