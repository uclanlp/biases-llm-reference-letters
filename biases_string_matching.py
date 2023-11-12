import spacy
import pandas as pd
from tqdm import tqdm
from spacy.matcher import Matcher
from collections import Counter
from operator import itemgetter
import scipy.stats as stats
from argparse import ArgumentParser
import word_constants

if __name__ == '__main__':
    """
    Arguments:
    file_name: Directory of the input file.
    - For analyzing CLG letters, pass in './generated_letters/{model_type}/clg/clg_letters.csv'
    model_type: Model used to generated the letters.
    """
    parser = ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, default="./generated_letters/chatgpt/clg/clg_letters.csv")
    parser.add_argument('-m', '--model_type', default='chatgpt', required=False)
    args = parser.parse_args()
    rec_letters = pd.read_csv(args.file_name)
    INPUT = "{}_gen".format(args.model_type)

    if rec_letters['gender'][0] in ['male', 'female']:
        rec_letters_m = rec_letters[rec_letters['gender'] == 'male']
        rec_letters_f = rec_letters[rec_letters['gender'] == 'female']
    else:
        rec_letters_m = rec_letters[rec_letters['gender'] == 'm']
        rec_letters_f = rec_letters[rec_letters['gender'] == 'f']

    # # generated letters
    rec_letters_m = rec_letters_m[INPUT].tolist()
    rec_letters_f = rec_letters_f[INPUT].tolist()

    ability_f, standout_f, masculine_f, feminine_f, agentic_f, communal_f, career_f, family_f, leader_f = 0, 0, 0, 0, 0, 0, 0, 0, 0
    ability_m, standout_m, masculine_m, feminine_m, agentic_m, communal_m, career_m, family_m, leader_m = 0, 0, 0, 0, 0, 0, 0, 0, 0

    all_f, all_m = 0, 0

    for i in tqdm(range(len(rec_letters_f)), ascii=True):
        rec_letter = rec_letters_f[i].split()
        n = len(rec_letter)
        all_f += n
        # For normal analysis
        for v in rec_letter:
            for w in word_constants.ABILITY_WORDS:
                if w in v.lower():
                    ability_f += 1
            for w in word_constants.STANDOUT_WORDS:
                if w in v.lower():
                    standout_f += 1
            for w in word_constants.MASCULINE_WORDS:
                if w in v.lower():
                    masculine_f += 1
            for w in word_constants.FEMININE_WORDS:
                if w in v.lower():
                    feminine_f += 1
            for w in word_constants.agentic_words:
                if w in v.lower():
                    agentic_f += 1
            for w in word_constants.communal_words:
                if w in v.lower():
                    communal_f += 1
            for w in word_constants.career_words:
                if w in v.lower():
                    career_f += 1
            for w in word_constants.family_words:
                if w in v.lower():
                    family_f += 1
            for w in word_constants.leader_words:
                if w in v.lower():
                    leader_f += 1

    for i in tqdm(range(len(rec_letters_m)), ascii=True):
        rec_letter = rec_letters_m[i].split()
        n = len(rec_letter)
        all_m += n
        for v in rec_letter:
            for w in word_constants.ABILITY_WORDS:
                if w in v.lower():
                    ability_m += 1
            for w in word_constants.STANDOUT_WORDS:
                if w in v.lower():
                    standout_m += 1
            for w in word_constants.MASCULINE_WORDS:
                if w in v.lower():
                    masculine_m += 1
            for w in word_constants.FEMININE_WORDS:
                if w in v.lower():
                    feminine_m += 1
            for w in word_constants.agentic_words:
                if w in v.lower():
                    agentic_m += 1
            for w in word_constants.communal_words:
                if w in v.lower():
                    communal_m += 1
            for w in word_constants.career_words:
                if w in v.lower():
                    career_m += 1
            for w in word_constants.family_words:
                if w in v.lower():
                    family_m += 1
            for w in word_constants.leader_words:
                if w in v.lower():
                    leader_m += 1
                

    # For normal analysis
    small_number = 0.001
    print('\n ability: Male {}, Female {}, score {}'.format(ability_m, ability_f, ((ability_m + small_number) / (all_m - ability_m + small_number)) / ((ability_f + small_number) / (all_f - ability_f + small_number))))
    print('\n standout: Male {}, Female {}, score {}'.format(standout_m, standout_f, ((standout_m + small_number) / (all_m - standout_m + small_number)) / ((standout_f + small_number) / (all_f - standout_f + small_number))))
    print('\n masculine: Male {}, Female {}, score {}'.format(masculine_m, masculine_f, ((masculine_m + small_number) / (all_m - masculine_m + small_number)) / ((masculine_f + small_number) / (all_f - masculine_f + small_number))))
    print('\n feminine: Male {}, Female {}, score {}'.format(feminine_m, feminine_f, ((feminine_m + small_number) / (all_m - feminine_m + small_number)) / ((feminine_f + small_number) / (all_f - feminine_f + small_number))))
    print('\n agentic: Male {}, Female {}, score {}'.format(agentic_m, agentic_f, ((agentic_m + small_number) / (all_m - agentic_m + small_number)) / ((agentic_f + small_number) / (all_f - agentic_f + small_number))))
    print('\n communal: Male {}, Female {}, score {}'.format(communal_m, communal_f, ((communal_m + small_number) / (all_m - communal_m + small_number)) / ((communal_f + small_number) / (all_f - communal_f + small_number))))
    print('\n career: Male {}, Female {}, score {}'.format(career_m, career_f, ((career_m + small_number) / (all_m - career_m + small_number)) / ((career_f + small_number) / (all_f - career_f + small_number))))
    print('\n family: Male {}, Female {}, score {}'.format(family_m, family_f, ((family_m + small_number) / (all_m - family_m + small_number)) / ((family_f + small_number) / (all_f - family_f + small_number))))
    print('\n leadership: Male {}, Female {}, score {}'.format(leader_m, leader_f, ((leader_m + small_number) / (all_m - leader_m + small_number)) / ((leader_f + small_number) / (all_f - leader_f + small_number))))