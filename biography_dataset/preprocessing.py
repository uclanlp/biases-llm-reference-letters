import os
import pandas as pd
import json
import ast
import re
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from preprocessing_util import generate_swapped_name_gender_fn, generate_swapped_name_gender_summarization_fn
import random
from token_counter import count_tokens
from transformers import GPT2TokenizerFast

def preprocessing_names(file_name_m, file_name_f): 
    resume_df_m = pd.read_csv(file_name_m)
    resume_df_f = pd.read_csv(file_name_f)

    m_first_names, f_first_names, last_names = [], [], []

    # separate and save male names
    m_names = resume_df_m['name'].tolist()
    for name in m_names:
        sep_names = re.sub(r'\([^)]*\)', "", ''.join(name)).split(' ')
        first, last = sep_names[0], sep_names[-1]
        m_first_names.append(first)
        last_names.append(last)

    # separate and save female names
    f_names = resume_df_f['name'].tolist()
    for name in m_names:
        sep_names = re.sub(r'\([^)]*\)', "", ''.join(name)).split(' ')
        first, last = sep_names[0], sep_names[-1]
        f_first_names.append(first)
        last_names.append(last)

    # remove empty strings and duplicates in names list
    m_first_names = list(filter(None, list(set(m_first_names))))
    f_first_names = list(filter(None, list(set(f_first_names))))
    last_names = list(filter(None, list(set(last_names))))

    return m_first_names, f_first_names, last_names

def extract_person_info(file_name, output_folder, lim_para_count=-1):
    """
    :param file_name: The input original file, e.g., "./original_bios/df_f.csv"
    :param output_folder: Path to output folder, e.g., "./sampled_bios"
    :return: corpus that contain the information from both personal life and
    career sections, this aims for the recommendation letter sections

    This function works very similar to def preprocess_csv(), just concatenate
    two different sections.

    Usage:
    extract_person_info("./original_bios/df_m.csv")
    extract_person_info("./original_bios/df_f.csv")
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    df = pd.read_csv(file_name)
    res = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        name = re.sub(r'\([^)]*\)', '', row['name'])
        first_last_name = name.split(' ')
        if len(first_last_name) > 1:
            first_name, last_name = name.split(' ')[0], name.split(' ')[-1] # [1]
        else: 
            # Pass the ones that do not have last names, e.g.,
            # https://en.wikipedia.org/wiki/Brainstormers
            continue
        person_info = row['person_info']
        if isinstance(person_info, str):
            person_info = ast.literal_eval(person_info)
        else:
            continue
        sections = [item['section'] for item in person_info]
        collected_info = ''
        career_sec, personal_sec = '', ''
        selected_section_index = []
        selected_section_type = {}
        has_personal, has_career = False, False
        for index, section_name in enumerate(sections):
            section_name = section_name.lower()
            if 'career' in section_name or 'personal' in section_name: 
                selected_section_index.append(index)
                if 'career' in section_name:
                    selected_section_type[index] = 'career'
                    has_career = True
                elif 'personal' in section_name:
                    selected_section_type[index] = 'personal'
                    has_personal = True
        if (not has_career) or (not has_personal):
            continue

        # MODIFIED: Shuffling to randomly select two paragraphs
        # random.shuffle(selected_section_index)

        personal_para_count, career_para_count = 0, 0
        for index in selected_section_index:
            all_info = person_info[index]['info']
            if lim_para_count > 0:
                para_kept = []
                para_career, para_personal = [], []
                # only keep the first two paragraphs to reduce #tokens.
                for item in all_info:
                    item = re.sub("[\[].*?[\]]", "", item)
                    item = re.sub("\n", "", item)
                    if item == "" or item == " ":
                        continue
                    if selected_section_type[index] == 'career':
                        if (career_para_count >= lim_para_count):
                            break
                        para_career.append(item)
                        if '\n' in item:
                            career_para_count += 1
                    elif selected_section_type[index] == 'personal':
                        if (personal_para_count >= lim_para_count):
                            break
                        para_personal.append(item)
                        if '\n' in item:
                            personal_para_count += 1
                    para_kept.append(item)
            else:
                para_kept = []
                for item in all_info:
                    item = re.sub("[\[].*?[\]]", "", item)
                    item = re.sub("\n", "", item)
                    if item == "" or item == " ":
                        continue
                    para_kept.append(item)
                    if selected_section_type[index] == 'career':
                        para_career.append(item)
                    elif selected_section_type[index] == 'personal':
                        para_personal.append(item)
            info = ' '.join(item for item in para_kept)
            collected_info += info

            career_sec += ' '.join(item for item in para_career)
            personal_sec += ' '.join(item for item in para_personal)

        n_tokens = count_tokens(re.sub(r'\[\d+\]', '', collected_info), debug=True)
        # Drop biographies that exceeds ChatGPT's input token limit
        if n_tokens['n_tokens'] >= 3500:
            continue
        elif len(collected_info):
            res.append({
                'name': name,
                'first_name': first_name,
                'last_name': last_name,
                'gender': row['gender'],
                # MODIFIED: Keep track of career and personal information
                'career_sec': career_sec,
                'personal_sec': personal_sec,
                'info': re.sub(r'\[\d+\]', '', collected_info), # 'info'
                'occupation': row['occupation']
            })
    if lim_para_count > 0:
        pd.DataFrame(res).to_csv('{}/processed_career_life_{}_para_{}'.format(output_folder, lim_para_count, file_name.split('/')[1]),
                                index=False)
    else:
        pd.DataFrame(res).to_csv('{}/processed_career_life_all_{}'.format(output_folder, file_name.split('/')[1]),
                            index=False)
    return


if __name__ == "__main__":
    # Configuration
    parser = ArgumentParser()
    parser.add_argument('-if', '--input_file', default='./original_bios/df_m.csv')
    parser.add_argument('-of', '--output_folder', default='./sampled_bios')
    parser.add_argument('-l', '--lim_para_count', type=int, default=2)
    args = parser.parse_args()

    extract_person_info(args.input_file, output_folder=args.output_folder, lim_para_count=args.lim_para_count)
