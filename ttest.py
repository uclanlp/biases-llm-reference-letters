import scipy.stats as stats
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-if", "--input_file", type=str, default="./generated_letters/chatgpt/cbg/all_2_para_w_chatgpt-eval.csv")
    # Evaluation on the hallucinated part of generated letters
    parser.add_argument('--eval_hallucination_part', action='store_true')
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    if df['gender'][0] in ['male', 'female']:
        df_m = df[df['gender'] == 'male']
        df_f = df[df['gender'] == 'female']
    else:
        df_m = df[df['gender'] == 'm']
        df_f = df[df['gender'] == 'f']

    for inference in ["per_pos", "per_for", "per_ac"]:
        if not args.eval_hallucination_part:
            per_f = df_f[inference].tolist()    
            per_m = df_m[inference].tolist()

            res = stats.ttest_ind(a=per_m, b=per_f, equal_var=True, alternative='greater')
            statistic, pvalue = res[0], res[1]
            print("Inference type: {}\nStatistic: {}\nP-value: {}".format(inference, statistic, pvalue))

        if args.eval_hallucination_part:
            hal_f = df_f[inference].tolist()   
            ori_f = df_f['{}_1'.format(inference)].tolist()
            hal_m = df_m[inference].tolist()
            ori_m = df_m['{}_1'.format(inference)].tolist()

            res1 = stats.ttest_ind(a=hal_m, b=ori_m, equal_var=True, alternative='greater')
            statistic1, pvalue1 = res1[0], res1[1]
            print("Inferencing hallucinated vs. original contents of Male generated letters. Inference type: {}\nStatistic: {}\nP-value: {}".format(inference, statistic1, pvalue1))

            res2 = stats.ttest_ind(a=ori_f, b=hal_f, equal_var=True, alternative='greater')
            statistic2, pvalue2 = res2[0], res2[1]
            print("Inferencing original vs. hallucinated contents of Female generated letters. Inference type: {}\nStatistic: {}\nP-value: {}".format(inference, statistic2, pvalue2))
