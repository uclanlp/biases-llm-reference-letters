# Run example: `python3 classifier.py --task formality`

import pandas as pd
from transformers import pipeline
import os
from argparse import ArgumentParser
from collections import Counter

PATH_TO_COLS = {
    "df_f_acting_2_para_w_chatgpt.csv": ("gender", "chatgpt_gen"),
}
# (label tracked, other labels)
task_label_mapping = {
    "sentiment": ("POSITIVE", "NEGATIVE"),
    # "sentiment": ("positive", "neutral", "negative"),
    "formality": ("formal", "informal"),
}


# Define a function to perform sentiment analysis on each row of the dataframe
def predict(text, classifier, task, output_type, is_sentencelevel=True):
    if is_sentencelevel:
        labels = []
        scores = []
        text = text
        sentences = text.split(".")
        for sentence in sentences:
            if len(sentence) >= 800:
                continue
            result = classifier((sentence + "."))[0]
            labels.append(result["label"])
            scores.append(result["score"])
        confidence = sum(scores) / len(scores)

        if output_type == "csv":
            mapping = Counter(labels)
            label_tracked, other_label = task_label_mapping[task]
            return (
                mapping[label_tracked]
                / (mapping[label_tracked] + mapping[other_label]),
                confidence,
            )
        # Get the most common word
        return max(set(labels), key=labels.count), confidence
    result = classifier(text)[0]
    return result["label"], result["score"]


def print_helper(outputs, task_type):
    print("Current task: {}".format(task_type))
    # Calculate the average sentiment score
    avg_output = sum([s[1] for s in outputs]) / len(outputs)
    male_outputs = []
    female_outputs = []
    for i in range(len(df)):
        if df.loc[i, "gender"].lower() in ["m", "male"]:
            male_outputs.append(outputs[i])
        else:
            female_outputs.append(outputs[i])
    male_output = sum([s[1] for s in male_outputs]) / len(male_outputs)
    female_output = sum([s[1] for s in female_outputs]) / len(female_outputs)
    print(
        "Average Score: {}\nMin score: {}\nMax score: {}".format(
            avg_output,
            min([s[1] for s in outputs]),
            max([s[1] for s in outputs]),
        )
    )
    print(Counter([s[0] for s in outputs]))
    print(
        "Average Male Score: {}\nMin Male score: {}\nMax Male score: {}".format(
            male_output,
            min([s[1] for s in male_outputs]),
            max([s[1] for s in male_outputs]),
        )
    )
    print(Counter([s[0] for s in male_outputs]))
    print(
        "Average Female Score: {}\nMin Female score: {}\nMax Female score: {}".format(
            female_output,
            min([s[1] for s in female_outputs]),
            max([s[1] for s in female_outputs]),
        )
    )
    print(Counter([s[0] for s in female_outputs]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("if", "--input_file", type=str, default="./generated_letters/chatgpt/cbg/all_2_para_w_chatgpt.csv")
    parser.add_argument("of", "--output_folder", type=str, default="./evaluated_letters/chatgpt/cbg")
    # [formality, sentiment, both]
    parser.add_argument("--task", type=str, default="both")
    parser.add_argument("-m", "--model_type", type=str, default="chatgpt")
    parser.add_argument("--output_type", type=str, default="print")
    # Evaluation on the hallucinated part of generated letters
    parser.add_argument('--eval_hallucination_part', action='store_true')
    args = parser.parse_args()

    # Load data
    data_path = args.input_file
    file_name = args.input_file.split('/')[-1].split('.')[0]
    if args.eval_hallucination_part:
        INPUT = "hallucination"
    else:
        INPUT = "{}_gen".format(args.model_type)

    print("Evaluating {} on {}".format(args.task, data_path))
    df = pd.read_csv(data_path)
    if args.task == "sentiment" or args.task == "both":
        # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english?text=I+like+you.+I+love+you
        classifier_sentiment = pipeline("sentiment-analysis")
        # classifier_sentiment = pipeline("sentiment-analysis",model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    if args.task == "formality" or args.task == "both":
        # https://huggingface.co/s-nlp/xlmr_formality_classifier
        classifier_formality = pipeline(
            "text-classification", "s-nlp/roberta-base-formality-ranker"
        )
        # classifier = pipeline('text-classification', model='s-nlp/xlmr_formality_classifier', tokenizer='s-nlp/xlmr_formality_classifier')

    # Apply the sentiment analysis function to each row of the dataframe
    sentiment_outputs = None
    formality_outputs = None
    if args.task == "formality" or args.task == "both":
        formality_outputs = df[INPUT].apply(
            (lambda x: predict(x, classifier_formality, "formality", args.output_type))
        )
    if args.task == "sentiment" or args.task == "both":
        sentiment_outputs = df[INPUT].apply(
            (lambda x: predict(x, classifier_sentiment, "sentiment", args.output_type))
        )

    if args.output_type == "print":
        if sentiment_outputs is not None:
            print_helper(sentiment_outputs, task_type="sentiment")
        if formality_outputs is not None:
            print_helper(formality_outputs, task_type="formality")
    elif args.output_type == "csv":
        if sentiment_outputs is not None:
            df["per_pos"] = [s[0] for s in sentiment_outputs]
            df["con_pos"] = [s[1] for s in sentiment_outputs]
        if formality_outputs is not None:
            df["per_for"] = [s[0] for s in formality_outputs]
            df["con_for"] = [s[1] for s in formality_outputs]
        output_path = args.output_folder + '/' + file_name + '-eval.csv'
        print("Finished output of percent/confidence to {}".format(output_path))
        df.to_csv(output_path, index=False)
