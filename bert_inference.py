import pandas as pd
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def calculate_acc(df, mode="test"):
    acc = 0
    for i in range(len(df)):
        row = df.iloc[i, :]
        output = classifier(row["text"])
        if row["label"] == rev_map[output[0]["label"]]:
            acc += 1
    print("Total {} accuracy: {} for model ".format(mode, acc / len(df)))

def run_inference(df, INPUT, TASK, is_sentencelevel=True):
    inferences = []
    for i in tqdm(range(len(df)), ascii=True):
        if is_sentencelevel:
            labels = []
            scores = []
            sentences = df.iloc[i, :][INPUT].split(".")
            for sentence in sentences:
                if len(sentence) >= 800:
                    continue
                output = classifier((sentence + ".").lower())[0]
                labels.append(label_mapping[TASK][rev_map[output["label"]]])
                scores.append(output["score"])
            confidence = sum(scores) / len(scores)
            mapping = Counter(labels)
            label_tracked, other_label = task_label_mapping[TASK]
            inferences.append(
                (
                    mapping[label_tracked]
                    / (mapping[label_tracked] + mapping[other_label]),
                    confidence,
                )
            )
        else:
            output = classifier(df.iloc[i, :][INPUT])[0]
            inferences.append(
                (label_mapping[TASK][rev_map[output["label"]]], output["score"])
            )

    return inferences

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./evaluated_letters/chatgpt/cbg/all_2_para_w_chatgpt-eval.csv")
    parser.add_argument("-m", "--model_type", type=str, default="chatgpt")
    parser.add_argument("-r", "--report_classifier_acc", action="store_true")
    # Evaluation on the hallucinated part of generated letters
    parser.add_argument('--eval_hallucination_part', action='store_true')
    args = parser.parse_args()

    model_path = "./checkpoints/checkpoint-48"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    rev_map = {v: k for k, v in model.config.id2label.items()}

    if args.eval_hallucination_part:
        INPUT = "hallucination"
    else:
        INPUT = "{}_gen".format(args.model_type)

    TASK = "ac_classifier"
    task_label_mapping = {
        # Track percentage agentic / percentage agentic + percentage communal
        "ac_classifier": ("agentic", "communal"),
    }
    label_mapping = {
        "ac_classifier": {
            0: "communal",
            1: "agentic",
        }
    }

    if args.report_classifier_acc:
        val_df = pd.read_csv("./agency_classifier/agency_dataset/val.csv")
        val_df = val_df.sample(frac=1).reset_index(drop=True)
        calculate_acc(val_df, mode="val")

        test_df = pd.read_csv("./agency_classifier/agency_dataset/test.csv")
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        calculate_acc(test_df, mode="test")

    sample_df = pd.read_csv(args.input_file)
    print("Running inference and outputting to: {}".format(args.input_file))
    inferences = run_inference(sample_df, INPUT, TASK)
    sample_df["per_ac"] = [i[0] for i in inferences]
    sample_df["con_ac"] = [i[1] for i in inferences]
    sample_df.to_csv(args.input_file, index=False)
