echo "Running inference with formality and sentiment classifier"
python classifier.py -if ./generated_letters/chatgpt/cbg/all_2_para_w_chatgpt-eval_hallucination.csv -of ./evaluated_letters/chatgpt/cbg --task both -m chatgpt --output_type csv --eval_hallucination_part
echo "Running inference with agentic vs communal classifier"
python bert_inference.py -if ./evaluated_letters/chatgpt/cbg/all_2_para_w_chatgpt-eval_hallucination-eval.csv -m chatgpt --eval_hallucination_part
echo "Running t-test on inference"
python ttest.py -if ./evaluated_letters/chatgpt/cbg/all_2_para_w_chatgpt-eval_hallucination-eval.csv --eval_hallucination_part
