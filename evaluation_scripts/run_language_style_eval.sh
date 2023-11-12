echo "Running inference with formality and sentiment classifier"
python classifier.py -if ./generated_letters/chatgpt/cbg/all_2_para_w_chatgpt.csv -of ./evaluated_letters/chatgpt/cbg --task both -m chatgpt --output_type csv
echo "Running inference with agentic vs communal classifier"
python bert_inference.py -if ./evaluated_letters/chatgpt/cbg/all_2_para_w_chatgpt.csv -m chatgpt
echo "Running t-test on inference"
python ttest.py -if ./evaluated_letters/chatgpt/cbg/all_2_para_w_chatgpt.csv 
