from datasets import load_dataset, load_metric
import datasets

glue_list = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte']

for glue in glue_list:
    load_dataset('glue', glue)


for glue in glue_list:
    load_metric('glue', glue)


# dl as well bert-base-uncased
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Path: dl_model.py
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)