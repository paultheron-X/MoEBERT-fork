import argparse
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering, AutoModel 

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="distilbert-base-uncased")
args = parser.parse_args()
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# transformers.logging.set_verbosity_debug()
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model)

model = "deepset/bert-base-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(model)


models_list = [
    "anirudh21/bert-base-uncased-finetuned-rte",
    "ModelTC/bert-base-uncased-cola",
    "Intel/bert-base-uncased-mrpc",
    "gchhablani/bert-base-cased-finetuned-sst2",
    "textattack/bert-base-uncased-QNLI",
    "gchhablani/bert-base-cased-finetuned-qqp",
    "textattack/bert-base-uncased-MNLI",
]

for model_ in models_list:
    tokenizer = AutoTokenizer.from_pretrained(model_, use_fast=True)
    model = AutoModel.from_pretrained(model_)
