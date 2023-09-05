
conda activate MoEBERT
# Make Hugging Face cache folder on drive
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
mkdir -p $HF_LOCAL_DIR
# HF folder in shared file system
HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
export HF_HOME="${HF_LOCAL_DIR}"

HF_MODEL_NAME="bert-base-uncased"  #
GLUE_TASK="sst2"  # default "sst2"
BATCH_SIZE=8
LEARNING_RATE="2e-5"
WEIGHT_DECAY="0.01"
NUM_EPOCHS=5

python load_datasets.py
#python load_datasets.py --task ${GLUE_TASK}
python load_model.py --model ${HF_MODEL_NAME}
rsync -a --ignore-existing ${HF_LOCAL_DIR}/ $HF_USER_DIR