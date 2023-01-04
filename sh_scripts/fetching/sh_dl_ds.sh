# Make Hugging Face cache folder on drive
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
mkdir -p $HF_LOCAL_DIR
# HF folder in shared file system
HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
export HF_HOME="${HF_LOCAL_DIR}"

python sh_scripts/fetching/sh_dl_ds.py
rsync -a --ignore-existing ${HF_LOCAL_DIR}/ $HF_USER_DIR