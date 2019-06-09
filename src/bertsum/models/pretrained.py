from argparse import Namespace

import torch
from pytorch_pretrained_bert import BertConfig

from .model_builder import Summarizer

# TODO: make this configurable
MODELS_DIR = "/home/beato/bert-server/models/"
CONFIG_DIR = "/home/beato/bert-server/config/"
MODEL = "bertsum_state_dict.pt"
MODEL_FP = f"{MODELS_DIR}{MODEL}"

# Preprocessing arguments
# Setting these as default arguments
# These attributes can be edited as needed
pp_args = Namespace(
    mode="",
    oracle_mode="greedy",
    shard_size=2000,
    min_nsents=3,
    max_nsents=100,
    min_src_ntokens=5,
    max_src_ntokens=200,
    lower=True,
    dataset="",
    n_cpus=2,
)

# Setting these as default arguments
# These attributes can be edited as needed
args = Namespace(
    encoder="transformer",
    mode="test",
    bert_data_path="/home/sample_data/cnndm",
    model_path=MODELS_DIR,
    result_path="/home/bert_results/cnndm",
    temp_dir="/home/temp/",
    bert_config_path=f"{CONFIG_DIR}bert_config_uncased_base.json",
    batch_size=1000,
    model_fp=MODEL_FP,
    use_interval=True,
    hidden_size=128,
    ff_size=2048,  # Size used during training
    heads=4,
    inter_layers=2,
    rnn_size=512,
    param_init=0.0,
    param_init_glorot=True,
    dropout=0.1,
    optim="adam",
    lr=1,
    beta1=0.9,
    beta2=0.999,
    decay_method="",
    warmup_steps=8000,
    max_grad_norm=0,
    save_checkpoint_steps=5,
    accum_count=1,
    world_size=1,
    report_every=1,
    train_steps=1000,
    recall_eval=False,
    visible_gpus="-1",
    gpu_ranks="0",
    log_file="/home/logs/cnndm.log",
    dataset="",
    seed=666,
    test_all=False,
    test_from=MODEL,
    train_from="",
    report_rouge=True,
    block_trigram=True,
)

# Read BERT config file
# This contains information about the BERT model (e.g. hidden size for the transformer layers, number of transformer layers, number of self attention heads)
config = BertConfig.from_json_file(args.bert_config_path)
# Instantiate Summarizer and load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
loaded = torch.load(args.model_fp, map_location=lambda storage, loc: storage)
model.load_cp(loaded)
