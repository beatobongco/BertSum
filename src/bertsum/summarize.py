from argparse import Namespace

import nltk
import numpy as np
import torch
from pytorch_pretrained_bert import BertConfig

from .models import data_loader, model_builder
from .models.data_loader import load_dataset
from .models.model_builder import Summarizer
from .prepro import data_builder

nltk.download("punkt")

# TODO: make this configurable
MODELS_DIR = "/home/beato/bert-server/models/"
CONFIG_DIR = "/home/beato/bert-server/config/"
MODEL = "model_step_43000.pt"
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

def example_api(
    self,
    example,
    step,
    top_n_sentences=3,
    device="cpu",
    cal_lead=False,
    cal_oracle=False,
):
    """
    Runs inference on a single test example. Designed for API deployemnt.
    """
    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i : i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    if not cal_lead and not cal_oracle:
        # Evaluate without performing backpropagation and dropout
        self.model.eval()
    # Set model device (cuda or cpu)
    self.model.to(device=device)
    source_article = []
    pred = []
    src = example.src
    labels = example.labels
    segs = example.segs
    clss = example.clss
    mask = example.mask
    mask_cls = example.mask_cls
    src_str = example.src_str

    source_article += [" ".join(article) for article in src_str]

    if cal_lead:
        selected_ids = [list(range(example.clss.size(1)))] * example.batch_size
    elif cal_oracle:
        selected_ids = [
            [j for j in range(example.clss.size(1)) if labels[i][j] == 1]
            for i in range(example.batch_size)
        ]
    else:
        sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        # Sort sentence ids in descending order based on sentence scores (representing summary importance)
        selected_ids = np.argsort(-sent_scores, 1)
    # selected_ids = np.sort(selected_ids,1)
    for i, idx in enumerate(selected_ids):
        _pred = []
        if len(example.src_str[i]) == 0:
            continue
        # Loop through each sentence
        # len(example.src_str[i]) refers to the number of sentences in the jth test example
        for j in selected_ids[i][: len(example.src_str[i])]:
            if j >= len(example.src_str[i]):
                continue
            candidate = example.src_str[i][j].strip()
            if self.args.block_trigram:
                if not _block_tri(candidate, _pred):
                    _pred.append(candidate)
            else:
                _pred.append(candidate)

            # len(_pred) == 3 means that we limit sentences to top top_n_sentences
            if (
                (not cal_oracle)
                and (not self.args.recall_eval)
                and len(_pred) == top_n_sentences
            ):
                break

        _pred = "<q>".join(_pred)

        pred.append(_pred)

    results = {"source_article": source_article, "predicted_summary": pred}
    return results


def summarize(src_str, args=args, pp_args=pp_args, top_n_sentences=3, tgt_str=""):
    """
    Summarizes input text by returning the most important sentences based on the BERT model fine-tuned on CNN and Daily Mail articles
    """
    cp = args.test_from
    step = int(cp.split(".")[-2].split("_")[-1])
    # Separate documents into list of sentences
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = [sent.split() for sent in nltk.tokenize.sent_tokenize(src_str)]
    tgt = [sent.split() for sent in nltk.tokenize.sent_tokenize(tgt_str)]
    bert = data_builder.BertData(pp_args)
    oracle_ids = data_builder.greedy_selection(src, tgt, 3)
    b_data = bert.preprocess(src, tgt, oracle_ids)
    indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
    b_dict = {
        "src": indexed_tokens,
        "labels": labels,
        "segs": segments_ids,
        "clss": cls_ids,
        "src_str": src_txt,
        "tgt_str": tgt_txt,
    }
    data = [
        [
            b_dict["src"],
            b_dict["labels"],
            b_dict["segs"],
            b_dict["clss"],
            b_dict["src_str"],
            b_dict["tgt_str"],
        ]
    ]
    batch = data_loader.Batch(data, is_test=True, device=device)
    trained_model = torch.load(args.model_fp, map_location=lambda storage, loc: storage)
    # Read BERT config file
    # This contains information about the BERT model (e.g. hidden size for the transformer layers, number of transformer layers, number of self attention heads)
    config = BertConfig.from_json_file(args.bert_config_path)
    # Instantiate Summarizer and load pretrained model
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    model.load_cp(trained_model)
    device_id = 0 if device == "cuda" else -1
    results = example_api(
        batch, step, top_n_sentences=top_n_sentences, device=device
    )
    return results
