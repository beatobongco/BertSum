from argparse import Namespace

import nltk
import numpy as np
import torch
from pytorch_pretrained_bert import BertConfig

from .models import data_loader
from .models.model_builder import Summarizer
from .prepro import data_builder


class BSummarizer(object):
    """Summarizer class that only requires a pretrained model"""
    def __init__(self, model_path):
        # BERT model args
        self.args = Namespace(
            encoder="transformer",
            mode="test",
            bert_data_path="/home/sample_data/cnndm",
            result_path="/home/bert_results/cnndm",
            temp_dir="/home/temp/",
            batch_size=1000,
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
            train_from="",
            report_rouge=True,
            block_trigram=True,
        )

        # Preprocessing arguments
        self.pp_args = Namespace(
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

        self.bert_config = BertConfig.from_dict({
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30522
        })

        self.BertData = data_builder.BertData(self.pp_args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Summarizer(self.args, self.device, load_pretrained_bert=False, bert_config=self.bert_config)
        model.load_cp(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model = model

    def _get_ngrams(self, n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i : i + n]))
        return ngram_set

    def _block_tri(self, c, p):
        tri_c = self._get_ngrams(3, c.split())
        for s in p:
            tri_s = self._get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    def summarize(self, src_str, top_n_sentences=3, tgt_str=""):
        """
        Summarizes input text by returning the most important sentences based on the BERT model fine-tuned on CNN and Daily Mail articles
        """
        # Separate documents into list of sentences
        src = [sent.split() for sent in nltk.tokenize.sent_tokenize(src_str)]
        b_data = self.BertData.preprocess(src, [], [])

        if not b_data:
            return {"error": "Not enough text to create a summary."}

        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        data = [
            [
                indexed_tokens,
                labels,
                segments_ids,
                cls_ids,
                src_txt,
                tgt_txt,
            ]
        ]
        batch = data_loader.Batch(data, is_test=True, device=self.device)
        sent_scores, mask = self.model(batch.src, batch.segs, batch.clss, batch.mask, batch.mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        # Sort sentence ids in descending order based on sentence scores (representing summary importance)
        # BRB: Since batch size is just 1 we can take the first element of the scored sentences and source string
        scored_sentences = np.argsort(-sent_scores, 1)[0]
        article = batch.src_str[0]
        _pred = []

        # Loop through each sentence
        for i in scored_sentences:
            candidate = article[i].strip()
            if self.args.block_trigram:
                if not self._block_tri(candidate, _pred):
                    _pred.append(candidate)
            else:
                _pred.append(candidate)

            # len(_pred) == 3 means that we limit sentences to top top_n_sentences
            if len(_pred) == top_n_sentences:
                break

        return {"result": " ".join(_pred)}
