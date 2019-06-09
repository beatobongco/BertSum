from argparse import Namespace

import nltk
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
            test_from=MODEL,
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
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    def summarize(self, src_str, top_n_sentences=3, tgt_str="")
        """
        Summarizes input text by returning the most important sentences based on the BERT model fine-tuned on CNN and Daily Mail articles
        """
        # Separate documents into list of sentences

        src = [sent.split() for sent in nltk.tokenize.sent_tokenize(src_str)]
        tgt = [sent.split() for sent in nltk.tokenize.sent_tokenize(tgt_str)]
        bert = data_builder.BertData(self.pp_args)
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
        batch = data_loader.Batch(data, is_test=True, device=self.device)

        source_article = [" ".join(article) for article in batch.src_str]

        sent_scores, mask = self.model(batch.src, batch.segs, batch.clss, batch.mask, batch.mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        # Sort sentence ids in descending order based on sentence scores (representing summary importance)
        selected_ids = np.argsort(-sent_scores, 1)
        for i, idx in enumerate(selected_ids):
            _pred = []
            if len(batch.src_str[i]) == 0:
                continue
            # Loop through each sentence
            # len(batch.src_str[i]) refers to the number of sentences in the jth test batch
            for j in selected_ids[i][: len(batch.src_str[i])]:
                if j >= len(batch.src_str[i]):
                    continue
                candidate = batch.src_str[i][j].strip()
                if args.block_trigram:
                    if not _block_tri(candidate, _pred):
                        _pred.append(candidate)
                else:
                    _pred.append(candidate)

                # len(_pred) == 3 means that we limit sentences to top top_n_sentences
                if len(_pred) == top_n_sentences:
                    break

            _pred = " ".join(_pred)

        return _pred
