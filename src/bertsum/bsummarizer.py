from argparse import Namespace

import nltk
import numpy as np
import torch
from pytorch_pretrained_bert import BertConfig

from .default_args import bert_config, preprocessing_args, summarizer_args
from .models import data_loader
from .models.model_builder import Summarizer
from .prepro import data_builder


class BSummarizer(object):
    """Ready-to-import summarizer class.

        Args:
            model_path: str: full path of the Summarizer's PyTorch state_dict
            summarizer_args: dict: configuration variables for the Summarizer model
            preprocessing_args: dict: configuration variables for data_builder.BertData
            bert_config: dict: configuration variables for the pretrained BERT model
    """
    def __init__(self, model_path, summarizer_args=summarizer_args, preprocessing_args=preprocessing_args, bert_config=bert_config):
        # BERT model args
        self.args = Namespace()
        self.args.__dict__ = summarizer_args

        # Preprocessing arguments
        self.pp_args = Namespace()
        self.pp_args.__dict__ = preprocessing_args

        self.bert_config = BertConfig.from_dict(bert_config)

        self.BertData = data_builder.BertData(self.pp_args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Summarizer(self.args, self.device, load_pretrained_bert=False, bert_config=self.bert_config)
        model.load_cp(torch.load(model_path, map_location=lambda storage, loc: storage))
        # Evaluate without performing backpropagation and dropout
        model.eval()
        model.to(device=self.device)
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
