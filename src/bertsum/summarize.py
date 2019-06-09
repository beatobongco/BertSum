import nltk
import numpy as np

from .models import data_loader
from .models.pretrained import model
from .prepro import data_builder

nltk.download("punkt")

def example_api(
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
        model.eval()
    # Set model device (cuda or cpu)
    model.to(device=device)
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
        sent_scores, mask = model(src, segs, clss, mask, mask_cls)

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
            if args.block_trigram:
                if not _block_tri(candidate, _pred):
                    _pred.append(candidate)
            else:
                _pred.append(candidate)

            # len(_pred) == 3 means that we limit sentences to top top_n_sentences
            if (
                (not cal_oracle)
                and (not args.recall_eval)
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

    results = example_api(
        batch, step, top_n_sentences=top_n_sentences, device=device
    )
    return results
