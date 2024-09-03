import evaluate

# from bert_score import BERTScorer


# def bert_score(preds, refs):
#     scorer = BERTScorer(
#         lang="en",
#         rescale_with_baseline=True,
#         use_fast_tokenizer=False,
#         model_type="microsoft/deberta-large-mnli",
#     )
#     P, R, F1 = scorer.score(refs, preds, batch_size=8, verbose=True)
#     return F1.mean().item()


def bleu_score(preds, refs):
    metric = evaluate.load_metric("blue")
    result = metric.compute(predictions=preds, references=refs)
    return result["blue"]


def rouge_score(preds, refs):
    metric = evaluate.load_metric("rouge")
    result = metric.compute(predictions=preds, references=refs, use_agregator=True)
    return result["rouge1"], result["rougeL"]


def exact_match_score(preds, refs):
    metric = evaluate.load_metric("exact_match")
    result = metric.compute(predictions=preds, references=refs)
    return result["exact_match"]


def evaluate_metrics(preds, refs):
    # create dictionary to store all the metrics
    metrics = {}
    print("Calculating rouge score ...")
    metrics["rouge1"], metrics["rougeL"] = rouge_score(preds, refs)
    print("Calculating bleu score ...")
    metrics["bleu"] = bleu_score(preds, refs)
    print("Calculating exact match score ...")
    metrics["exact_match"] = exact_match_score(preds, refs)
    # print("Calculating bert score ...")
    # metrics["bert_score"] = bert_score(preds, refs)
    return metrics
