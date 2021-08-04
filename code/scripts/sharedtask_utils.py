import json
import argparse

from datasets import load_dataset
from datasets import load_metric


def sharedtask1_metrics(prediction_json, split, cache_dir=None, report_exact_math_at_5=False):
    metric = load_metric("squad_v2")

    orig_predictions = json.load(open(prediction_json, "r"))
    predictions = orig_predictions

    if split == "validation":
        predictions = [
            {
                "id": sample_id,
                "prediction_text": text if type(text) == str else text[0]['text'],
                "no_answer_probability": 0
            }
            for sample_id, text in predictions.items()
        ]

    d_id_prediction = {}
    for ele in predictions:
        d_id_prediction[ele["id"]] = 0

    references = []
    d_id_reference = {}
    dataset = load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split=split,
        ignore_verifications=True,
        cache_dir=cache_dir,
    )
    for ex in dataset:
        if ex["id"] not in d_id_prediction:
            continue
        d_id_reference[ex["id"]] = 0
        references.append(
            {
                "id": ex["id"],
                "answers": ex["answers"],
            }
        )

    new_predictions = []
    for pred in predictions:
        if pred["id"] not in d_id_reference:
            continue
        new_predictions.append(pred)
    predictions = new_predictions

    new_d_id_prediction = {}
    for pred_id, value in d_id_prediction.items():
        if pred_id not in d_id_reference:
            continue
        new_d_id_prediction[pred_id] = value
    d_id_prediction = new_d_id_prediction

    assert (
        len(predictions)
        == len(references)
        == len(d_id_prediction)
        == len(d_id_reference)
    ), f"Ensure the matching count of instances of references and predictioins {len(predictions)} {len(references)} {len(d_id_prediction)} {len(d_id_reference)}"

    metric.add_batch(predictions=predictions, references=references)
    final_score = metric.compute()



    if report_exact_math_at_5:
        from transformers.data.metrics.squad_metrics import compute_exact

        for nbest in [5, 10]:
            matches = 0
            for ref in references:
                hyps = orig_predictions[ref['id']]
                matches += max(compute_exact(ref['answers']['text'][0], hyp['text']) for hyp in hyps[:nbest])
            final_score[f'exact@{nbest}'] = matches / len(references)

    """
    print(final_score)
    OrderedDict([('exact', 33.333333333333336), ('f1', 38.095238095238095), ('span', 33.333333333333336), ('total', 3), ('HasAns_exact', 33.333333333333336), ('HasAns_f1', 38.095238095238095), ('HasAns_total', 3)])
    """
    return final_score


def sharedtask2_metrics(prediction_json, split, cache_dir):
    metric_sacrebleu = load_metric("sacrebleu")

    predictions = json.load(open(prediction_json, "r"))
    d_id_prediction = {}
    model_predictions = []
    for ex in predictions:
        model_predictions.append(ex["utterance"])
        d_id_prediction[ex["id"]] = 0

    references_lst = []
    dataset = load_dataset(
        "doc2dial",
        name="dialogue_domain",
        split=split,
        ignore_verifications=True,
        cache_dir=cache_dir,
    )
    for ex in dataset:
        for turn in ex["turns"]:
            if turn["role"] == "agent":
                id_ = "{}_{}".format(ex["dial_id"], turn["turn_id"] - 1)
                if id_ not in d_id_prediction:
                    continue
                references_lst.append([turn["utterance"]])
    assert (
        len(model_predictions) == len(references_lst) == len(d_id_prediction)
    ), "Ensure the matching count of instances of references and predictioins"
    metric_sacrebleu.add_batch(predictions=model_predictions, references=references_lst)
    final_score = metric_sacrebleu.compute()["score"]
    """
    print(final_score)
    sacrebleu 8.234381476893315
    """
    return final_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Select metrics for task that is either 'subtask1' or 'subtask2'",
    )
    parser.add_argument(
        "--prediction_json",
        type=str,
        required=True,
        help="Path to predictions",
    )
    parser.add_argument(
        "--split",
        default="validation",
        type=str,
        help='Data split for validation that is either "validation" or "test"',
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path for caching the downloaded data by HuggingFace Datasets",
    )
    parser.add_argument(
        "--report_em_at_5",
        type=bool,
        default=False,
        help="Report EM@5",
    )
    args = parser.parse_args()
    if args.task == "subtask1":
        score = sharedtask1_metrics(args.prediction_json, args.split, args.cache_dir, args.report_em_at_5)
    else:
        score = sharedtask2_metrics(args.prediction_json, args.split, args.cache_dir)

    print(score)

if __name__ == "__main__":
    main()
