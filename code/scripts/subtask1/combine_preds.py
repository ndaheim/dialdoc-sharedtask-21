import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.special import softmax

parser = argparse.ArgumentParser()

parser.add_argument("--preds", nargs="+")
parser.add_argument("--f1s", nargs="+")
parser.add_argument("--output_dir")

args = parser.parse_args()

preds = []
for path, f1 in zip(args.preds, args.f1s):
    with open(path, "r") as f:
        preds.append(json.load(f))

priors = softmax(np.log([float(f1) for f1 in args.f1s]))
new = {}

for i, key in enumerate(preds[0].keys()):
    local_preds = [defaultdict(float, {pred["text"]: pred["probability"] for pred in preds_[key]}) for preds_ in preds]
    new[key] = []

    for text, prob in local_preds[0].items():
        new[key].append({
        "text": text,
        "probability": sum([f1 * float(local_pred[text]) for f1, local_pred in zip(priors, local_preds)]),
        "start_logit": 0.0,
        "end_logit": 0.0
        })

    if len(new[key]) == 0:
        all_preds = {k: v for d in local_preds for k, v in d.items()}
        best_single_pred = sorted(all_preds.items(), key=lambda pred: pred[1], reverse=True)[0]
        pred = {
            "text": best_single_pred[0],
            "probability": best_single_pred[1],
            "start_logit": 0.0,
            "end_logit": 0.0
        }
        new[key].append(pred)

    new[key] = sorted(new[key], key=lambda pred: pred["probability"], reverse=True)

preds = {}

for key, pred in new.items():
    preds[key] = pred[0]["text"]

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

with open(os.path.join(args.output_dir, "nbest_ensemble.json"), "w") as f:
    json.dump(new, f, indent=2)

with open(os.path.join(args.output_dir, "preds_ensemble.json"), "w") as f:
    json.dump(preds, f, indent=4)