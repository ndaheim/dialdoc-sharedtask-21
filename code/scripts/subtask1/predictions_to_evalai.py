import argparse
import json

parser = argparse.ArgumentParser(description='')
parser.add_argument('--id_file', type=str)
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--task', type=str)

args = parser.parse_args()

with open(args.id_file, "r") as f:
    ids = json.load(f)

with open(args.input_file, "r") as f:
    if args.task == "subtask1":
        preds = json.load(f)
    else:
        preds = f.readlines()
        preds = [pred.strip() for pred in preds]

if args.task == "subtask1":
    for dict_ in ids:
        prediction = preds[dict_["id"]]
        dict_["prediction_text"] = prediction
else:
    for dict_, pred in zip(ids, preds):
        dict_["utterance"] = pred

with open(args.output_file, "w") as f:
    json.dump(ids, f)
