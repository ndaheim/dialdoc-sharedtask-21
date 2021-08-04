# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def postprocess_qa_predictions(
    examples,
    features,
    predictions: np.ndarray,
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
    cheating: bool = False,
    add_title_to_pred: bool = False,
    extend_preds_by: int = 0,
    docs = None
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert len(predictions) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    count = 0

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            is_span_model = predictions.shape[-1] != 2
            offset_mapping = features[feature_index]["offset_mapping"]

            seq_length = predictions[feature_index].shape[0]
            if is_span_model:
                span_predictions = predictions[feature_index].reshape(seq_length, seq_length)
            else:
                span_predictions = predictions[feature_index].reshape(seq_length, 2)

            def get_pred_value(start, end):
                if is_span_model:
                    return span_predictions[start, end]
                else:
                    return span_predictions[start, 0] + span_predictions[end, 1]


            feature_null_score = get_pred_value(0, 0)
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                }

            if is_span_model:
                span_predictions[len(offset_mapping):, :] = -float('inf')
                span_predictions[:, len(offset_mapping):] = -float('inf')
                for i, om in enumerate(offset_mapping):
                    if om is None or om[0] is None:
                        span_predictions[i, :] = -float('inf')
                    if om is None or om[1] is None:
                        span_predictions[:, i] = -float('inf')
            else:
                span_predictions[len(offset_mapping):, :] = -float('inf')
                for i, om in enumerate(offset_mapping):
                    if om is None or om[0] is None:
                        span_predictions[i, 0] = -float('inf')
                    if om is None or om[1] is None:
                        span_predictions[i, 1] = -float('inf')




            if is_span_model:
                best_indices = np.argpartition(span_predictions.reshape(-1), -n_best_size)[-n_best_size:]
                for flat_index in best_indices:
                    start_index = flat_index // seq_length
                    end_index = flat_index % seq_length
                    if (start_index >= len(offset_mapping)
                         or end_index >= len(offset_mapping)
                         or offset_mapping[start_index] is None or offset_mapping[start_index][0] is None
                         or offset_mapping[end_index] is None or offset_mapping[end_index][1] is None):
                        if span_predictions[start_index, end_index] == -float('inf'):
                            continue
                        assert False
                    if end_index < start_index:# or end_index - start_index + 1 > max_answer_length:
                        if span_predictions[start_index, end_index] == -float('inf'):
                            continue
                        assert False
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": span_predictions[start_index, end_index],
                        }
                    )
            else:
                start_logits, end_logits = np.split(span_predictions, 2, axis=-1)
                start_logits = start_logits.squeeze()
                end_logits = end_logits.squeeze()
                start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index:
                            continue
                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )


        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        out_predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in out_predictions):
            out_predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in out_predictions:
            offsets = pred["offsets"]
            start, end = offsets[0], offsets[1]
            pred["text"] = context[start : end]
            if extend_preds_by > 0:
                pred["text"] = " ".join([" ".join(context[:start].split(" ")[-extend_preds_by:]), pred["text"], " ".join(context[end:].split(" ")[:extend_preds_by])])
            
            possible_spans = docs["doc_data"][example["domain"]][example["title"]]["spans"]

            if add_title_to_pred and docs is not None:
                span_idx = "1"
                while possible_spans[span_idx]["start_sp"] <= start and int(span_idx) < len(possible_spans):
                    span_idx = str(int(span_idx) + 1)
                pred["text"] = " ".join(["<sec_title>", possible_spans[str(int(span_idx) - 1)]["title"], "<doc_context>", pred["text"]])

        if cheating:

            if example is not None and "answers" in example:
                if not example["answers"]["text"][0].rstrip(" ") in [pred["text"] for pred in predictions]:
                    count = count + 1
                    
                    if len(predictions) != 0:
                        position = len(predictions) - 1
                        predictions[position] = {
                            "text": example["answers"]["text"][0],
                            "score": predictions[position]["score"],
                            "start_logit": predictions[position]["start_logit"],
                            "end_logit": predictions[position]["end_logit"]
                        }
                    else:
                        predictions.append({
                            "text": example["answers"]["text"][0],
                            "score": 2.0,
                            "start_logit": 1.0,
                            "end_logit": 1.0
                        })

        for pred in predictions:
            if "offsets" in pred:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(out_predictions) == 0 or (len(out_predictions) == 1 and out_predictions[0]["text"] == ""):
            out_predictions.insert(0, {"text": "empty", "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in out_predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, out_predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = out_predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while out_predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = out_predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["score"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in out_predictions
        ]

    logging.error(f"Out of {example_index} examples {count} times not in nbest list.")

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"predictions_{prefix}".json
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"nbest_predictions_{prefix}".json
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"null_odds_{prefix}".json
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions
