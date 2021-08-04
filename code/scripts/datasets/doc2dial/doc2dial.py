# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""Doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset v1.0.1"""

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import datasets

MAX_Q_LEN = 100  # Max length of question
YOUR_LOCAL_DOWNLOAD = "YOUR_LOCAL_DIR/sharedtask-dialdoc2021/data"  # For subtask1, Doc2Dial v1.0.1 is already included in the folder "data".

_CITATION = """\
@inproceedings{feng-etal-2020-doc2dial,
    title = "doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset",
    author = "Feng, Song  and Wan, Hui  and Gunasekara, Chulaka  and Patel, Siva  and Joshi, Sachindra  and Lastras, Luis",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.652",
}
"""

_DESCRIPTION = """\
Doc2dial is dataset of goal-oriented dialogues that are grounded in the associated documents. \
It includes over 4500 annotated conversations with an average of 14 turns that are grounded \
in over 450 documents from four domains. Compared to the prior document-grounded dialogue datasets \
this dataset covers a variety of dialogue scenes in information-seeking conversations.
"""

_HOMEPAGE = "https://doc2dial.github.io"


_URLs = "https://doc2dial.github.io/file/doc2dial_v1.0.1.zip"

def TODO():
    raise Exception("needs to be implemented")


class Doc2dial(datasets.GeneratorBasedBuilder):
    "Doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset v1.0.1"

    VERSION = datasets.Version("1.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="dialogue_domain",
            version=VERSION,
            description="This part of the dataset covers the dialgoue domain that has questions, answers and the associated doc ids",
        ),
        datasets.BuilderConfig(
            name="document_domain",
            version=VERSION,
            description="This part of the dataset covers the document domain which details all the documents in the various domains",
        ),
        datasets.BuilderConfig(
            name="doc2dial_rc",
            version=VERSION,
            description="Load Doc2Dial dataset for machine reading comprehension tasks",
        ),
        datasets.BuilderConfig(
            name="doc2dial_mc",
            version=VERSION,
            description="Load Doc2Dial dataset for multiple choice tasks",
        ),
    ]

    DEFAULT_CONFIG_NAME = "dialogue_domain"

    def _info(self):

        if self.config.name == "dialogue_domain":
            features = datasets.Features(
                {
                    "dial_id": datasets.Value("string"),
                    "doc_id": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "turns": [
                        {
                            "turn_id": datasets.Value("int32"),
                            "role": datasets.Value("string"),
                            "da": datasets.Value("string"),
                            "references": [
                                {
                                    "sp_id": datasets.Value("string"),
                                    "label": datasets.Value("string"),
                                }
                            ],
                            "utterance": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.name == "document_domain":
            features = datasets.Features(
                {
                    "domain": datasets.Value("string"),
                    "doc_id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "doc_text": datasets.Value("string"),
                    "spans": [
                        {
                            "id_sp": datasets.Value("string"),
                            "tag": datasets.Value("string"),
                            "start_sp": datasets.Value("int32"),
                            "end_sp": datasets.Value("int32"),
                            "text_sp": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "parent_titles": datasets.Value("string"),
                            "id_sec": datasets.Value("string"),
                            "start_sec": datasets.Value("int32"),
                            "text_sec": datasets.Value("string"),
                            "end_sec": datasets.Value("int32"),
                        }
                    ],
                    "doc_html_ts": datasets.Value("string"),
                    "doc_html_raw": datasets.Value("string"),
                }
            )
        elif self.config.name == "doc2dial_rc":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                            # "answer_end": datasets.Value("int32"),
                            # "spans": datasets.features.Sequence(datasets.Value("string"))
                        }
                    ),
                    "domain": datasets.Value("string"),
                    "spans": datasets.features.Sequence({
                        'start_idx': datasets.Value('int32'),
                        'end_idx': datasets.Value('int32'),
                    }),
                }
            )
        elif self.config.name == "doc2dial_mc":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "labels": datasets.Value("string"),
                    "sections": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "sec_id": datasets.Value("string"),
                            # "answer_end": datasets.Value("int32"),
                            # "spans": datasets.features.Sequence(datasets.Value("string"))
                        }
                    ),
                    "domain": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        my_urls = _URLs

        data_dir = dl_manager.download_and_extract(my_urls) 
        #data_dir = TODO() # point to local dir to avoid downloading the dataset again

        if self.config.name == "dialogue_domain":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_train.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_validation.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/test/doc2dial_dial_finaltest_2.json"
                        ),
                    },
                ),
            ]
        elif self.config.name == "document_domain":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/test/doc2dial_doc_with_unseen.json"
                        ),
                    },
                )
            ]
        elif self.config.name == "doc2dial_rc":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_validation.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/test/doc2dial_dial_finaltest_2.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_train.json"
                        ),
                    },
                ),
            ]
        elif self.config.name == "doc2dial_mc":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_validation.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/test/doc2dial_dial_finaltest_2.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_train.json"
                        ),
                    },
                ),
            ]
        elif self.config.name == "doc2dial_mc":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_validation.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_testdev_2.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "doc2dial/v1.0.1/doc2dial_dial_train.json"
                        ),
                    },
                ),
            ]

    def _load_doc_data_rc(self, filepath):
        if "test" in filepath:
            doc_filepath = os.path.join(os.path.dirname(filepath), "doc2dial_doc_with_unseen.json")
        else:
            doc_filepath = os.path.join(os.path.dirname(filepath), "doc2dial_doc.json")
        with open(doc_filepath, encoding="utf-8") as f:
            data = json.load(f)["doc_data"]
        return data

    def _get_answers_rc(self, references, spans, doc_text):
        """Obtain the grounding annotation for evaluation of subtask1."""
        if not references:
            return []
        start, end = -1, -1
        ls_sp = []
        for ele in references:
            sp_id = ele["sp_id"]
            start_sp, end_sp = spans[sp_id]["start_sp"], spans[sp_id]["end_sp"]
            if start == -1 or start > start_sp:
                start = start_sp
            if end < end_sp:
                end = end_sp
            ls_sp.append(doc_text[start_sp:end_sp])
        answer = {
            "text": doc_text[start:end],
            "answer_start": start,
            # "answer_end": end
            # "spans": ls_sp
        }
        return [answer]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        if self.config.name == "dialogue_domain":
            logging.info("generating examples from = %s", filepath)
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for domain in data["dial_data"]:
                    for doc_id in data["dial_data"][domain]:
                        for dialogue in data["dial_data"][domain][doc_id]:

                            x = {
                                "dial_id": dialogue["dial_id"],
                                "domain": domain,
                                "doc_id": doc_id,
                                "turns": dialogue["turns"],
                            }

                            yield dialogue["dial_id"], x

        elif self.config.name == "document_domain":

            logging.info("generating examples from = %s", filepath)
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for domain in data["doc_data"]:
                    for doc_id in data["doc_data"][domain]:
                        for dialogue in data["doc_data"][domain][doc_id]:

                            yield doc_id, {
                                "domain": domain,
                                "doc_id": doc_id,
                                "title": data["doc_data"][domain][doc_id]["title"],
                                "doc_text": data["doc_data"][domain][doc_id][
                                    "doc_text"
                                ],
                                "spans": [
                                    {
                                        "id_sp": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["id_sp"],
                                        "tag": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["tag"],
                                        "start_sp": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["start_sp"],
                                        "end_sp": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["end_sp"],
                                        "text_sp": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["text_sp"],
                                        "title": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["title"],
                                        "parent_titles": str(
                                            data["doc_data"][domain][doc_id]["spans"][
                                                i
                                            ]["parent_titles"]
                                        ),
                                        "id_sec": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["id_sec"],
                                        "start_sec": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["start_sec"],
                                        "text_sec": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["text_sec"],
                                        "end_sec": data["doc_data"][domain][doc_id][
                                            "spans"
                                        ][i]["end_sec"],
                                    }
                                    for i in data["doc_data"][domain][doc_id]["spans"]
                                ],
                                "doc_html_ts": data["doc_data"][domain][doc_id][
                                    "doc_html_ts"
                                ],
                                "doc_html_raw": data["doc_data"][domain][doc_id][
                                    "doc_html_raw"
                                ],
                            }

        elif self.config.name == "doc2dial_rc":
            """Load dialog data in the reading comprehension task setup, where context is the grounding document,
            input query is dialog history in reversed order, and output to predict is the next agent turn."""

            logging.error("generating examples from = %s", filepath)
            #split = filepath.split(".")[-2].split("_")[-1]
            split = next((split for split in ["train", "validation", "test"] if split in filepath))
            logging.error(f"Split: {split}")
            doc_data = self._load_doc_data_rc(filepath)
            with open(filepath, encoding="utf-8") as f:
                dial_data = json.load(f)["dial_data"]
                for domain, d_doc_dials in dial_data.items():
                    for doc_id, dials in d_doc_dials.items():
                        doc = doc_data[domain][doc_id]
                        for dial in dials:
                            all_prev_utterances = []
                            for idx, turn in enumerate(dial["turns"]):
                                all_prev_utterances.append(
                                    "\t{}: {}".format(turn["role"], turn["utterance"])
                                )
                                if split != "test": # testdev does not have references
                                    if "answers" not in turn:
                                        turn["answers"] = self._get_answers_rc(
                                            turn["references"],
                                            doc["spans"],
                                            doc["doc_text"],
                                        )
                                if turn["role"] == "agent":
                                    continue

                                if not split == "test":
                                    if idx + 1 < len(dial["turns"]):
                                        if dial["turns"][idx + 1]["role"] == "agent":
                                            turn_to_predict = dial["turns"][idx + 1]
                                        else:
                                            continue
                                    else:
                                        continue
                                question_str = " ".join(
                                    list(reversed(all_prev_utterances))
                                ).strip()
                                question = " ".join(question_str.split()[:MAX_Q_LEN])
                                id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"]) # For subtask1, the id should be this format.
                                qa = {
                                    "id": id_, # For subtask1, the id should be this format.
                                    "title": doc_id,
                                    "context": doc["doc_text"],
                                    "question": question,
                                    "answers": [],  # For subtask1, "answers" contains the grounding annotations for evaluation.
                                    "domain": domain,
                                    "spans": [],
                                }

                                if split != "test": # testdev does not have references
                                    if "answers" not in turn_to_predict:
                                        turn_to_predict["answers"] = self._get_answers_rc(
                                            turn_to_predict["references"],
                                            doc["spans"],
                                            doc["doc_text"],
                                        )
                                    if turn_to_predict["answers"]:
                                        qa["answers"] = turn_to_predict["answers"]

                                for span in doc["spans"].values():
                                    text = span["text_sec"]
                                    qa['spans'].append({
                                        'start_idx': span['start_sp'] + len(text) - len(text.lstrip()),
                                        'end_idx': span['end_sp'] - len(text) + len(text.rstrip()),
                                    })

                                yield id_, qa

        elif self.config.name == "doc2dial_mc":
            """Load dialog data in the reading comprehension task setup, where context is the grounding document,
            input query is dialog history in reversed order, and output to predict is the next agent turn."""

            logging.error("generating examples from = %s", filepath)
            split = next((split for split in ["train", "validation", "test"] if split in filepath))
            logging.error(f"Split: {split}")
            doc_data = self._load_doc_data_rc(filepath)
            with open(filepath, encoding="utf-8") as f:
                dial_data = json.load(f)["dial_data"]
                for domain, d_doc_dials in dial_data.items():
                    for doc_id, dials in d_doc_dials.items():
                        doc = doc_data[domain][doc_id]

                        doc_sections = []
                        for id_, info in doc["spans"].items():
                            if info["id_sec"] not in [item["sec_id"] for item in doc_sections]:
                                doc_sections.append({
                                    "sec_id": info["id_sec"],
                                    "text": info["text_sec"]
                                })
                        
                        for dial in dials:
                            all_prev_utterances = []
                            for idx, turn in enumerate(dial["turns"]):
                                all_prev_utterances.append(
                                    "\t{}: {}".format(turn["role"], turn["utterance"])
                                )

                                if not split == "test":
                                    if idx + 1 < len(dial["turns"]):
                                        if dial["turns"][idx + 1]["role"] == "agent":
                                            turn_to_predict = dial["turns"][idx + 1]
                                        else:
                                            continue
                                    else:
                                        continue
                                question_str = " ".join(
                                    list(reversed(all_prev_utterances))
                                ).strip()
                                question = " ".join(question_str.split()[:MAX_Q_LEN])
                                id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"]) # For subtask1, the id should be this format.
                                qa = {
                                    "id": id_, # For subtask1, the id should be this format.
                                    "title": doc_id,
                                    "question": question,
                                    "labels": doc["spans"][turn_to_predict["references"][0]["sp_id"]]["id_sec"] if not split == "test" else "1",
                                    "sections": doc_sections,
                                    "domain": domain,
                                }
                                yield id_, qa
