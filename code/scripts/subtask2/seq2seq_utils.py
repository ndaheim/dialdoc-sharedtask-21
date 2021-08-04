import json
import os
import argparse
from collections import defaultdict
import logging

from datasets import load_dataset

DOC_DOMAIN_SPLIT = "train"
YOUR_DATASETS_SOURCE_DIR = ""  # the root folder of your local `datasets` source code.


def text2line(text):
    return text.replace("\n", "\t").replace("\r", "\t").strip()


def btag(tag, text):  # tag the content
    return "<{}>\t{}".format(tag, text2line(text))


def load_doc2dial_seq2seq(args):
    doc_dataset = load_dataset(
        args.dataset_file,# "../datasets/doc2dial",
        name="document_domain",
        split=DOC_DOMAIN_SPLIT,
        cache_dir=args.cache_dir,
    )
    dial_dataset = load_dataset(
        args.dataset_file,# "../datasets/doc2dial",
        name="dialogue_domain",
        split=args.split,
        cache_dir=args.cache_dir,
        ignore_verifications=True,
    )

    if args.from_preds:
        with open(args.preds_file, "r") as f:
            preds = json.load(f)
    
    if args.id_file:
        with open(args.id_file, "r") as f:
            ids_to_use = json.load(f)
            ids_to_use = [dict_["id"] for dict_ in ids_to_use]
    
    d_doc = defaultdict(dict)
    for ex in doc_dataset:
        d_doc[ex["doc_id"]]["doc_text"] = ex["doc_text"]
        for d_span in ex["spans"]:
            d_doc[ex["doc_id"]][d_span["id_sp"]] = d_span
    source = []
    target = []
    ids = []
    sec_ids = []
    if args.split != "test":
        for ex in dial_dataset:
            doc_id = ex["doc_id"]
            d_doc_spans = d_doc[doc_id]
            dial_context = []
            contexts = None
            for i, turn in enumerate(ex["turns"]):
                if not turn[
                    "references"
                ]:  # this task only uses instances and evalutes on the grounded turns.
                    continue
                utterance = text2line(turn["utterance"])
                utterance_context = btag(turn["role"], utterance)
                if turn["role"] in args.role:  # if current turn is to predict
                    
                    contexts = [
                        btag("last_turn", dial_context[-1].split("\t", 1)[-1])
                    ]  # add previous utterance as tagged query context
                    contexts.extend(
                        dial_context[::-1]
                    )  # add dialog history in reverse order as tagged dialogue context
                    if not args.no_context:
                        if args.full_doc:
                            # add entire document as tagged document context
                            contexts += [
                                btag("title", ex["doc_id"]),
                                btag("doc_context", d_doc[doc_id]["doc_text"]),
                            ]
                        else:
                            reference_content = ""  # the grounding span content
                            d_sec = {}
                            ref_label = ""
                            for ref in turn["references"]:
                                sp_id = ref["sp_id"]
                                sp_label = ref["label"]
                                sec_id = d_doc_spans[sp_id]["id_sec"]
                                # rename sec_id for sorting the text sections in order.
                                if sec_id.startswith("t"):
                                    sec_id = sec_id.split("_", 1)[-1] + "_0"
                                else:
                                    sec_id = sec_id + "_1"
                                sec_content = d_doc_spans[sp_id]["text_sec"]
                                d_sec[sec_id] = sec_content
                                if "solution" in sp_label:
                                    ref_label = "solution"
                                elif "precondition" in sp_label:
                                    ref_label = "precondition"
                                if "reference" not in sp_label:
                                    reference_content += "\t" + d_doc_spans[sp_id]["text_sp"]

                            sec_contents = []

                            if not args.from_preds:
                                for k, v in sorted(d_sec.items()):
                                    sec_contents.append(v)
                                    if args.extend_context:
                                        predecessor_span_id = int(turn["references"][0]["sp_id"]) - 1
                                        while predecessor_span_id >= 1 and d_doc_spans[str(predecessor_span_id)]["text_sec"] == d_doc_spans[sp_id]["text_sec"]:
                                            predecessor_span_id = predecessor_span_id - 1
                                        if predecessor_span_id >= 1:
                                            sec_content = d_doc_spans[str(predecessor_span_id)]["text_sec"]
                                            sec_contents.insert(0, " ".join(sec_content.split(" ")[-args.context_window_size:]))
                                        successor_span_id = int(turn["references"][-1]["sp_id"]) + 1
                                        while successor_span_id < len(d_doc_spans) and d_doc_spans[str(successor_span_id)]["text_sec"] == d_doc_spans[sp_id]["text_sec"]:
                                            successor_span_id = successor_span_id + 1
                                        if int(successor_span_id) < len(d_doc_spans):
                                            sec_content = d_doc_spans[str(successor_span_id)]["text_sec"]
                                            sec_contents.append(" ".join(sec_content.split(" ")[:args.context_window_size]))

                                    if args.add_sec_title:
                                        contexts += [
                                            btag("title", ex["doc_id"]),
                                            btag("sec_title", d_doc_spans[sp_id]["title"]),
                                        ]
                                        if not any(["<doc_context>" in content for content in sec_contents]):
                                            contexts.append(btag("doc_context", "\t".join(sec_contents)))
                                        else:
                                            contexts.append("\t".join(sec_contents))
                                    else:
                                        contexts += [
                                            btag("title", ex["doc_id"]),
                                            btag(
                                                "doc_context", "\t".join(sec_contents)
                                            ),  # use a combine of related sections as document context.
                                        ]
                            else:
                                pred = preds["{}_{}".format(ex["dial_id"], turn["turn_id"] - 1)]

                                if "doc_context" in pred:
                                    contexts += [
                                        btag("title", ex["doc_id"]),
                                        "\t".join([pred.replace("\n", " ")]),
                                    ]
                                else:
                                    contexts += [
                                        btag("title", ex["doc_id"]),
                                        btag(
                                            "doc_context", "\t".join([pred])
                                        ),  # use a combine of related sections as document context.
                                    ]

                            if args.include_da:
                                da = get_da_name(
                                    turn["da"],
                                    turn["role"],
                                    turn["turn_id"],
                                    ref_label,
                                    args.simply_da,
                                )
                                da_context = btag("da", da)
                                contexts.extend(da_context)
                    source.append("\t".join(contexts))
                    target.append(utterance)
                    ids.append("{}_{}".format(ex["dial_id"], turn["turn_id"] - 1))
                    sp_id = turn["references"][0]["sp_id"]
                    sec_ids.append("{}\t{}".format(ex["doc_id"], d_doc[ex["doc_id"]][sp_id]["id_sec"]))
                dial_context.append(utterance_context)
    else:
        for ex in dial_dataset:
            id_ = "{}_{}".format(ex["dial_id"], ex["turns"][-1]["turn_id"])
            contexts = [
                btag("last_turn", ex["turns"][-1]["utterance"])
            ]
            for turn in ex["turns"][::-1]:
                contexts.append(btag(turn["role"], turn["utterance"]))
            contexts.append(btag("title", ex["doc_id"]))
            if args.from_preds:
                pred = preds[id_]
                contexts.append(btag("doc_context", pred))
            if args.full_doc:
                contexts.append(btag("doc_context", d_doc[ex["doc_id"]]["doc_text"]))
            source.append("\t".join(contexts))
            target.append("dummy target")
            ids.append(id_)

    assert len(source) == len(
        target
    ), "Need to ensure that source and target are same sized."
    if args.split == "validation":
        args.split = "val"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(
        os.path.join(args.output_dir, "{}.source".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(source))
        fp.close()
    with open(
        os.path.join(args.output_dir, "{}.target".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(target))
        fp.close()
    with open(
        os.path.join(args.output_dir, "{}.ids".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(ids))
        fp.close()
    with open(
        os.path.join(args.output_dir, "{}.sec_ids".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(sec_ids))
        fp.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Data split is 'train', 'validation' or 'test'",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path for caching the downloaded data by HuggingFace Datasets",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="agent",
        help="which role's utterance for generation",
    )
    parser.add_argument(
        "--full_doc",
        type=bool,
        default=False,
        help="whether use entire document",
    )
    parser.add_argument(
        "--include_da",
        type=bool,
        default=False,
        help="whether to include DA as input",
    )
    parser.add_argument(
        "--from_preds",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output the data files",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="path to the dataset file",
    )
    parser.add_argument(
        "--preds_file",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--id_file",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--extend_context",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--context_window_size",
        type=int,
        required=False
    )
    parser.add_argument(
        "--no_context",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--add_sec_title",
        action="store_true",
        default=False,
    )


    args = parser.parse_args()
    logging.error(args)
    load_doc2dial_seq2seq(args)


if __name__ == "__main__":
    main()
