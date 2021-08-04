import argparse

parser = argparse.ArgumentParser()
parser.add_argument("generator_model_name_or_path", type=str)
parser.add_argument("encoder_model_name_or_path", type=str)
parser.add_argument("checkpoint_path", type=str)
args, additional_args = parser.parse_known_args()

from transformers import AutoConfig
from modeling_rag import PreTrainedRagModel
from configuration_rag import RagConfig

import logging

generator_config = AutoConfig.from_pretrained(args.generator_model_name_or_path)
config = RagConfig(
    pretrained_question_encoder_tokenizer_name_or_path=args.encoder_model_name_or_path if len(args.encoder_model_name_or_path) > 0 else None,
    pretrained_question_encoder_name_or_path=args.encoder_model_name_or_path if len(args.encoder_model_name_or_path) > 0 else None,
    pretrained_generator_tokenizer_name_or_path=args.generator_model_name_or_path,
    pretrained_generator_name_or_path=args.generator_model_name_or_path,
    **generator_config.to_diff_dict()
)

model = PreTrainedRagModel(config=config)
model.save_pretrained(args.checkpoint_path)
