import os
import shutil
import urllib.request
from typing import Union, List, Dict, Iterator

from sisyphus import *

def TODO():
    raise Exception("needs to be adapted")

class WriteSeq2SeqFilesJob(Job):

    def __init__(self,
        full_doc=False,
        predictions=None,
        id_file=None,
        context_window_size=0,
        add_sec_title=False,
        no_context=False
    ):
        self.full_doc = full_doc
        self.predictions = predictions
        self.id_file = id_file
        self.context_window_size = context_window_size
        self.add_sec_title = add_sec_title
        self.no_context = no_context
        self.out_path = self.output_path("data")

    def run(self):
        if self.predictions is None:
            for split in ["train", "validation"]:
                self.sh(
                    '''{python} {file} \
                            --split {split} \
                            {full_doc} \
                            {context_window} \
                            {no_context} \
                            {sec_title} \
                            --output_dir {out_path} \
                            --dataset_file {dataset_name}
                        ''',
                    python=gs.PYTHON_EXE,
                    file=f"{gs.SCRIPTS_ROOT}/subtask2/seq2seq_utils.py",
                    split=split,
                    full_doc="--full_doc" if self.full_doc else "",
                    context_window=f"--context_window_size {self.context_window_size}" if self.context_window_size > 0 else "",
                    sec_title="--add_sec_title" if self.add_sec_title else "",
                    no_context="--no_context" if self.no_context else "",
                    out_path=self.out_path,
                    dataset_name=f"'{gs.SCRIPTS_ROOT}/datasets/doc2dial/doc2dial.py'",
            )
        else:
            import os
            self.sh(
                '''{python} {file} \
                        --split {split} \
                        --from_preds \
                        {predictions} \
                        {full_doc} \
                        {context_window} \
                        {no_context} \
                        {sec_title} \
                        --output_dir {out_path} \
                        --dataset_file {dataset_name}
                    ''',
                python=gs.PYTHON_EXE,
                file=f"{gs.SCRIPTS_ROOT}/subtask2/seq2seq_utils.py",
                split="test",
                full_doc="--full_doc" if self.full_doc else "",
                context_window=f"--context_window_size {self.context_window_size}" if self.context_window_size > 0 else "",
                sec_title="--add_sec_title" if self.add_sec_title else "",
                no_context="--no_context" if self.no_context else "",
                predictions=f"--preds_file {os.path.join(self.predictions, 'predictions.json')}",
                out_path=self.out_path,
                dataset_name=f"'{gs.SCRIPTS_ROOT}/datasets/doc2dial/doc2dial.py'",
            )

    def tasks(self):
        yield Task('run', resume='run', rqmt={
            'cpu': 1,
            'gpu': 0,
            'mem': 4,
            'time': 4,
        }, mini_task=True)


class TrainSpanQAJob(Job):

    def __init__(
        self,
        gpus=1,
        max_seq_length=512,
        num_epochs=10,
        model_name_or_path="google/electra-base-discriminator",
        extra_time=0,
        batch_size=1,
        gradient_accum_steps=16,
        warmup_steps=1000,
        learning_rate=3e-5,
        **kwargs
    ):
        self.output_path = self.output_path("runs")
        self.gpus = gpus
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.model_name_or_path = model_name_or_path
        self.extra_time = extra_time
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.gradient_accum_steps = gradient_accum_steps
        self.learning_rate = learning_rate

    def run(self):
        self.sh(
            '''{python} {file} \
                --dataset_name {dataset_name} \
                --dataset_config_name doc2dial_rc \
                --model_name_or_path {model_name_or_path} \
                --do_train \
                --do_eval \
                --logging_steps 2000 \
                --save_steps 2000 \
                --learning_rate {learning_rate}  \
                --num_train_epochs {num_epochs} \
                --max_seq_length {max_seq_length}  \
                --max_answer_length 150 \
                --doc_stride 128  \
                --cache_dir {cache_dir} \
                --output_dir {output_path} \
                --logging_dir {output_path} \
                --overwrite_output_dir  \
                --per_device_train_batch_size {batch_size} \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps {gradient_accum_steps}  \
                --warmup_steps {warmup_steps} \
                --weight_decay 0.01  \
                --fp16 \
                --qa_type span_based \
                --eval_accumulation_steps 1000
                ''',
            python=gs.PYTHON_EXE,
            file=f"{gs.SCRIPTS_ROOT}/subtask1/run_qa.py",
            dataset_name=f"'{gs.SCRIPTS_ROOT}/datasets/doc2dial/doc2dial.py'",
            output_path=self.output_path,
            max_seq_length=self.max_seq_length,
            num_epochs=self.num_epochs,
            model_name_or_path=self.model_name_or_path,
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            gradient_accum_steps = self.gradient_accum_steps,
            learning_rate = self.learning_rate,
            cache_dir=TODO()
        )

    def tasks(self):
        yield Task('run', resume='run', rqmt={
            'cpu': self.gpus,
            'gpu': self.gpus,
            'mem': 22 * self.gpus,
            'time': 4 + self.extra_time,
        })

class EvalSpanQAJob(Job):

    def __init__(
        self,
        gpus=1,
        max_seq_length=512,
        num_epochs=10,
        model_name_or_path="google/electra-base-discriminator",
        extra_time=0,
        batch_size=1,
        gradient_accum_steps=16,
        warmup_steps=1000,
        learning_rate=3e-5,
        qa_type="span_based",
        do_eval=True,
        do_predict=True,
        predict_on="test",
        **kwargs
        ):
        self.output_path = self.output_path("runs")
        self.gpus = gpus
        self.max_seq_length = max_seq_length
        self.num_epochs = num_epochs
        self.model_name_or_path = model_name_or_path
        self.extra_time = extra_time
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.gradient_accum_steps = gradient_accum_steps
        self.learning_rate = learning_rate
        self.qa_type = qa_type
        self.do_eval = do_eval
        self.do_predict = do_predict
        self.predict_on = predict_on

    def run(self):
        self.sh(
            '''{python} {file} \
                --dataset_name {dataset_name} \
                --dataset_config_name doc2dial_rc \
                --model_name_or_path {model_name_or_path} \
                {do_eval} \
                {do_predict} \
                --logging_steps 2000 \
                --save_steps 2000 \
                --learning_rate {learning_rate}  \
                --num_train_epochs {num_epochs} \
                --max_seq_length {max_seq_length}  \
                --max_answer_length 150 \
                --doc_stride 128  \
                --cache_dir {cache_dir} \
                --output_dir {output_path} \
                --logging_dir {output_path} \
                --overwrite_output_dir  \
                --per_device_train_batch_size {batch_size} \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps {gradient_accum_steps}  \
                --warmup_steps {warmup_steps} \
                --weight_decay 0.01  \
                --fp16 \
                --qa_type {qa_type} \
                --eval_accumulation_steps 1000 \
                {predict_on}
                ''',
            python=gs.PYTHON_EXE,
            file=f"{gs.SCRIPTS_ROOT}/subtask1/run_qa.py",
            dataset_name=f"'{gs.SCRIPTS_ROOT}/datasets/doc2dial/doc2dial.py'",
            output_path=self.output_path,
            max_seq_length=self.max_seq_length,
            num_epochs=self.num_epochs,
            model_name_or_path=self.model_name_or_path,
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            gradient_accum_steps = self.gradient_accum_steps,
            learning_rate = self.learning_rate,
            do_eval = "--do_eval" if self.do_eval else "",
            do_predict="--do_predict" if self.do_predict else "",
            qa_type=self.qa_type,
            predict_on = f"--predict_on {self.predict_on}" if self.do_predict else "",
            cache_dir=TODO()
        )

    def tasks(self):
        yield Task('run', resume='run', rqmt={
            'cpu': self.gpus,
            'gpu': self.gpus,
            'mem': 22 * self.gpus,
            'time': 32 + self.extra_time,
        })
        


class TrainQAJob(Job):
    __sis_hash_exclude__ = {
        'span_restricted_only_in_eval': None,
        'use_original_dataset': False,
    }

    def __init__(
        self,
        gpus=1,
        max_seq_length=512,
        name=None,
        qa_type=None,
        use_original_dataset=False,
        span_restricted_only_in_eval=None,
        **kwargs
        ):
        self.out_path = self.output_path("runs")
        self.gpus = gpus
        self.max_seq_length = max_seq_length
        self.name = name
        self.num_epochs = kwargs.get('num_epochs', 5)
        self.model_name_or_path = kwargs.get('model_name_or_path', 'google/electra-base-discriminator')
        self.save_steps = kwargs.get('save_steps', 2000)

        self.per_device_train_batch_size = kwargs.get('per_device_train_batch_size', 2)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 16)
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        self.learning_rate = kwargs.get('learning_rate', 3e-5)

        self.qa_type = qa_type
        self.biaffine_hidden_size = kwargs.get('biaffine_hidden_size', None)
        self.span_restricted = kwargs.get('span_restricted', False)
        self.time_rqmt = kwargs.get('time_rqmt', 32)

        self.predictions = self.output_path('runs/predictions.json')
        self.nbest_predictions = self.output_path('runs/nbest_predictions.json')

        self.use_original_dataset = use_original_dataset

    def run(self):
        if not os.path.exists('cache'):
            os.mkdir('cache')
        self.sh(
            '''{python} {file}/subtask1/run_qa.py \
                --dataset_name {dataset_name} \
                --dataset_config_name doc2dial_rc \
                --model_name_or_path {model_name_or_path} \
                --do_train \
                --do_eval \
                --logging_steps 500 \
                --save_steps {save_steps} \
                --learning_rate {learning_rate}  \
                --num_train_epochs {num_epochs} \
                --max_seq_length {max_seq_length}  \
                --max_answer_length 50 \
                --doc_stride 128  \
                --cache_dir {cache_dir} \
                --output_dir {output_path} \
                --logging_dir {output_path} \
                --overwrite_output_dir  \
                --per_device_train_batch_size {per_device_train_batch_size} \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps {gradient_accumulation_steps}  \
                --eval_accumulation_steps 1000 \
                --warmup_steps {warmup_steps} \
                --weight_decay 0.01  \
                --span_restricted {span_restricted} \
                --fp16 \
                {extra_args}''',
            python=gs.PYTHON_EXE,
            file=gs.SCRIPTS_ROOT,
            dataset_name=f'{gs.SCRIPTS_ROOT}/datasets/doc2dial/doc2dial{"_orig" if self.use_original_dataset else ""}.py',
            output_path=self.out_path,
            max_seq_length=self.max_seq_length,
            num_epochs=self.num_epochs,
            model_name_or_path=self.model_name_or_path,
            save_steps=self.save_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            span_restricted=self.span_restricted,
            cache_dir=TODO(),
            extra_args=(f" --qa_type {self.qa_type} " if self.qa_type is not None else "") + (f" --biaffine_hidden_size {self.biaffine_hidden_size} " if self.biaffine_hidden_size is not None else "")
        )

    def tasks(self):
        yield Task('run', resume='run', rqmt={
            'cpu': self.gpus,
            'gpu': self.gpus,
            'mem': 14 * self.gpus,
            'time': 32,
        })



class TrainLanguageGenerationJob(Job):

    def __init__(
        self,
        data_dir=f"{gs.SCRIPTS_ROOT}/subtask2/seq2seq_files",
        model_name_or_path=None,
        batch_size=4,
        gpus=1,
        max_length=1024,
        gradient_accum_steps=1,
        num_epochs=10,
        is_rag_model=False,
        n_best_list=None,
        do_eval=False,
        do_predict=False,
        seed=42,
        **kwargs
        ):

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.model_name_or_path = model_name_or_path
        self.output_path = self.output_path("runs")
        self.gpus=gpus
        self.max_length=max_length
        self.gradient_accum_steps = gradient_accum_steps
        self.is_rag_model = is_rag_model
        self.n_best_list = n_best_list
        self.do_eval = do_eval
        self.do_predict = do_predict
        self.num_epochs = num_epochs
        self.seed = seed

    def run(self):
        self.sh(
            '''{python} {file}/finetune_trainer.py \
                --data_dir {data_dir} \
                --cache_dir {cache_dir} \
                --output_dir {output_path} \
                --logging_dir {output_path} \
                --model_name_or_path {model_name_or_path} \
                --learning_rate 6.25e-6 \
                --adam_epsilon 1e-06 \
                --do_train \
                {do_eval} \
                {do_predict} \
                --per_device_train_batch_size={batch_size} \
                --per_device_eval_batch_size={batch_size} \
                --overwrite_output_dir \
                --adam_eps 1e-06 \
                --max_source_length {max_length} \
                --max_target_length 75 \
                --val_max_target_length 75 \
                --test_max_target_length 75 \
                --task translation \
                --warmup_steps 500 \
                --save_steps 2000 \
                --evaluation_strategy epoch \
                --predict_with_generate \
                --num_train_epochs {num_epochs} \
                --gradient_accumulation_steps {gradient_accum_steps} \
                --seed {seed} \
                {n_best_list} \
                {is_rag_model} \
                {add_ids_to_batch} \
                 ''',
            python=gs.PYTHON_EXE,
            file=f'{gs.SCRIPTS_ROOT}/subtask2',
            output_path=self.output_path,
            batch_size=self.batch_size,
            model_name_or_path=self.model_name_or_path,
            max_length=self.max_length,
            data_dir=self.data_dir,
            gradient_accum_steps=self.gradient_accum_steps,
            n_best_list=f"--nbest_list {self.n_best_list}" if self.n_best_list is not None else "",
            is_rag_model="--is_rag_model" if self.is_rag_model else "",
            add_ids_to_batch="--add_ids_to_batch" if self.is_rag_model else "",
            do_eval="--do_eval" if self.do_eval else "",
            do_predict="--do_predict" if self.do_predict else "",
            num_epochs=self.num_epochs,
            seed=self.seed,
            cache_dir=TODO()
        )

    def tasks(self):
        yield Task('run', resume='run', rqmt={
            'cpu': self.gpus,
            'gpu': self.gpus,
            'mem': 14 * self.gpus,
            'time': 32,
        })

class TrainMultipleChoiceJob(Job):

    def __init__(self, model_name_or_path, gpus=1, **kwargs):
        self.gpus = gpus
        self.model_name_or_path = model_name_or_path
        self.output_path = self.output_path("runs")

    def run(self):
        self.sh(
            '''{python} {file}/run_swag.py \
                --model_name_or_path {model_name_or_path} \
                --output_dir {output_path} \
                --pad_to_max_length \
                --cache_dir {cache_dir} \
                --do_train \
                --do_eval \
                --overwrite_output_dir \
                --per_device_train_batch_size=1 \
                --per_device_eval_batch_size=1 \
                --max_seq_length=512 \
                --gradient_accumulation_steps 16 \
                --num_train_epochs 10
                 ''',
            python=gs.PYTHON_EXE,
            file=f'{gs.SCRIPTS_ROOT}/retrieval',
            output_path=self.output_path,
            model_name_or_path=self.model_name_or_path,
            cache_dir=TODO()
        )

    def tasks(self):
        yield Task('run', resume='run', rqmt={
            'cpu': self.gpus,
            'gpu': self.gpus,
            'mem': 14 * self.gpus,
            'time': 32,
        })


class EvalLanguageGenerationJob(Job):

    def __init__(
        self,
        data_dir=f"{gs.SCRIPTS_ROOT}/subtask2/seq2seq_files",
        model_name_or_path=None,
        batch_size=4,
        gpus=1,
        max_length=1024,
        gradient_accum_steps=1,
        num_epochs=10,
        is_rag_model=False,
        n_best_list=None,
        do_eval=False,
        do_predict=True,
        seed=42,
        **kwargs
        ):

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.model_name_or_path = model_name_or_path
        self.output_path = self.output_path("runs")
        self.gpus=gpus
        self.max_length=max_length
        self.gradient_accum_steps = gradient_accum_steps
        self.is_rag_model = is_rag_model
        self.n_best_list = n_best_list
        self.do_eval = do_eval
        self.do_predict = do_predict
        self.num_epochs = num_epochs
        self.seed = seed

    def run(self):
        self.sh(
            '''{python} {file}/finetune_trainer.py \
                --data_dir {data_dir} \
                --cache_dir {cache_dir} \
                --output_dir {output_path} \
                --logging_dir {output_path} \
                --model_name_or_path {model_name_or_path} \
                {do_eval} \
                {do_predict} \
                --predict_on test \
                --per_device_train_batch_size={batch_size} \
                --per_device_eval_batch_size={batch_size} \
                --overwrite_output_dir \
                --adam_eps 1e-06 \
                --max_source_length {max_length} \
                --max_target_length 75 \
                --val_max_target_length 75 \
                --test_max_target_length 75 \
                --task translation \
                --evaluation_strategy epoch \
                --predict_with_generate \
                --gradient_accumulation_steps {gradient_accum_steps} \
                --seed {seed} \
                {n_best_list} \
                {is_rag_model} \
                {add_ids_to_batch} \
                 ''',
            python=gs.PYTHON_EXE,
            file=f'{gs.SCRIPTS_ROOT}/subtask2',
            output_path=self.output_path,
            batch_size=self.batch_size,
            model_name_or_path=self.model_name_or_path,
            max_length=self.max_length,
            data_dir=self.data_dir,
            gradient_accum_steps=self.gradient_accum_steps,
            n_best_list=f"--nbest_list {self.n_best_list}" if self.n_best_list is not None else "",
            is_rag_model="--is_rag_model" if self.is_rag_model else "",
            add_ids_to_batch="--add_ids_to_batch" if self.is_rag_model else "",
            do_eval="--do_eval" if self.do_eval else "",
            do_predict="--do_predict" if self.do_predict else "",
            num_epochs=self.num_epochs,
            seed=self.seed,
            cache_dir=TODO()
        )

    def tasks(self):
        yield Task('run', resume='run', rqmt={
            'cpu': self.gpus,
            'gpu': self.gpus,
            'mem': 14 * self.gpus,
            'time': 32,
        })

class EvalQAJob(Job):

    def __init__(
        self, 
        model_name_or_path, 
        predict_on_train=False, 
        predict_on="validation", 
        gpus=1, 
        n_best_size=100, 
        do_eval=False, 
        add_title_to_pred=False,
        extend_preds_by=0,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.predict_on_train = predict_on_train
        self.output_path = self.output_path("runs")
        self.gpus = gpus
        self.n_best_size = n_best_size
        self.predict_on = predict_on
        self.do_eval = do_eval
        self.add_title_to_pred = add_title_to_pred
        self.extend_preds_by=0

    def run(self):
        self.sh(
            '''
            {python} {file}/run_qa.py \
            --dataset_name '../datasets/doc2dial/doc2dial.py' \
            --dataset_config_name doc2dial_rc \
            --model_name_or_path {model} \
            {do_eval} \
            --do_predict \
            {predict_on_train} \
            --logging_steps 2000 \
            --save_steps 2000 \
            --learning_rate 3e-5  \
            --num_train_epochs 5 \
            --max_seq_length 512  \
            --max_answer_length 200 \
            --doc_stride 128  \
            --cache_dir {cache_dir} \
            --output_dir {output_path} \
            --overwrite_output_dir  \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 4  \
            --warmup_steps 1000 \
            --weight_decay 0.01  \
            --fp16 \
            --n_best_size {n_best_size} \
            --predict_on {predict_on} \
            {add_title_to_pred}
            ''',
            python=gs.PYTHON_EXE,
            file=f'{gs.SCRIPTS_ROOT}/subtask1',
            model=self.model_name_or_path,
            predict_on_train="--predict_on_train" if self.predict_on_train else "",
            output_path=self.output_path,
            n_best_size=self.n_best_size,
            predict_on=self.predict_on,
            do_eval="--do_eval" if self.do_eval else "",
            add_title_to_pred="--add_title_to_pred" if self.add_title_to_pred else "",
            cache_dir=TODO()
        )

    def tasks(self):
        yield Task('run', resume='run', rqmt={
            'cpu': self.gpus,
            'gpu': self.gpus,
            'mem': 14 * self.gpus,
            'time': 4,
        })

import json

from sisyphus import *


class CreateRagCheckpoint(Job):
    __sis_hash_exclude__ = {
        'new_hash': False
    }

    def __init__(self, encoder_model_name_or_path, generator_model_name_or_path, n_docs=20, new_hash=False):
        self.encoder_model_name_or_path = encoder_model_name_or_path
        self.generator_model_name_or_path = generator_model_name_or_path
        self.checkpoint = self.output_path('checkpoint', directory=True)

    def run(self):
        self.sh(
            '{python} {root}/subtask2/create_rag_checkpoint.py "{generator_model}" "{encoder_model}" "{output_checkpoint}"',
            python=gs.PYTHON_EXE,
            root=gs.SCRIPTS_ROOT,
            generator_model=self.generator_model_name_or_path,
            encoder_model=self.encoder_model_name_or_path,
            output_checkpoint=self.checkpoint,
        )

    def tasks(self):
        yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 8, 'time': 1})

class EnsembleQAPredictionsJob(Job):

    def __init__(self, file_paths, predictions, strategy="bayesian"):
        self.file_paths = file_paths
        self.predictions = predictions
        self.out_ensemble_predictions = self.output_path("runs")

    def read_f1s(self):
        f1_scores = []
        for file_path in self.file_paths:
            with open(os.path.join(file_path, "eval_results.txt"), "r") as f:
                content = f.readlines()
                f1 = float(content[-1].split("=")[-1].strip())
                f1_scores.append(f1)
        
        if not os.path.isdir(self.out_ensemble_predictions):
            os.mkdir(self.out_ensemble_predictions)

        with open(os.path.join(self.out_ensemble_predictions, "f1s.json"), "w") as f:
            json.dump(f1_scores, f)

    def run(self):
        with open(os.path.join(self.out_ensemble_predictions, "f1s.json"), "r") as f:
            f1_scores = json.load(f)
            print(f1_scores)

        self.sh(
            '{python} {file} {f1s} {predictions} {output_dir}',
            python=gs.PYTHON_EXE,
            file="/u/daheim/sharedtask-dialdoc2021/scripts/subtask1/combine_preds.py",
            f1s=f"--f1s {' '.join([str(f1) for f1 in f1_scores])}",
            predictions=f"--preds {' '.join(self.predictions)}",
            output_dir=f"--output_dir {self.out_ensemble_predictions}"
        )

    def tasks(self):
        yield Task('read_f1s', rqmt={'cpu': 1, 'gpu': 0, 'mem': 2, 'time': 1}, mini_task=True)
        yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 2, 'time': 1}, mini_task=True)

class CalculateQAErrorJob(Job):
    __sis_hash_exclude__ = {
        'report_exact_math_at_5': False,
    }

    def __init__(self, predictions, split='validation', report_exact_math_at_5=False):
        self.predictions = predictions
        self.split = split

        self.out_report = self.output_path('scores.txt')

        self.report_exact_math_at_5 = report_exact_math_at_5

    def run(self):
        sys.path.insert(-1, gs.SCRIPTS_ROOT)
        from sharedtask_utils import sharedtask1_metrics

        if not os.path.exists('cache'):
            os.mkdir('cache')

        prediction_json = self.predictions
        split = self.split
        final_scores = sharedtask1_metrics(prediction_json=prediction_json, split=split, cache_dir=os.path.realpath('cache'), report_exact_math_at_5=self.report_exact_math_at_5)

        with open(self.out_report, 'w') as fp:
            json.dump(final_scores, fp)

    def tasks(self):
        yield Task('run', mini_task=True)