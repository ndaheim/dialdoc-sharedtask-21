import os
import pathlib
import sys
import copy
from transformers import AutoTokenizer

sys.setrecursionlimit(2000)

# ------------------------------ Sisyphus -------------------------------------

from sisyphus import *
import sisyphus.toolkit as tk

Path = tk.Path

# ------------------------------ Recipes --------------------------------------

from dialdoc.dialdoc import *

def TODO():
  raise Exception("Needs to be implemented")

async def async_main():
  num_epochs = 10
  train_jobs = []
  predictions = []
  qa_types = ["default", "span_based"]

  for model in ["roberta-base"]:
    for gradient_accum_steps in [4]:
      for warmup_steps in [500]:
        for qa_type in qa_types:

          train_job = TrainQAJob(
            gpus=1, 
            model_name_or_path=model, 
            max_seq_length=512, 
            batch_size=2, 
            num_epochs=num_epochs, 
            extra_time=5,
            gradient_accum_steps=gradient_accum_steps,
            warmup_steps=warmup_steps,
            qa_type=qa_type,
          )
          tk.register_output(f"{model}-{qa_type}-bsz4-{gradient_accum_steps}-ga-steps-{warmup_steps}-wa-steps-{num_epochs}epochs", train_job.out_path)

          train_jobs.append(train_job.out_path)
          
          eval_qa_on_test = EvalSpanQAJob(model_name_or_path=train_job.out_path, do_eval=False, predict_on="test", qa_type=qa_type)
          predictions.append(eval_qa_on_test)
          tk.register_output(f"{model}-bsz4-{gradient_accum_steps}-ga-steps-{warmup_steps}-wa-steps-{num_epochs}epochs_on_test", eval_qa_on_test.output_path)

  nbest_files = [os.path.join(prediction_path.output_path, "nbest_predictions.json") for prediction_path in predictions]
  ensemble_job = EnsembleQAPredictionsJob(train_jobs, nbest_files)
  tk.register_output("subtask1_ensemble", ensemble_job.out_ensemble_predictions)

  data_dir = WriteSeq2SeqFilesJob().out_path 
  train_bart_job = TrainLanguageGenerationJob(model_name_or_path="facebook/bart-base", data_dir=data_dir, max_length=1024, batch_size=4, dummy="asd")
  tk.register_output("subtask2_bart_base", train_bart_job.output_path)
  
  for prediction, qa_type in zip(predictions, qa_types):
    data_dir_test = WriteSeq2SeqFilesJob(predictions=prediction.output_path).out_path
    eval_bart_on_test = EvalLanguageGenerationJob(model_name_or_path=train_bart_job.output_path ,data_dir=data_dir_test, predict_on="test")
    tk.register_output(f"subtask2-bart-base-on-test-roberta-base-{qa_type}", eval_bart_on_test.output_path)

  # initial_rag_checkpoint = CreateRagCheckpoint(
  #   'roberta-base', 
  #   'facebook/bart-base'
  # ).checkpoint
  # tk.register_output("rag_bart_roberta", initial_rag_checkpoint)

  # rag_checkpoint = "/work/smt4/thulke/daheim/rag_checkpoint"
  # rag_sequences_wo_context = TODO()
  # nbest = TOOD() # references n-best list from first subtask
  # train_rag_job = TrainLanguageGenerationJob(
  #   data_dir=rag_sequences_wo_context,
  #   model_name_or_path=rag_checkpoint,
  #   is_rag_model=True,
  #   n_best_list=nbest,
  #   batch_size=1,
  #   gradient_accum_steps=8,
  #   do_eval=True,
  #   do_predict=False,
  # )
  # eval_qa = EvalQAJob(model_name_or_path="/u/daheim/sharedtask-dialdoc2021/work/dialdoc/TrainQAJob.htwh5iPgv0AM/output/runs/", do_eval=True)
  # tk.register_output("subtask2_rag", train_rag_job.output_path)


async def py():
  await async_main()
