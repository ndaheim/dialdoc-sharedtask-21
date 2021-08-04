python run_qa.py \
 --dataset_name '../datasets/doc2dial/doc2dial.py' \
 --dataset_config_name doc2dial_rc \
 --model_name_or_path google/electra-base-discriminator \
 --do_train \
 --do_eval \
 --logging_steps 2000 \
 --save_steps 2000 \
 --learning_rate 3e-5  \
 --num_train_epochs 5 \
 --max_seq_length 512  \
 --max_answer_length 50 \
 --doc_stride 128  \
 --cache_dir $YOUR_CACHE_DIR \
 --output_dir $YOUR_OUTPUT_DIR \
 --overwrite_output_dir  \
 --per_device_train_batch_size 8 \
 --per_device_train_batch_size 8 \
 --gradient_accumulation_steps 4  \
 --warmup_steps 1000 \
 --weight_decay 0.01  \
 --fp16 
