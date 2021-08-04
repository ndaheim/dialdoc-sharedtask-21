# Copyright 2020 The HuggingFace Team. All rights reserved.
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

python3 run_swag.py \
  --model_name_or_path allenai/longformer-base-4096 \
  --output_dir $YOUR_OUTPUT_DIR \
  --pad_to_max_length \
  --cache_dir $YOUR_CACHE_DIR \
  --do_train \
  --overwrite_output_dir \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --max_seq_length=2048
