@REM # Copyright (c) Microsoft Corporation.
@REM # Licensed under the MIT license.

set DATA_DIR=data/
set OUTPUT_DIR=test/
set TOKENIZERS_PARALLELISM=false
set LOADMODEL_ERROR=0

@REM # Please refer to task.py for available options.
@REM
@REM # train an ELECTRA-large model using pytorch DDP (clear previous existing models)
set CUDA_VISIBLE_DEVICES=0,1
@REM python -m torch.distributed.launch --nproc_per_node=2 task.py --append_descr 1 --data_version csqa_ret_3datasets --lr 1e-5 --append_answer_text 1 --weight_decay 0.01 --preset_model_type electra --batch_size 2 --max_seq_length 50 --num_train_epochs 10 --save_interval_step 2 --continue_train --print_number_per_epoch 2 --vary_segment_id --seed 42 --warmup_proportion 0.1 --optimizer_type adamw --ddp --print_loss_step 10 --clear_output_folder
@REM
@REM # train a deberta v2 xlarge (900M) model using deepspeed, testing the pipeline
@REM deepspeed task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --lr 5e-6 --append_answer_text 1 --weight_decay 0 --preset_model_type debertav2-xlarge --batch_size 1 --max_seq_length 50 --num_train_epochs 15 --save_interval_step 4 --continue_train --print_number_per_epoch 1 --vary_segment_id --seed 42 --warmup_proportion 0.1 --optimizer_type adamw --ddp --deepspeed --test_mode --clear_output_folder
@REM
@REM # reproduce results of DeBERTa v3 in paper
@REM deepspeed task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --lr 4e-6 --append_answer_text 1 --weight_decay 0.1 --preset_model_type debertav3 --batch_size 4 --max_seq_length 512 --num_train_epochs 15 --save_interval_step 4 --continue_train --print_number_per_epoch 1 --vary_segment_id --seed 42 --warmup_proportion 0.1 --optimizer_type adamw --ddp --deepspeed --freq_rel 1

python task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --lr 4e-6 --append_answer_text 1 --weight_decay 0.1 --preset_model_type debertav3-xsmall --batch_size 2 --max_seq_length 512 --num_train_epochs 15 --save_interval_step 4 --continue_train --print_number_per_epoch 1 --vary_segment_id --seed 42 --warmup_proportion 0.1 --optimizer_type adamw --freq_rel 1
