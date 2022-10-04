@REM # Copyright (c) Microsoft Corporation.
@REM # Licensed under the MIT license.
set DATA_DIR=data/
set OUTPUT_DIR=test/

@REM # make predictions on test set for a model trained with Pytorch DDP
@REM python task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --append_answer_text 1 --model_type electra --batch_size 1 --max_seq_length 512 --vary_segment_id --bert_model_dir test/ --mission output --predict_dir $OUTPUT_DIR/prediction/ --pred_file_name pred_test.csv --bert_vocab_dir google/electra-large-discriminator
@REM
@REM
@REM # make predictions on test set for a model trained with DeepSpeed
@REM deepspeed --include="localhost:0" task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --append_answer_text 1 --model_type debertav2 --batch_size 1 --max_seq_length 512 --vary_segment_id --ddp --deepspeed --bert_model_dir test/ --predict_dir $OUTPUT_DIR/prediction/ --pred_file_name pred_test.csv --mission output --bert_vocab_dir microsoft/deberta-v2-xxlarge --deepspeed_config debertav2-test

python task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --append_answer_text 1 --model_type debertav2 --batch_size 1^
        --max_seq_length 512 --vary_segment_id --bert_model_dir test --mission output --predict_dir $OUTPUT_DIR/prediction/ --pred_file_name pred_dev.csv^
        --bert_vocab_dir microsoft/deberta-v3-xsmall --predict_dev

