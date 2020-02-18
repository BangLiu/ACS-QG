# This script we should repeat it for different parts of data
# For example:
# 1. replace 'da_start_index 0' with 'da_start_index 10000'
# 2. replace 'da_end_index 10000' with 'da_end_index 20000'
# 3. replace '0_10000' with '10000_20000'
# Similarly, 20000~30000, 30000~40000, 40000~50000, .....


# # STEP 3: perform data augmentation. Raw input data -> sentences txt file -> augmented sentences pkl file

python3 DA_main.py \
        --da_task sentences2augmented_sentences \
        --da_input_type wiki10000 \
        --da_input_file ../../../../Datasets/original/Wiki10000/wiki10000.json \
        --da_sentences_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.txt \
        --da_paragraphs_file ../../../../Datasets/processed/Wiki10000/wiki10000.paragraphs.txt \
        --da_augmented_sentences_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.pkl \
        --da_start_index 0 \
        --da_end_index 10000

python3 DA_main.py \
        --da_task sentences2augmented_sentences \
        --da_input_type squad \
        --da_input_file ../../../../Datasets/original/SQuAD2.0/train-v2.0.json \
        --da_sentences_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.txt \
        --da_paragraphs_file ../../../../Datasets/processed/SQuAD2.0/train.paragraphs.txt \
        --da_augmented_sentences_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.pkl \
        --da_start_index 0 \
        --da_end_index 10000

# # STEP 4: use trained FQG model to generate new QG data using augmented sentences
# prepro: it doesn't need GPU
python3 QG_augment_main.py \
        --not_processed_data  \
        --batch_size 8 \
        --epochs 10 \
        --copy_type hard-oov \
        --copy_loss_type 1 \
        --use_style_info \
        --use_clue_info \
        -beam_size 20 \
        --use_refine_copy_tgt_src \
        --da_augmented_sentences_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.pkl \
        --qg_augmented_sentences_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.pkl \
        --qg_result_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.output.txt \
        --da_paragraphs_file ../../../../Datasets/processed/SQuAD2.0/train.paragraphs.txt \
        --qa_data_file ../../../../Datasets/processed/SQuAD2.0/train.qa.0_10000.txt \
        --mode prepro


python3 QG_augment_main.py \
        --not_processed_data  \
        --batch_size 8 \
        --epochs 10 \
        --copy_type hard-oov \
        --copy_loss_type 1 \
        --use_style_info \
        --use_clue_info \
        -beam_size 20 \
        --use_refine_copy_tgt_src \
        --da_augmented_sentences_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.pkl \
        --qg_augmented_sentences_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.pkl \
        --qg_result_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.output.txt \
        --da_paragraphs_file ../../../../Datasets/processed/Wiki10000/wiki10000.paragraphs.txt \
        --qa_data_file ../../../../Datasets/processed/Wiki10000/wiki10000.qa.0_10000.txt \
        --mode prepro

# generate: needs GPU
python3 QG_augment_main.py \
        --not_processed_data  \
        --batch_size 8 \
        --epochs 10 \
        --copy_type hard-oov \
        --copy_loss_type 1 \
        --use_style_info \
        --use_clue_info \
        -beam_size 20 \
        --use_refine_copy_tgt_src \
        --da_augmented_sentences_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.pkl \
        --qg_augmented_sentences_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.pkl \
        --qg_result_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.output.txt \
        --da_paragraphs_file ../../../../Datasets/processed/SQuAD2.0/train.paragraphs.txt \
        --qa_data_file ../../../../Datasets/processed/SQuAD2.0/train.qa.0_10000.txt


python3 QG_augment_main.py \
        --not_processed_data  \
        --batch_size 8 \
        --epochs 10 \
        --copy_type hard-oov \
        --copy_loss_type 1 \
        --use_style_info \
        --use_clue_info \
        -beam_size 20 \
        --use_refine_copy_tgt_src \
        --da_augmented_sentences_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.pkl \
        --qg_augmented_sentences_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.pkl \
        --qg_result_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.output.txt \
        --da_paragraphs_file ../../../../Datasets/processed/Wiki10000/wiki10000.paragraphs.txt \
        --qa_data_file ../../../../Datasets/processed/Wiki10000/wiki10000.qa.0_10000.txt


sort ../../../../Datasets/processed/SQuAD2.0/train.qa.0_10000.txt | uniq  > ../../../../Datasets/processed/SQuAD2.0/train.qa.0_10000.uniq.txt
sort ../../../../Datasets/processed/Wiki10000/wiki10000.qa.0_10000.txt | uniq > ../../../../Datasets/processed/Wiki10000/wiki10000.qa.0_10000.uniq.txt

# STEP 5: use trained entailment model to append entailment score column
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path ../../../file/ET/models/xlnet-base-cased/ \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir ../../../file/ET/glue_data/squad-rte/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir ../../../file/ET/et_outdir/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file ../../../../Datasets/processed/SQuAD2.0/train.qa.0_10000.uniq.txt \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file ../../../../Datasets/processed/SQuAD2.0/train.qa.0_10000.entail.txt

python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path ../../../file/ET/models/xlnet-base-cased/ \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir ../../../file/ET/glue_data/squad-rte/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir ../../../file/ET/et_outdir/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file ../../../../Datasets/processed/Wiki10000/wiki10000.qa.0_10000.uniq.txt \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file ../../../../Datasets/processed/Wiki10000/wiki10000.qa.0_10000.entail.txt

# STEP 6: perform data evaluation to filter low-quality data samples and tag data samples with quality metrics: language model, entailment model, language complexity
python3 DE_main.py \
        --input_file  ../../../../Datasets/processed/SQuAD2.0/train.qa.0_10000.entail.txt \
        --input_augmented_pkl_file ../../../../Datasets/processed/SQuAD2.0/train.sentences.augmented.0_10000.processed.pkl \
        --output_file ../../../../Datasets/processed/SQuAD2.0/train.qa.0_10000.entail.de.txt

python3 DE_main.py \
        --input_file  ../../../../Datasets/processed/Wiki10000/wiki10000.qa.0_10000.entail.txt \
        --input_augmented_pkl_file ../../../../Datasets/processed/Wiki10000/wiki10000.sentences.augmented.0_10000.processed.pkl \
        --output_file ../../../../Datasets/processed/Wiki10000/wiki10000.qa.0_10000.entail.de.txt
