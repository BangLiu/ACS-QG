
cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
CUDA_VISIBLE_DEVICES=1 python3 QG_gpt2_train.py \
    --eval_before_start \
    --n_epochs 4 \
    --model_name_or_path gpt2 \
    --output_dir ../../../file/QG/gpt2_question_generation
