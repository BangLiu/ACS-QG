
python3 QG_main.py \
        --mode train \
        --batch_size 8 \
        --epochs 10 \
        --copy_type hard-oov \
        --copy_loss_type 1 \
        --use_style_info \
        --use_clue_info \
        -beam_size 20 \
        --use_refine_copy_tgt_src
# NOTICE: if you haven't process data, add --not_processed_data
