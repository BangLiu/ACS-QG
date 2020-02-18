# STEP 3: perform data augmentation. Raw input data -> sentences txt file -> augmented sentences pkl file
# run each code piece in one machine. process data in parallel.


# debug
cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/SQuAD2.0/"
output_path="../../../../Datasets/processed/SQuAD2.0/"
data_type="squad"
data_file_prefix="train"
st_idx=0
ed_idx=50000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.4e8b.debug.json" \
    --debug



# squad data
cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/SQuAD2.0/"
output_path="../../../../Datasets/processed/SQuAD2.0/"
data_type="squad"
data_file_prefix="train"
st_idx=0
ed_idx=50000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/SQuAD2.0/"
output_path="../../../../Datasets/processed/SQuAD2.0/"
data_type="squad"
data_file_prefix="train"
st_idx=50000
ed_idx=92210
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



# wiki data
cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=0
ed_idx=50000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=50000
ed_idx=100000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=100000
ed_idx=150000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=150000
ed_idx=200000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=200000
ed_idx=250000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=250000
ed_idx=300000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=300000
ed_idx=350000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=350000
ed_idx=400000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=400000
ed_idx=450000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=450000
ed_idx=500000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=500000
ed_idx=550000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=550000
ed_idx=600000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=600000
ed_idx=650000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=650000
ed_idx=700000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=700000
ed_idx=750000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=750000
ed_idx=800000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=800000
ed_idx=850000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=850000
ed_idx=900000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=900000
ed_idx=9500000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"



cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=950000
ed_idx=1000000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/2batchsize/ \
    --filename "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.cache.qg.gpt2.pth" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.json"
