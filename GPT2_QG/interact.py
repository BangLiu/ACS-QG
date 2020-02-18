# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Forked and modified by Bang Liu
"""

# test on SQuAD1.1-Zhou dev set
cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/8batchsize/ \
    --filename ../../../../Datasets/original/SQuAD1.1-Zhou/dev.txt \
    --filecache ../../../../Datasets/processed/SQuAD1.1-Zhou/dev.cache.pth \
    --data_type squad \
    --output_file ../../../../Datasets/processed/SQuAD1.1-Zhou/dev.qg.generated.gpt2.debug.json \
    --debug


# test on SQuAD1.1-Zhou dev set with greedy decoding
cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/8batchsize/ \
    --filename ../../../../Datasets/original/SQuAD1.1-Zhou/dev.txt \
    --filecache ../../../../Datasets/processed/SQuAD1.1-Zhou/dev.cache.pth \
    --data_type squad \
    --output_file ../../../../Datasets/processed/SQuAD1.1-Zhou/dev.qg.generated.gpt2.debug.json \
    --debug \
    --no_sample


We can test on different checkpoints, and find out the best model.



# test on Wiki
cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/Wiki10000/"
output_path="../../../../Datasets/processed/Wiki10000/"
data_type="wiki10000"
data_file_prefix="wiki10000"
st_idx=900000
ed_idx=950000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/8batchsize/ \
    --filename "" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.debug.json" \
    --debug


# test on SQuAD
cd /ceph4/bangliu/FQG/src/model/FactorizedQG/
input_path="../../../../Datasets/original/SQuAD2.0/"
output_path="../../../../Datasets/processed/SQuAD2.0/"
data_type="squad"
data_file_prefix="train"
st_idx=0
ed_idx=50000
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py \
    --model_type gpt2 \
    --model_name_or_path ../../../file/QG/gpt2_question_generation/4epochs/8batchsize/ \
    --filename "" \
    --filecache "$output_path${data_file_prefix}.sentences.augmented.${st_idx}_${ed_idx}.pkl" \
    --data_type augmented_sents \
    --output_file "$output_path${data_file_prefix}.qa.${st_idx}_${ed_idx}.qg.generated.gpt2.debug.json" \
    --debug

"""
import os
import json
import random
import tqdm
from argparse import ArgumentParser
from pprint import pformat
import torch
import torch.nn.functional as F
from datetime import datetime
from shutil import copy2, move
from common.constants import EXP_PLATFORM
# !!! for running experiments on Venus
if EXP_PLATFORM.lower() == "venus":
    from pip._internal import main as pipmain
    pipmain(["install", "transformers"])
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .train import SPECIAL_TOKENS
from .train import build_para_only_input_from_segments, build_acsq_only_input_from_segments
from .dataloader import get_dataset
from metric.text_generation_metrics import compute_metrics_by_file


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(inst, tokenizer, model, args, para_cache):
    """ Sample to generate questions using finetuned QG model """
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    inst['original_question'] = inst['question']  # when generate, if no question, we can set question = []
    inst['question'] = []

    # Ignore the paragraph while building the input instance and token type ids
    instance, _ = build_acsq_only_input_from_segments(inst, tokenizer, with_eos=False)  # as this is used for generate, with_eos is False.
    input_ids = torch.tensor(instance['input_ids'], device=args.device).unsqueeze(0)
    token_type_ids = torch.tensor(instance['token_type_ids'], device=args.device).unsqueeze(0)
    # above we get `<answer> answer <clue> clue <style> style <question>`, this is start point to generate question

    # Initialize the past using the paragraph cache hidden representations
    past = para_cache["hidden_states"]

    prev = None
    # This will be either <question>. Previously, it is designed to incorporate question type info.
    # But here we do not need it for current project
    token_type = instance['token_type_ids'][-1]

    for i in range(args.max_length):
        if i != 0:
            # In the first step of decoding, we want to look at the entire answer
            # In the subsequent steps, we can just cache the hidden representations from previous steps
            input_ids = prev.unsqueeze(0)
            token_type_ids = torch.tensor([token_type]).unsqueeze(0).to(args.device)

        logits, past = model(input_ids, token_type_ids=token_type_ids, past=past)

        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break

        inst['question'].append(prev.item())

    return inst


def run():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt2", help="gpt or gpt2")
    parser.add_argument("--model_name_or_path", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--model_name", type=str, default="", help="Path, url or short name of the model")
    # Notice: here model_name is used to let us test checkpoint performance.
    parser.add_argument("--calc_tg_metrics", action='store_true', help="whether we need to calculate text generation metrics: bleu, etc.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--filename", type=str, default="", help="File to use for decoding")
    parser.add_argument("--filecache", type=str, default="", help="Cache File to use for decoding")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument('--data_type', default='augmented_sents', type=str, help='data type')
    parser.add_argument('--output_file', default='generated_questions.json', type=str, help='output json file')
    parser.add_argument("--save_freq", type=int, default=2000, help="maximum number of questions to generate")
    parser.add_argument("--debug", action='store_true', help="If true we use debug mode")
    parser.add_argument("--debug_num", type=int, default=20, help="debug num")
    args = parser.parse_args()

    print(pformat(args))

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Get pretrained model and tokenizer")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    if args.model_name != "":
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)
    model.eval()

    start = datetime.now()
    # data = get_positional_dataset_from_file(tokenizer, args.filename, filetype=args.data_type, debug=args.debug)
    data = get_dataset(tokenizer, args.filecache, args.filename, filetype=args.data_type, debug=args.debug)
    print(("Time of get_positional_dataset_from_file: {}").format(datetime.now() - start))

    final_output_dict = {
        "version": "fqg",
        "data": [{
            "paragraphs": []
        }]
    }
    para_cache = {
        "index": None,
        "hidden_states": None
    }
    processed_para_indexs = []
    para_index2output_dict_paras_index = {}

    # process each instance
    question_number = 0
    start = datetime.now()
    for inst in tqdm.tqdm(data):  # !!! each inst contains one different paragraph
        try:
            with torch.no_grad():
                inst["para_index"] = inst["para_id"]
                para_index = inst["para_index"]  # !!! NOTICE: we can use our sid as this when generate new questions
                # Questions from the same paragraph all appear together
                # We can re-use the paragraph hidden representations for different questions in the same paragraph
                if para_index != para_cache["index"]:
                    # Since we have moved to a new paragraph, generate its cache
                    para_cache["hidden_states"] = None
                    # Ignore the answer and question while building the input
                    instance, _ = build_para_only_input_from_segments(inst, tokenizer)  # !!!! this is designed to re-use paragraph.
                    input_ids = torch.tensor(instance['input_ids'], device=args.device).unsqueeze(0)
                    token_type_ids = torch.tensor(instance['token_type_ids'], device=args.device).unsqueeze(0)

                    # Run a forward pass to generate the para caches
                    _, para_cache["hidden_states"] = model(input_ids, token_type_ids=token_type_ids)

                # Sample a question using the paragraph cache
                try:
                    output = sample_sequence(inst, tokenizer, model, args, para_cache)
                except:  # !!! NOTICE: sometimes (very rare case) the sample_sequence got size mismatch in modeling_gpt2
                    continue

            original_paragraph = tokenizer.decode(output['paragraph'])
            generated_question = tokenizer.decode(output['question'], skip_special_tokens=True)
            original_answer = tokenizer.decode(output['answer'], skip_special_tokens=True)
            para_index = inst['para_index']
            para_cache["index"] = inst['para_index']

            # verify whether the answer position is correct, since this will be utilized for filtering
            original_ans_position = output["answer_position"]
            if original_paragraph[output["answer_position"]:output["answer_position"] + len(original_answer)] != original_answer:
                # This should never be executed, only used as a last resort
                print("Answer mismatch!")
                print("answer by position: ", original_paragraph[output["answer_position"]:output["answer_position"] + len(original_answer)])
                print("original_answer: ", original_answer)
                new_answer_position = original_paragraph.find(original_answer)
                if new_answer_position > 0:  # not -1
                    output["answer_position"] = new_answer_position
                    original_ans_position = output["answer_position"]
                    print("After revised answer position: ", original_paragraph[output["answer_position"]:output["answer_position"] + len(original_answer)])
                else:
                    continue
                # original_ans_position = original_paragraph.index(original_answer)

            # Output in a SQUAD-like format with questions clumped together under their parent paragraph
            original_question = tokenizer.decode(output['original_question'], skip_special_tokens=True)
            if para_index in processed_para_indexs:  # check whether is an existing paragraph.
                # verify whether the paragraph text is identical
                output_para_index = para_index2output_dict_paras_index[para_index]
                assert original_paragraph == final_output_dict["data"][0]["paragraphs"][output_para_index]['context']
                # append the question answer pair
                final_output_dict["data"][0]["paragraphs"][output_para_index]['qas'].append({
                    'id': 'question_%d' % question_number,
                    'question': generated_question,
                    'original_question': original_question,
                    'answers': [{
                        'text': original_answer,
                        'answer_start': original_ans_position,
                    }],
                    'clue': tokenizer.decode(inst["clue"]),
                    'clue_start': inst["clue_start"],
                    'ques_type': inst['ques_type'],
                    'is_impossible': False
                })
            else:
                # add a new question to the list of QA pairs
                final_output_dict['data'][0]['paragraphs'].append({
                    'context': original_paragraph,
                    'qas': [{
                        'id': 'question_%d' % question_number,
                        'question': generated_question,
                        'original_question': original_question,
                        'answers': [{
                            'text': original_answer,
                            'answer_start': original_ans_position,
                        }],
                        'clue': tokenizer.decode(inst["clue"]),
                        'clue_start': inst["clue_start"],
                        'ques_type': inst['ques_type'],
                        'is_impossible': False
                    }]
                })
                processed_para_indexs.append(para_index)
                para_index2output_dict_paras_index[para_index] = len(processed_para_indexs) - 1

            question_number += 1

            if len(processed_para_indexs) % args.save_freq == 0:
                if os.path.isfile(args.output_file):
                    move(args.output_file, args.output_file + ".copy.json")
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(final_output_dict, f, indent=4, ensure_ascii=False)
                    print("saved generated questions.", question_number)

            if args.debug and question_number >= args.debug_num:
                break
        except:
            continue

    print(("Time of generate {} questions: {}").format(question_number, datetime.now() - start))

    if os.path.isfile(args.output_file):
        move(args.output_file, args.output_file + ".copy.json")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_output_dict, f, indent=4, ensure_ascii=False)
        # f.write(json.dumps(final_output_dict))

    #!!! calc bleu, etc. if original question is not empty
    if args.calc_tg_metrics:
        output_prefix = args.model_name if args.model_name != "" else args.model_name_or_path
        gold_file = output_prefix + "gold_gpt2.txt"
        pred_file = output_prefix + "pred_gpt2.txt"
        fgold = open(gold_file, "w", encoding="utf-8")
        fpred = open(pred_file, "w", encoding="utf-8")
        data_to_write = final_output_dict["data"][0]['paragraphs']
        for d in data_to_write:
            fgold.write(d["qas"][0]["original_question"].rstrip() + "\n")
            fpred.write(d["qas"][0]["question"].rstrip() + "\n")
        fgold.close()
        fpred.close()
        ret_scores = compute_metrics_by_file([gold_file], pred_file)
        print(ret_scores)


if __name__ == "__main__":
    run()
