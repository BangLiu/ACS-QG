import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--train_dataset_path", type=str, default="",
    help="Path of the train/dev split in the dataset. Default files are instances_train.pickle, instances_dev.pickle."
)
parser.add_argument(
    "--dev_dataset_path", type=str, default="",
    help="Path of the train/dev split in the dataset. Default files are instances_train.pickle, instances_dev.pickle."
)
parser.add_argument(
    "--filetype", type=str, default="squad",
    help="Path of the train/dev split in the dataset. Default files are instances_train.pickle, instances_dev.pickle."
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Output directory to store tensorboard logs and saved models."
)
parser.add_argument(
    "--dataset_cache", type=str, default="",
    help="Path to store the dataset caches. There is a different cache for GPT and GPT2"
)
parser.add_argument(
    "--model_name_or_path", type=str, default="gpt2", help="Path, url or short name of the model"
)
parser.add_argument(
    "--train_batch_size", type=int, default=4, help="Batch size for training"
)
parser.add_argument(
    "--valid_batch_size", type=int, default=4, help="Batch size for validation"
)
parser.add_argument(
    "--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps"
)
parser.add_argument(
    "--lr", type=float, default=6.25e-5, help="Learning rate"
)
parser.add_argument(
    "--max_norm", type=float, default=1.0, help="Clipping gradient norm"
)
parser.add_argument(
    "--n_epochs", type=int, default=3, help="Number of training epochs"
)
parser.add_argument(
    "--eval_before_start", action='store_true', help="If true start with a first evaluation before training"
)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
)
parser.add_argument(
    "--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)"
)
parser.add_argument(
    "--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)"
)
parser.add_argument(
    "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument(
    '--data_type', default='squad', type=str, help='data type')
parser.add_argument(
    "--debug", action='store_true', help="If true we use debug mode"
)
