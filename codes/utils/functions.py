import argparse
import ujson
import csv
from omegaconf import OmegaConf
from tqdm import tqdm


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    _args = parser.parse_args()
    args = OmegaConf.load(_args.config)
    return args


def setup_parser_override():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dim', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--backbone', type=str, default=None, choices=['clip', 'bert_resnet'])
    parser.add_argument('--percentage', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    _args = parser.parse_args()
    args = OmegaConf.load(_args.config)
    if _args.task is not None:
        args.task = _args.task
        print('task : ', args.task)
    if _args.lr is not None:
        args.lr = _args.lr
        print('lr : ', args.lr)
    if _args.dim is not None:
        args.model.dim = _args.dim
        print('dim : ', args.model.dim)
    if _args.seed is not None:
        args.seed = _args.seed
        print('seed : ', args.seed)
    if _args.backbone is not None:
        args.backbone = _args.backbone
        print('backbone : ', args.backbone)
    if _args.percentage is not None:
        args.data.percentage = _args.percentage
        print('percentage : ', args.data.percentage)
    if _args.epoch is not None:
        args.trainer.max_epochs = _args.epoch
        print('epoch : ', args.trainer.max_epochs)
    if _args.run_name is not None:
        args.run_name = _args.run_name
        print('run_name : ', args.run_name)
    if _args.checkpoint is not None:
        args.checkpoint = _args.checkpoint
        print('checkpoint : ', args.checkpoint)
    return args


def load_json_file(filepath):
    data = []
    if isinstance(filepath, str):
        with open(filepath, 'r', encoding='utf-8') as f:
            d = ujson.load(f)
            data.extend(d)
    elif isinstance(filepath, list):
        for path in filepath:
            with open(path, 'r', encoding='utf-8') as f:
                d = ujson.load(f)
                data.extend(d)
    return data


def load_jsonl_file(filepath, desc='', key=None):
    if key is None:
        data = []
    else:
        data = dict()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=desc):
            item = ujson.loads(line)
            if key:
                item_key = item.get(key)
                data[item_key] = item
            else:
                data.append(item)
    return data


def load_candidate(filepath):
    mention_to_candidate = dict()
    with open(filepath, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            mention_id = row[0]
            candidates = row[1:]
            mention_to_candidate[mention_id] = candidates
    return mention_to_candidate
