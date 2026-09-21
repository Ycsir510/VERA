#!/usr/bin/env python3
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from codes.lit_model import LightModule
from codes.utils.functions import setup_parser_override


if __name__ == '__main__':
    args = setup_parser_override()
    root = Path(__file__).resolve().parents[1]
    args.run_name = f'VERA_{args.task}_s{args.seed}_{args.run_name}'
    args.output = {
        'checkpoints': str(root / 'checkpoints'),
        'runs': str(root / 'runs'),
        'rank_save': str(root / 'rank_save'),
    }
    for path in args.output.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError('VERA requires at least one visible CUDA device')

    module = LightModule(args)
    logger = CSVLogger(args.output['runs'], name=args.run_name, flush_logs_every_n_steps=30)
    checkpoint = ModelCheckpoint(
        dirpath=str(Path(args.output['checkpoints']) / args.task / args.run_name),
        filename='epoch{epoch:02d}-step{step}', monitor='Val/mrr', mode='max',
        save_top_k=1, save_last=True, save_weights_only=True,
    )
    early_config = getattr(args, 'early_stopping', None)
    early_stop = EarlyStopping(
        monitor='Val/mrr', mode='max',
        patience=int(getattr(early_config, 'patience', 12)),
        min_delta=float(getattr(early_config, 'min_delta', 0.0)),
    )
    trainer_args = dict(args.trainer)
    trainer_args.pop('accelerator', None)
    trainer_args.pop('devices', None)
    trainer_args['strategy'] = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
    trainer = pl.Trainer(
        **trainer_args, accelerator='gpu', devices=torch.cuda.device_count(),
        deterministic=True, logger=logger, default_root_dir=args.output['runs'],
        callbacks=[checkpoint, early_stop],
    )
    if getattr(args, 'checkpoint', None):
        trainer.test(module, ckpt_path=args.checkpoint)
    else:
        trainer.fit(module)
        trainer.test(module, ckpt_path='best')
