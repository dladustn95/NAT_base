import argparse
import logging
import os
import numpy as np
from tqdm import tqdm
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from nat_base import NATransformer
from mytokenizer import MyTokenizer


parser = argparse.ArgumentParser(description='KoBART Summarization')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/iwslt14/train',
                            help='train file')

        parser.add_argument('--valid_file',
                            type=str,
                            default='data/iwslt14/valid',
                            help='valid file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='')
        return parser

class TranslationDataset(Dataset):
    def __init__(self, filepath, src_tok, tgt_tok, src_lang, tgt_lang, max_seq_len=256) -> None:
        self.filepath = filepath
        self.src_tokenizer = src_tok
        self.tgt_tokenizer = tgt_tok
        self.max_seq_len = max_seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.srcs, self.tgts = self.load_data(self.filepath)

    def __len__(self):
        return len(self.srcs)

    def make_enc_input(self, input_id):
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.src_tokenizer.pad()]
                attention_mask += [0]
        else:
            input_id = input_id[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

        return input_id, attention_mask

    def make_dec_input(self, q_id, a_id):
        length = len(a_id)
        c = len(q_id)/len(a_id)

        dec_input = []
        for i in range(length):
            dec_input.append(q_id[math.floor(i*c)])

        attention_mask = [1] * len(dec_input)
        if len(dec_input) < self.max_seq_len:
            while len(dec_input) < self.max_seq_len:
                dec_input += [self.src_tokenizer.pad()]
                attention_mask += [0]
        else:
            dec_input = dec_input[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

        return dec_input, attention_mask

    def __getitem__(self, index):
        q_tokens = self.srcs[index].strip()
        q_id = self.src_tokenizer.encode("<len> " + q_tokens)
        a_tokens = self.tgts[index].strip()
        a_id = self.tgt_tokenizer.encode(a_tokens)

        encoder_input_id, encoder_attention_mask = self.make_enc_input(q_id)
        decoder_input_id, decoder_attention_mask = self.make_dec_input(q_id[1:], a_id)

        labels = a_id[:self.max_seq_len]
        len_labels = len(labels)

        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                labels += [-100]

        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                'labels': np.array(labels, dtype=np.int_),
                'len_labels': np.array(len_labels, dtype=np.int_)}

    def load_data(self, file_path):
        srcs = []
        tgts = []
        f = open(file_path + "." + self.src_lang, 'r', encoding="UTF-8")
        for line in tqdm(f):
            srcs.append(line.strip())
        f.close()

        f = open(file_path + "." + self.tgt_lang, 'r', encoding="UTF-8")
        for line in tqdm(f):
            tgts.append(line.strip())
        f.close()

        assert len(srcs) == len(tgts), "length different"
        return srcs, tgts

class TranslationModule(pl.LightningDataModule):
    def __init__(self, train_file, valid_file, src_tok, tgt_tok, src_lang, tgt_lang, max_len, batch_size=8, num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.train_file_path = train_file
        self.valid_file_path = valid_file
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len

        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        parser.add_argument('--src_lang',
                            type=str,
                            default="src",
                            help='source language name ex)src, de')
        parser.add_argument('--tgt_lang',
                            type=str,
                            default="tgt",
                            help='target language name ex)tgt, en')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = TranslationDataset(self.train_file_path, self.src_tok, self.tgt_tok, self.src_lang, self.tgt_lang, self.max_len)
        self.valid = TranslationDataset(self.valid_file_path, self.src_tok, self.tgt_tok, self.src_lang, self.tgt_lang, self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.valid,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val


class Base(pl.LightningModule):
    def __init__(self, args, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(args)
        self.args = args

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=32,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=5e-4,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.05,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')

        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--feedforward', type=int, default=2048)
        parser.add_argument('--dropout', type=int, default=0.1)
        parser.add_argument("--max_len", type=int, default=256, help="Maximum length of the output utterances")

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

class Model(Base):
    def __init__(self, args, **kwargs):
        super(Model, self).__init__(args, **kwargs)
        src_tok = MyTokenizer(extra_special_symbols=["<len>"])
        src_tok.read_vocab('data/iwslt14/dict.de')
        self.src_tokenizer = src_tok

        tgt_tok = MyTokenizer()
        tgt_tok.read_vocab('data/iwslt14/dict.en')
        self.tgt_tokenizer = tgt_tok

        self.pad_token_id = src_tok.pad()

        self.model = NATransformer(args, self.src_tokenizer, self.tgt_tokenizer)

    def forward(self, inputs):
        return self.model(inputs['input_ids'],
                          inputs['attention_mask'],
                          inputs['decoder_input_ids'],
                          inputs['decoder_attention_mask'],
                          inputs['labels'],
                          inputs['len_labels'])

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs[-1]
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs[-1]
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)


if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = TranslationModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    model = Model(args)

    dm = TranslationModule(args.train_file,
                           args.valid_file,
                           model.src_tokenizer, model.tgt_tokenizer,
                           args.src_lang, args.tgt_lang,
                           args.max_len,
                           batch_size=args.batch_size,
                           num_workers=args.num_workers)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='version_4/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, gpus=args.gpus, accelerator="dp", logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)