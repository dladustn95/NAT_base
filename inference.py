import argparse
from tqdm import tqdm
import yaml
import math

import torch
import torch.nn.functional as F

from mytokenizer import MyTokenizer
from train_nat import Model

def duplicate_encoder_out(encoder_out, att_mask, bsz, beam_size):
    new_encoder_out = encoder_out.unsqueeze(2).repeat(beam_size, 1, 1, 1).view(bsz * beam_size, encoder_out.size(1), -1)
    new_att_mask = att_mask.unsqueeze(1).repeat(beam_size, 1, 1).view(bsz * beam_size, -1)
    return new_encoder_out, new_att_mask

def predict_length_beam(gold_target_len, predicted_lengths, length_beam_size):
    if gold_target_len is not None:
        beam_starts = gold_target_len - (length_beam_size - 1) // 2
        beam_ends = gold_target_len + length_beam_size // 2 + 1
        beam = list(range(beam_starts, beam_ends))
        beam = [x if x > 1 else 1 for x in beam]
    else:
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1]
        beam[beam < 2] = 2
        beam = beam[0].tolist()
    return beam

def make_dec_input(q_id, a_len, max_len, pad_id):
    c = len(q_id)/a_len

    dec_input = []
    for i in range(a_len):
        dec_input.append(q_id[math.floor(i*c)])

    attention_mask = [1] * len(dec_input)
    if len(dec_input) < max_len:
        while len(dec_input) < max_len:
            dec_input += [pad_id]
            attention_mask += [0]
    else:
        dec_input = dec_input[:max_len]
        attention_mask = attention_mask[:max_len]

    return dec_input, attention_mask

def generate_step_with_prob(out):
    probs = F.softmax(out, dim=-1) # 전체 확률
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=None, type=str)
    parser.add_argument("--model_binary", default=None, type=str)
    parser.add_argument("--testfile", default=None, type=str)
    parser.add_argument("--outputfile", default=None, type=str)
    parser.add_argument("--gold_len", default=False, type=bool)
    parser.add_argument("--length_beam_size", default=3, type=int)
    args = parser.parse_args()

    with open(args.hparams) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
        hparams.update(vars(args))

    args = argparse.Namespace(**hparams)

    inf = Model.load_from_checkpoint(args.model_binary, args=args)
    model = inf.model
    model.to('cuda')
    model.eval()

    src_tokenizer = inf.src_tokenizer
    tgt_tokenizer = inf.tgt_tokenizer
    """
    src_tokenizer = MyTokenizer(extra_special_symbols=["<len>"])
    src_tokenizer.read_vocab('data/iwslt14/dict.de')

    tgt_tokenizer = MyTokenizer()
    tgt_tokenizer.read_vocab('data/iwslt14/dict.en')
    """
    src_lines = []
    f = open(args.testfile + '.de', 'r', encoding="utf-8-sig")
    for line in f:
        src_lines.append(line)
    f.close()

    tgt_lines = []
    f = open(args.testfile + '.en', 'r', encoding="utf-8-sig")
    for line in f:
        tgt_lines.append(line)
    f.close()

    f = open(args.outputfile, 'w', encoding="utf-8-sig")

    for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
        encoder_input_id = src_tokenizer.encode("<len> " + src_line)
        if len(encoder_input_id) > args.max_len:
            encoder_input_id = encoder_input_id[:args.max_len]
        enc_attention_mask = [1] * len(encoder_input_id)

        enc_attention_mask = torch.tensor(enc_attention_mask)
        enc_attention_mask = enc_attention_mask.unsqueeze(0)
        enc_attention_mask = enc_attention_mask.cuda()

        encoder_input = torch.tensor(encoder_input_id)
        encoder_input = encoder_input.unsqueeze(0)
        encoder_input = encoder_input.cuda()

        enc_outputs, length = model.encoder(encoder_input, enc_attention_mask)
        if args.gold_len:
            gold_target_len = len(tgt_line.split())
        else:
            gold_target_len = None

        length[:, 0] += float('-inf')  # Cannot predict the len_token
        length = F.log_softmax(length, dim=-1)
        length_cands = predict_length_beam(gold_target_len, length, args.length_beam_size)
        max_len = max(length_cands)

        dec_inputs = []
        dec_att_masks = []
        for len_can in length_cands:
            dec_input, dec_att_mask = make_dec_input(encoder_input_id[1:], len_can, max_len, src_tokenizer.pad())
            dec_inputs.append(dec_input)
            dec_att_masks.append(dec_att_mask)

        dec_attention_mask = torch.tensor(dec_att_masks)
        dec_attention_mask = dec_attention_mask.cuda()
        decoder_input_id = torch.tensor(dec_inputs)
        decoder_input_id = decoder_input_id.cuda()

        enc_outputs, enc_attention_mask = duplicate_encoder_out(enc_outputs, enc_attention_mask, enc_outputs.size(0), args.length_beam_size)

        dec_outputs, _ = model.decoder(decoder_input_id, dec_attention_mask, enc_outputs, enc_attention_mask[:, 1:])
        lm_logits = model.projection(dec_outputs)

        tgt_tokens, token_probs, _ = generate_step_with_prob(lm_logits)
        for i in range(args.length_beam_size):
            tgt_tokens[i][length_cands[i]:] = tgt_tokenizer.pad()
            token_probs[i][length_cands[i]:] = 1

        lprobs = token_probs.log().sum(-1)
        avg_log_prob = lprobs / torch.tensor(length_cands).cuda()
        best_lengths = avg_log_prob.max(-1)[1].item()

        result_line = tgt_tokenizer.decode(tgt_tokens[best_lengths].tolist())

        f.write(result_line)
        f.write('\n')

    f.close()
