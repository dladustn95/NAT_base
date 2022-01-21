import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import math
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = int(args.d_model / args.n_heads)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        scores.masked_fill_(attn_mask, -1e9)
        last_attention_weight = scores

        attn = self.dropout(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)

        return context, attn, last_attention_weight


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.args = args
        self.d_k = int(args.d_model / args.n_heads)
        self.d_v = int(args.d_model / args.n_heads)
        self.n_heads = args.n_heads
        self.W_Q = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_K = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_V = nn.Linear(args.d_model, self.d_v * args.n_heads)
        self.li1 = nn.Linear(args.n_heads * self.d_v, args.d_model)
        self.layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)

        context, attn, last_attention_weight = ScaledDotProductAttention(self.args)(q_s, k_s, v_s, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.li1(context)

        return output + residual, attn, last_attention_weight


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.feedforward, out_channels=args.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        residual = inputs
        output = self.dropout(self.relu(self.conv1(inputs.transpose(1, 2))))
        output = self.conv2(output).transpose(1, 2)

        return output + residual


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.args = args
        self.dropout = args.dropout
        self.enc_self_attn = Attention(self.args.d_model, self.args.n_heads, self.args.dropout)
        self.pos_ffn = PoswiseFeedForwardNet(self.args)
        self.self_attn_layer_norm = nn.LayerNorm(args.d_model)
        self.final_layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, enc_inputs, enc_self_attn_mask=None):
        residual = enc_inputs
        hidden_states, attn, _ = self.enc_self_attn(
            hidden_states=enc_inputs,
            attention_mask=enc_self_attn_mask,
            layer_head_mask=None,
            output_attentions=True,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        enc_outputs = self.pos_ffn(hidden_states)
        enc_outputs = self.final_layer_norm(enc_outputs)

        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()

        self.args = args
        self.dec_self_attn = Attention(self.args.d_model, self.args.n_heads, self.args.dropout, is_decoder=True,)
        self.dec_enc_attn = Attention(self.args.d_model, self.args.n_heads, self.args.dropout, is_decoder=True,)
        self.pos_ffn = PoswiseFeedForwardNet(args)
        self.dropout = args.dropout

        self.self_attn_layer_norm = nn.LayerNorm(args.d_model)
        self.encoder_attn_layer_norm = nn.LayerNorm(args.d_model)
        self.final_layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, dec_inputs, encoder_hidden_states, dec_self_attn_mask, encoder_attention_mask):
        residual = dec_inputs
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.dec_self_attn(
            hidden_states=dec_inputs,
            attention_mask=dec_self_attn_mask,
            output_attentions=True,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            #cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.dec_enc_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=True,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        dec_outputs = self.pos_ffn(hidden_states)
        dec_outputs = self.final_layer_norm(dec_outputs)

        att_weights = (self_attn_weights, cross_attn_weights)

        return dec_outputs, att_weights


class Decoder(nn.Module):
    def __init__(self, args, vocab_size, pad_ids, embed_tokens: Optional[nn.Embedding] = None):
        super(Decoder, self).__init__()
        self.args = args
        self.pad_ids = pad_ids
        self.layers = nn.ModuleList([DecoderLayer(self.args) for _ in range(args.n_layers)])

        if embed_tokens is not None:
            self.src_emb = embed_tokens
        else:
            self.src_emb = nn.Embedding(vocab_size, args.d_model, self.pad_ids)

        self.pos_embedding = PositionalEncoding(args.d_model, args.max_len)
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(self.args.d_model)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        return expanded_attn_mask

    def forward(self, dec_inputs, attention_mask, encoder_hidden_states, encoder_attention_mask):
        dec_outputs = self.src_emb(dec_inputs) + self.pos_embedding(dec_inputs)

        attention_mask = _expand_mask(attention_mask, dec_outputs.dtype)

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, dec_outputs.dtype, tgt_len=dec_inputs.size()[-1])

        for layer in self.layers:
            dec_outputs, last_attention_weight = layer(
                dec_outputs,
                encoder_hidden_states,
                attention_mask,
                encoder_attention_mask
            )

        return dec_outputs, last_attention_weight


class Encoder(nn.Module):
    def __init__(self, args, vocab_size, pad_ids, embed_tokens: Optional[nn.Embedding] = None):
        super(Encoder, self).__init__()

        self.args = args
        self.pad_ids = pad_ids
        self.d_model = args.d_model

        if embed_tokens is not None:
            self.src_emb = embed_tokens
        else:
            self.src_emb = nn.Embedding(vocab_size, self.d_model, self.pad_ids)
        self.pos_embedding = PositionalEncoding(self.d_model, args.max_len)
        self.layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(self.args.n_layers)])

        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.length_classifier = nn.Linear(self.d_model, args.max_len+1)

    def forward(self, enc_inputs, attention_mask=None):
        enc_outputs = self.src_emb(enc_inputs) + self.pos_embedding(enc_inputs)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, enc_outputs.dtype)

        for layer in self.layers:
            enc_outputs, _ = layer(enc_outputs, attention_mask)
        #[bsz, seq_len, hidden]
        length = self.length_classifier(enc_outputs[:, 0, :])

        return enc_outputs[:,1:,:], length


class NATransformer(nn.Module):
    def __init__(self, args, src_tokenizer, tgt_tokenizer):
        super(NATransformer, self).__init__()
        self.src_pad_ids = src_tokenizer.pad()
        self.tgt_pad_ids = tgt_tokenizer.pad()
        self.src_vocab_size = src_tokenizer.vocab_size()
        self.tgt_vocab_size = tgt_tokenizer.vocab_size()

        #self.shared = nn.Embedding(self.vocab_size, args.d_model, self.pad_ids)

        self.encoder = Encoder(args, self.src_vocab_size, self.src_pad_ids)
        self.decoder = Decoder(args, self.src_vocab_size, self.src_pad_ids)

        self.projection = nn.Linear(args.d_model, self.tgt_vocab_size)

        self.max_len = args.max_len
        self.args = args

    def forward(self, enc_inputs, enc_attention_mask, dec_input, dec_attention_mask=None, labels=None, len_labels=None):
        enc_outputs, length = self.encoder(enc_inputs, enc_attention_mask)
        dec_outputs, _ = self.decoder(dec_input, dec_attention_mask, enc_outputs, enc_attention_mask[:,1:])
        lm_logits = self.projection(dec_outputs)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(lm_logits.view(-1, self.tgt_vocab_size), labels.view(-1))
            loss = lm_loss

        if len_labels is not None:
            loss_fct = CrossEntropyLoss()
            len_loss = loss_fct(length.view(-1, self.args.max_len+1), len_labels.view(-1))
            if loss is not None:
                loss = loss + len_loss
            else:
                loss = len_loss

        return lm_logits, length, loss

    """def forward(self, enc_inputs, enc_attention_mask, dec_input, dec_attention_mask=None, labels=None):
        enc_outputs = self.encoder(enc_inputs, enc_attention_mask)
        dec_outputs, _ = self.decoder(dec_input, dec_attention_mask, enc_outputs, enc_attention_mask)
        lm_logits = self.projection(dec_outputs)"""