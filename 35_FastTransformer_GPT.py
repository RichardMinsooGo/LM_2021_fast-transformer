'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
# !pip install sentencepiece

data_dir = "/content"

! pip list | grep sentencepiece

import sentencepiece as spm
import csv
import sys
import os
import math
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

from tqdm import tqdm, tqdm_notebook, trange

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from IPython.display import display

# Setup seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# for using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
D3. [PASS] Tokenizer Install & import
'''
# Keras Tokenizer is a tokenizer provided by default in tensorflow 2.X and is a word level tokenizer. It does not require a separate installation.

'''
D4. Define Hyperparameters for Data Engineering
'''
ENCODER_LEN  = 15
DECODER_LEN  = 23
BATCH_SIZE   = 16

'''
D5. Load and modifiy to pandas dataframe
'''
import pandas as pd

pd.set_option('display.max_colwidth', None)

"""
raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),

    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),

    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
    ("I have something very important to tell you.", "Il me faut vous dire quelque chose de très important."),
    ("I have three times as many books as he does.", "J'ai trois fois plus de livres que lui."),
    ("I have to change the batteries in the radio.", "Il faut que je change les piles de cette radio."),
    ("I have to finish up some things before I go.", "Je dois finir deux trois trucs avant d'y aller."),

    ("I have to think about what needs to be done.", "Je dois réfléchir sur ce qu'il faut faire."),
    ("I haven't been back here since the incident.", "Je ne suis pas revenu ici depuis l'accident."),
    ("I haven't eaten anything since this morning.", "Je n'ai rien mangé depuis ce matin."),
    ("I hear his business is on the verge of ruin.", "Apparemment son entreprise est au bord de la faillite."),
    ("I hope I didn't make you feel uncomfortable.", "J'espère que je ne t'ai pas mis mal à l'aise."),
    ("I hope to continue to see more of the world.", "J'espère continuer à voir davantage le monde."),
    ("I hope to see reindeer on my trip to Sweden.", "J'espère voir des rennes lors de mon voyage en Suède."),
    ("I hope you'll find this office satisfactory.", "J'espère que ce bureau vous conviendra."),

    ("I hurried in order to catch the first train.", "Je me dépêchai pour avoir le premier train."),
    ("I just can't stand this hot weather anymore.", "Je ne peux juste plus supporter cette chaleur."),
    ("I just don't want there to be any bloodshed.", "Je ne veux tout simplement pas qu'il y ait une effusion de sang."),
    ("I just thought that you wouldn't want to go.", "J'ai simplement pensé que vous ne voudriez pas y aller."),
    ("I plan to go. I don't care if you do or not.", "Je prévois d'y aller. Ça m'est égal que vous y alliez aussi ou pas."),
    ("I prefer soap as a liquid rather than a bar.", "Je préfère le savon liquide à une savonnette."),
    ("I promise you I'll explain everything later.", "Je vous promets que j'expliquerai tout plus tard."),
    ("I ran as fast as I could to catch the train.", "Je courus aussi vite que je pus pour attraper le train."))


raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"))
"""

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"))

import unicodedata
import re

from tensorflow.keras.preprocessing.text import Tokenizer

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
def preprocess(sent):
    # 위에서 구현한 함수를 내부적으로 호출
    sent = unicode_to_ascii(sent.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sent = re.sub(r"([?.!,¿])", r" \1", sent)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

    sent = re.sub(r"\s+", " ", sent)
    return sent

# 인코딩 테스트
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous deja dine?"

print(preprocess(en_sent))
print(preprocess(fr_sent).encode('utf-8'))

raw_encoder_input, raw_data_fr = list(zip(*raw_data))
raw_encoder_input, raw_data_fr = list(raw_encoder_input), list(raw_data_fr)

raw_src = [preprocess(data) for data in raw_encoder_input]
raw_trg = [preprocess(data) for data in raw_data_fr]

print(raw_src[:4])
print(raw_trg[:4])

'''
D9. Define dataframe
'''
SRC_df = pd.DataFrame(raw_src)
TRG_df = pd.DataFrame(raw_trg)

SRC_df.rename(columns={0: "SRC"}, errors="raise", inplace=True)
TRG_df.rename(columns={0: "TRG"}, errors="raise", inplace=True)
train_df = pd.concat([SRC_df, TRG_df], axis=1)

print('Translation Pair :',len(train_df)) # 리뷰 개수 출력
train_df.sample(3)

raw_src_df  = train_df['SRC']
raw_trg_df  = train_df['TRG']

src_sentence  = raw_src_df
trg_sentence  = raw_trg_df

'''
D10. Define tokenizer
'''

with open('corpus_src.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_df['SRC']))

with open('corpus_trg.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_df['TRG']))

# This is the folder to save the data. Modify it to suit your environment.
data_dir = "/content"

corpus = "corpus_src.txt"
prefix = "nmt_src_vocab"
vocab_size = 200
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

corpus = "corpus_trg.txt"
prefix = "nmt_trg_vocab"

vocab_size = 200
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" +               # max sentence length
    " --pad_id=0 --pad_piece=[PAD]" +               # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +               # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +               # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +               # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")   # other additional tokens

for f in os.listdir("."):
    print(f)

vocab_src_file = f"{data_dir}/nmt_src_vocab.model"
vocab_src = spm.SentencePieceProcessor()
vocab_src.load(vocab_src_file)

vocab_trg_file = f"{data_dir}/nmt_trg_vocab.model"
vocab_trg = spm.SentencePieceProcessor()
vocab_trg.load(vocab_trg_file)

n_enc_vocab = len(vocab_src)
n_dec_vocab = len(vocab_trg)

print('Word set size of Encoder :',n_enc_vocab)
print('Word set size of Decoder :',n_dec_vocab)

'''
Token List
'''
# Recommend : For small number of vocabulary, please test each IDs.
# src_vocab_list
src_vocab_list = [[vocab_src.id_to_piece(id), id] for id in range(vocab_src.get_piece_size())]

# trg_vocab_list
trg_vocab_list = [[vocab_trg.id_to_piece(id), id] for id in range(vocab_trg.get_piece_size())]

'''
D11. Tokenizer test
'''
# Source Tokenizer
lines = [  SRC_df.iloc[1,0],  SRC_df.iloc[2,0],  SRC_df.iloc[3,0]]
for line in lines:
    print("Input        :", line)
    txt_2_ids = vocab_src.encode_as_ids(line)
    print("EncodeIds    :", txt_2_ids)
    print("DecodeIds    :", vocab_src.DecodeIds(txt_2_ids))

    txt_2_tkn = vocab_src.encode_as_pieces(line)
    print("EncodePieces :", txt_2_tkn)
    print("DecodePieces :", vocab_src.DecodePieces(txt_2_tkn))

    ids2 = vocab_src.piece_to_id(txt_2_tkn)
    print("Piece_2_IDs  :", ids2)
    print("Id_2_Pieces  :", vocab_src.id_to_piece(ids2))
    print("\n")

print("\n")

# Target Tokenizer
lines = [  TRG_df.iloc[1,0],  TRG_df.iloc[2,0],  TRG_df.iloc[3,0]]
for line in lines:
    print("Input        :", line)
    txt_2_ids = vocab_trg.encode_as_ids(line)
    print("EncodeIds    :", txt_2_ids)
    print("DecodeIds    :", vocab_trg.DecodeIds(txt_2_ids))
    
    txt_2_tkn = vocab_trg.encode_as_pieces(line)
    print("EncodePieces :", txt_2_tkn)
    print("DecodePieces :", vocab_trg.DecodePieces(txt_2_tkn))

    ids2 = vocab_trg.piece_to_id(txt_2_tkn)
    print("Piece_2_IDs  :", ids2)
    print("Id_2_Pieces  :", vocab_trg.id_to_piece(ids2))
    print("\n")

'''
D12. Tokenize
'''
# tokenize / encode integers / add start and end tokens / padding
tokenized_src  = vocab_src.encode_as_ids(src_sentence.to_list())
tokenized_trg  = vocab_trg.encode_as_ids(trg_sentence.to_list())

# Add [BOS], [EOS] token ids to each target list elements.
new_list = [ x.insert(0, 2) for x in tokenized_trg]
new_list = [ x.insert(len(x), 3) for x in tokenized_trg]

tokenized_inputs  = tokenized_src
tokenized_outputs = tokenized_trg

'''
D13. [EDA] Explore the tokenized datasets
'''

len_result = [len(s) for s in tokenized_inputs]

print('Maximum length of source : {}'.format(np.max(len_result)))
print('Average length of source : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

len_result = [len(s) for s in tokenized_outputs]

print('Maximum length of target : {}'.format(np.max(len_result)))
print('Average length of target : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

'''
D14. Pad sequences
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
tkn_sources = pad_sequences(tokenized_inputs,  maxlen=ENCODER_LEN, padding='post', truncating='post')
tkn_targets = pad_sequences(tokenized_outputs, maxlen=DECODER_LEN, padding='post', truncating='post')

'''
D15. Send data to device
'''

tensors_src   = torch.tensor(tkn_sources).to(device)
tensors_trg   = torch.tensor(tkn_targets).to(device)

'''
D16. [EDA] Explore the Tokenized datasets
'''
print('Size of source language data(shape) :', tkn_sources.shape)
print('Size of target language data(shape) :', tkn_targets.shape)

# Randomly output the 0th sample
print(tkn_sources[0])
print(tkn_targets[0])

'''
D17. [PASS] Split Data
'''

'''
D18. Build dataset
'''

from torch.utils.data import TensorDataset   # 텐서데이터셋
from torch.utils.data import DataLoader      # 데이터로더

dataset    = TensorDataset(tensors_src, tensors_trg)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


'''
D19. [PASS] Define some useful parameters for further use
'''

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''
from tqdm import tqdm, tqdm_notebook, trange


from __future__ import annotations
from math import pi, log

import torch
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat

from typing import Literal

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as tortoise-tts was using it

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    dtype = t.dtype

    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place    
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len = 0

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_scales_seq_len = 0

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype = dtype, device = device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        freqs = self.forward(seq, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            exists(seq_len) and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales_seq_len
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len = seq_len

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            freqs = self.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs

# ---------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce

# from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# blocks

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

class FastAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        max_seq_len = None,
        pos_emb = None
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # rotary positional embedding

        assert not (exists(pos_emb) and not exists(max_seq_len)), 'max_seq_len must be passed in if to use rotary positional embeddings'

        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len

        # if using relative positional encoding, make sure to reduce pairs of consecutive feature dimension before doing projection to attention logits

        kv_attn_proj_divisor = 1 if not exists(pos_emb) else 2

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting queries to query attention logits
        self.to_k_attn_logits = nn.Linear(dim_head // kv_attn_proj_divisor, 1, bias = False)  # for projecting keys to key attention logits

        # final transformation of values to "r" as in the paper

        self.to_r = nn.Linear(dim_head // kv_attn_proj_divisor, dim_head)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        n, device, h, use_rotary_emb = x.shape[1], x.device, self.heads, exists(self.pos_emb)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        # if relative positional encoding is needed

        if use_rotary_emb:
            freqs = self.pos_emb(torch.arange(self.max_seq_len, device = device), cache_key = self.max_seq_len)
            freqs = rearrange(freqs[:n], 'n d -> () () n d')
            q_aggr, k_aggr, v_aggr = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        else:
            q_aggr, k_aggr, v_aggr = q, k, v

        # calculate query attention logits

        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim = -1)

        # calculate global query token

        global_q = einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        k = k * global_q

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            k = reduce(k, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # now calculate key attention logits

        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim = -1)

        # calculate global key token

        global_k = einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # bias the values

        u = v_aggr * global_k

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # transformation step

        r = self.to_r(u)

        # paper then says to add the queries as a residual

        r = r + q

        # combine heads

        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)

# main class

class FastTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        absolute_pos_emb = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        # positional embeddings

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if absolute_pos_emb else None

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, 'dimension of the head must be divisible by 4 to use rotary embeddings'
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        # layers

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = FastAttention(dim, dim_head = dim_head, heads = heads, pos_emb = layer_pos_emb, max_seq_len = max_seq_len)
            ff = FeedForward(dim, mult = ff_mult)

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        # weight tie projections across all layers

        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> () n d')

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.to_logits(x)

# ---------------------------------------------------------------------------------------------

import torch
# from fast_transformer_pytorch import FastTransformer

model = FastTransformer(
    num_tokens = n_dec_vocab,
    dim = 512,
    depth = 2,
    max_seq_len = 4096,
    absolute_pos_emb = True   # default uses relative positional encoding, but if that isn't working, then turn on absolute positional embedding by setting this to True
)

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# 네트워크 초기화
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner층의 초기화
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# TransformerBlock모듈의 초기화 설정
model.apply(initialize_weights)

import os.path

if os.path.isfile('./checkpoints/GPT_model_Sentencepiece.pt'):
    model.load_state_dict(torch.load('./checkpoints/GPT_model_Sentencepiece.pt'))

print('네트워크 초기화 완료')

# 손실 함수의 정의
criterion = nn.CrossEntropyLoss()

# 최적화 설정
# learning_rate = 2e-4
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

from IPython.display import clear_output
import datetime

Model_start_time = time.time()

# 학습 정의
def train(epoch, model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    accuracies = []

    with tqdm_notebook(total=len(dataloader), desc=f"Train {epoch+1}") as pbar:
        for batch_idx, samples in enumerate(dataloader):
            src_inputs, trg_outputs = samples

            # print("src_inputs  Shape :", src_inputs.shape)
            # print(src_inputs)
            mask_src = (src_inputs!=0).int()
            # print(mask_src)

            # print("trg_outputs Shape :", trg_outputs.shape)
            # print("trg_outputs :\n", trg_outputs)
            mask_trg = (trg_outputs!=0).int()
            # print(mask_trg)

            Input_concat = torch.concat((src_inputs, trg_outputs),dim=1)
            # print("Input_concat Shape :", Input_concat.shape)
            # print("Input_concat :\n", Input_concat)

            with torch.set_grad_enabled(True):
                
                mask_bool = Input_concat.bool()

                # Transformer에 입력
                logits_lm = model(Input_concat,mask_bool)
                # print("logits_lm  Shape :", logits_lm.shape)
                
                pad       = torch.LongTensor(trg_outputs.size(0), 1).fill_(0).to(device)
                preds_id  = torch.transpose(logits_lm,1,2)
                labels_lm = torch.cat((trg_outputs[:, 1:], pad), -1)
                # print("labels_lm Shape: \n",labels_lm.shape)
                # print("labels_lm : \n",labels_lm)

                labels_concat = torch.concat((src_inputs, labels_lm),dim=1)
                # print("labels_concat Shape :", labels_concat.shape)
                # print("labels_concat :\n", labels_concat)

                optimizer.zero_grad()
                loss = criterion(preds_id, labels_concat)  # loss 계산


                # Accuracy
                # print("preds_id  : \n",preds_id.shape)
                mask_0 = (labels_concat!=0).int()
                arg_preds_id = torch.argmax(preds_id, axis=1)
                # print("arg_preds : \n",arg_preds_id)
                # print("arg_preds : \n",arg_preds_id.shape)
                # print("mask_0    : \n",mask_0)

                accuracy_1 = torch.eq(labels_concat, arg_preds_id).int()
                # print("accuracy_1 : \n",accuracy_1)

                accuracy_2 = torch.mul(arg_preds_id, accuracy_1).int()
                # print("accuracy_2 : \n",accuracy_2)

                accuracy = torch.count_nonzero(accuracy_2) / torch.count_nonzero(mask_0)
                # print("Accuracy : ",accuracy.clone().detach().cpu().numpy())
                accuracies.append(accuracy.clone().detach().cpu().numpy())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss +=loss.item()

            pbar.update(1)
            # pbar.set_postfix_str(f"Loss {epoch_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
            # pbar.set_postfix_str(f"Loss {loss.result():.4f}")
    print("accuracies :", np.mean(accuracies))
    return epoch_loss / len(dataloader)

CLIP = 0.5

epoch_ = []
epoch_train_loss = []
# 네트워크가 어느정도 고정되면 고속화
torch.backends.cudnn.benchmark = True
# epoch 루프
best_epoch_loss = float("inf")

N_EPOCHS = 100

for epoch in range(N_EPOCHS):

    train_loss = train(epoch, model, dataloader, optimizer, criterion, CLIP)

    if train_loss < best_epoch_loss:
        if not os.path.isdir("checkpoints"):
            os.makedirs("checkpoints")
        best_epoch_loss = train_loss
        torch.save(model.state_dict(), './checkpoints/GPT_model_Sentencepiece.pt')

    epoch_.append(epoch)
    epoch_train_loss.append(train_loss)
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

    # print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, N_EPOCHS, epoch_loss))
    # clear_output(wait = True)

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.plot(epoch_,epoch_train_loss, label='Average loss')

ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

plt.show()

# Build evaluation code.

# Predict the trained model
trained_model = FastTransformer(
    num_tokens = n_dec_vocab,
    dim = 512,
    depth = 2,
    max_seq_len = 4096,
    absolute_pos_emb = True   # default uses relative positional encoding, but if that isn't working, then turn on absolute positional embedding by setting this to True
)

trained_model.to(device)

trained_model.load_state_dict(torch.load('./checkpoints/GPT_model_Sentencepiece.pt'))


def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def evaluate(text):
    text = preprocess_sentence(text)
    # print(text)
    text = [vocab_src.encode_as_ids(text)]
    # print(text)
    encoder_input = pad_sequences(text, maxlen=ENCODER_LEN, padding='post', truncating='post')
    # print(encoder_input)

    decoder_input = [2]   #[BOS] token is 2
    # print(decoder_input)
    
    input  = torch.tensor(encoder_input).to(device)
    output = torch.tensor([decoder_input]).to(device)

    # print("input :", input)
    # print("output:", output)

    for i in range(DECODER_LEN):
        concate_input = torch.concat((input, output),dim=1)
        
        mask_bool = concate_input.bool()

        # Transformer에 입력
        predictions = model(concate_input,mask_bool)
                
        # print("concate_input :", concate_input)
        # print(predictions)

        predictions = predictions[:, -1:, :]
        # print(predictions)

        # PAD, UNK, START 토큰 제외
        predicted_id = torch.argmax(predictions, axis=-1)
        # print(predicted_id)
        if predicted_id== 3:
            break

        output = torch.cat((output, predicted_id),-1)
    return output

def predict(text):
    prediction = evaluate(text)[0].detach().cpu().numpy()
    prediction = prediction[1:]
    # print("Pred IDs :", prediction)

    predicted_sentence = vocab_trg.DecodeIds(prediction.tolist())
    # print(predicted_sentence)
    return predicted_sentence

for idx in (0, 1, 2, 3):
    print("Input        :", raw_src[idx])
    print("Prediction   :", predict(raw_src[idx]))
    print("Ground Truth :", raw_trg[idx],"\n")



'''
M13. [PASS] Explore the training result with test dataset
'''
    
