# %%
# Original code from minGPT by Andrej Karpathy
# https://github.com/karpathy/minGPT/
# Modifications by @jaspock

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

import itertools

use_simple_attention = True

def make_batch(input_sentences, output_tags, word_index, tag_index, max_len, batch_size, device):
    input_batch = []
    output_batch = []
    data_cycle = itertools.cycle(zip(input_sentences, output_tags))

    # to-do: adjust T to be minimum of the actual max length of the batch or max_len

    while True:
        for s,t in data_cycle:
            words = s.split()
            tags = t.split()
            assert len(words) == len(tags)
            inputs = [word_index[n] for n in words]
            inputs = inputs + [0] * (max_len - len(inputs))  # padded inputs
            tags = [tag_index[n] for n in tags]
            tags = tags + [0] * (max_len - len(tags))  # padded outputs
            input_batch.append(inputs)
            output_batch.append(tags)

            if len(input_batch) == batch_size:
                yield torch.LongTensor(input_batch, device=device), torch.LongTensor(output_batch, device=device)
                input_batch = []
                output_batch = []

class LayerNorm(nn.Module):
    # to-do: check that this is equivalent to the PyTorch implementation and replace nn.LayerNorm with it
    # to-do: check the init_weights function

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class HeadAttention(nn.Module):
    def __init__(self, n_embd, n_embd_head, attn_pdrop=0.1):
        super().__init__()
        self.q_lin = nn.Linear(n_embd, n_embd_head)
        self.k_lin = nn.Linear(n_embd, n_embd_head)
        self.v_lin = nn.Linear(n_embd, n_embd_head)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)

    def forward(self, x, padding_mask): 
        q = self.q_lin(x) # (B, T, n_embd) -> (B, T, n_embd_head)
        k = self.k_lin(x)
        v = self.v_lin(x)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # expand mask so that broadcasting between (B, T, T) and (B, 1, T) applies
        expanded_mask = padding_mask.unsqueeze(1) # (T, T) -> (B, 1, T)
        att = att.masked_fill(expanded_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v # (B, T, T) @ (B, T, n_embd_head) -> (B, T, n_embd_head)


class MultiHeadAttentionSimple(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads as a list
        self.heads = nn.ModuleList([HeadAttention(n_embd, n_embd // n_head, attn_pdrop) for _ in range(n_head)])
        self.c_proj = nn.Linear(n_embd, n_embd)  # output projection to integrate head outputs
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x, padding_mask):
        y = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1)  # (B, T, n_embd)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MultiHeadAttentionOriginal(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

        # to-do: check that this class behaves similar to the simpler one

    def forward(self, x, padding_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        mask = padding_mask.view(1,1,T,T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask[:,:,:T,:T] == 1, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        if not use_simple_attention: # original code
            self.attn = MultiHeadAttentionOriginal(n_embd, n_head, attn_pdrop, resid_pdrop)
        else:
            self.attn = MultiHeadAttentionSimple(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),  # ffw hidden layer size is fixed to 4*n_embd
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        
    def forward(self, x, padding_mask):
        x = x + self.attn(self.ln_1(x),padding_mask)
        m = self.mlp  # just a shorter name
        x = x +  m.dropout(m.c_proj(m.act(m.c_fc(self.ln_2(x)))))
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, input_vocab_size, output_vocab_size, max_len, 
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(input_vocab_size, n_embd),
            wpe = nn.Embedding(max_len, n_embd),
            drop = nn.Dropout(embd_pdrop),
            h = nn.ModuleList([Block(n_embd, n_head, attn_pdrop, resid_pdrop) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, output_vocab_size, bias=False)
        self._init_weights()
        
        # report number of parameters (note we don't count the parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def forward(self, inputs):
        padding_mask = inputs == 0
        device = inputs.device
        b, t = inputs.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        tok_emb = self.transformer.wte(inputs) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, padding_mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

if __name__ == '__main__':

    n_layer = 2
    n_head = 2
    n_embd =  64
    # dropout hyperparameters
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    batch_size = 3
    max_len = 12
    
    input_sentences = [
        "The cat sat on the mat .",
        "I love eating pizza .",
        "John is running in the park .",
        "She gave him a beautiful gift .",
        "They are playing soccer together .",
        "The cat is eating pizza in the park ."
    ]

    output_tags = [
        "DET NOUN VERB ADP DET NOUN PUNCT",
        "PRON VERB VERB NOUN PUNCT",
        "PROPN AUX VERB ADP DET NOUN PUNCT",
        "PRON VERB PRON DET ADJ NOUN PUNCT",
        "PRON AUX VERB NOUN ADV PUNCT",
        "DET NOUN AUX VERB NOUN ADP DET NOUN PUNCT"
    ]

    word_list = list(set(" ".join(input_sentences).split()))
    # CLS, SEP and MASK tags added just in case we do BERT-like language modelling
    word_index = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3} 
    for i, w in enumerate(word_list):
        word_index[w] = i + 4
    index_word = {i: w for i, w in enumerate(word_index)}
    input_vocab_size = len(word_index)
    tag_list = list(set(" ".join(output_tags).split()))
    tag_index = {'[PAD]': 0}  # padding index must be 0
    for i, t in enumerate(tag_list):
        tag_index[t] = i + 1
    index_tag = {i:t for i, t in enumerate(tag_index)}
    output_vocab_size = len(tag_index)
    # print vocab sizes
    print("input vocab size: %d" % input_vocab_size)
    print("output vocab size: %d" % output_vocab_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    model = EncoderTransformer(n_embd=n_embd, n_head=n_head, n_layer=n_layer, input_vocab_size=input_vocab_size, output_vocab_size=output_vocab_size, 
                 max_len=max_len, embd_pdrop=embd_pdrop, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
    if device == 'cuda':
        model = model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # to-do: add learning rate scheduler

    model.train()
    step = 0
    for inputs, outputs in make_batch(input_sentences=input_sentences, output_tags=output_tags, word_index=word_index, 
                                      tag_index=tag_index, max_len=max_len, batch_size=batch_size, device=device):
        padding_mask = inputs == 0
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss_lm = criterion(logits.view(-1,logits.size(-1)), outputs.view(-1)) 
        loss_lm = loss_lm.mean()
        if (step + 1) % 100 == 0:
            print('Step:', '%04d' % (step + 1), 'cost =', '{:.6f}'.format(loss_lm))
        loss_lm.backward()
        optimizer.step()
        step += 1
        if (step==1000):
            break

    # predict tags
    model.eval()
    inputs, outputs = make_batch(input_sentences=input_sentences, output_tags=output_tags, word_index=word_index, tag_index=tag_index, max_len=max_len, batch_size=batch_size, device=device).__next__()
    print(inputs,outputs)
    logits = model(inputs)
    _, indices = torch.max(logits, dim=-1)
    predict_tags, true_tags, input_words = [], [], []  # 3 lists are required, not one
    for i in range(batch_size):
        predict_tags.append(" ".join([index_tag[each.item()] for each in indices[i]]))
        true_tags.append(" ".join([index_tag[each.item()] for each in outputs[i]]))
        input_words.append(" ".join([index_word[each.item()] for each in inputs[i]]))
    print("Input:\n", "\n".join(input_words))
    print("Prediction: \n", "\n".join(predict_tags))
    print("Target: \n", "\n".join(true_tags))


# %%
