# %%
# Original code from minGPT by Andrej Karpathy
# https://github.com/karpathy/minGPT/
# Modifications by @jaspock

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import math
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(430)

def make_language_modeling_batch(corpus, word_index, max_len, batch_size, device):

    tokens = corpus.split()
    token_indices = [word_index.get(token, word_index['[UNK]']) for token in tokens]
    n_tokens = len(token_indices)  # number of tokens in the corpus
    batch_token_length = batch_size * max_len  # the total number of tokens in a batch
    assert n_tokens >= batch_token_length, f'Short corpus ({n_tokens} tokens), must be at least {batch_length} tokens long'

    while True:
        input_batch, output_batch = [], []
        
        for _ in range(batch_size):
            start_index = random.randint(0, n_tokens - 1)  # random start
            end_index = start_index + max_len
            input_seq = token_indices[start_index:end_index]
            if end_index > n_tokens:
                input_seq += token_indices[:end_index - n_tokens]
            
            # output is the same as input, except shifted one token to the right
            output_seq = input_seq[1:] + [token_indices[end_index % n_tokens]]

            input_batch.append(input_seq)
            output_batch.append(output_seq)

        yield torch.LongTensor(input_batch).to(device), torch.LongTensor(output_batch).to(device)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class HeadAttention(nn.Module):
    def __init__(self, n_embd, n_embd_head, attn_pdrop=0.1):
        super().__init__()
        self.q_lin = nn.Linear(n_embd, n_embd_head)
        self.k_lin = nn.Linear(n_embd, n_embd_head)
        self.v_lin = nn.Linear(n_embd, n_embd_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)

    def forward(self, x, mask): 
        B, T, C = x.size()  # batch size, sequence length, main embedding dim, C' = head embedding dim
        q = self.q_lin(x)  # (B, T, C) -> (B, T, C')
        k = self.k_lin(x)
        v = self.v_lin(x)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = mask.view(1,T,T) # expand mask, (T, T) -> (1, T, T)
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v  # (B, T, T) @ (B, T, C') -> (B, T, C')


class MultiHeadAttentionSimple(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.heads = nn.ModuleList([HeadAttention(n_embd, n_embd // n_head, attn_pdrop) for _ in range(n_head)])
        self.c_proj = nn.Linear(n_embd, n_embd)  # output projection to integrate head outputs
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x, padding_mask):
        y = torch.cat([h(x, padding_mask) for h in self.heads], dim=-1)  # [(B,T,C')] -> (B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MultiHeadAttentionEfficient(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, main embedding dim, C' = head embedding dim
        H = self.n_head
        Cp = C // H
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C')
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C')
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C')

        # mask = mask.view(1,1,T,T)

        # self-attention: (B, H, T, C') x (B, H, C', T) -> (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att.masked_fill_(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, H, T, T) x (B, H, T, C') -> (B, H, T, C')
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


MultiHeadAttention = MultiHeadAttentionEfficient


class Block(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),  # ffw hidden layer size is fixed to 4*n_embd
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        
    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x),mask)
        x = x +  self.mlp.dropout(self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(self.ln_2(x)))))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, vocab_size, max_len, 
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(max_len, n_embd),
            drop = nn.Dropout(embd_pdrop),
            h = nn.ModuleList([Block(n_embd, n_head, attn_pdrop, resid_pdrop) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self._init_weights()
        # report number of parameters (note we don't count the parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"number of parameters: {(n_params/1e6):.2f}M")
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # elif isinstance(module, nn.LayerNorm):
            #    torch.nn.init.zeros_(module.bias)
            #    torch.nn.init.ones_(module.weight)

    def forward(self, inputs):
        B, T = inputs.size()
        device = inputs.device
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()  # causal attention mask            
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        tok_emb = self.transformer.wte(inputs)  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)  # (1, T, C)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

n_layer = 2
n_head = 2
n_embd =  64
embd_pdrop = 0.1
resid_pdrop = 0.1
attn_pdrop = 0.1
batch_size = 4
max_len = 32
train_steps = 1000
eval_steps = 100
lr = 0.001

corpus = """
En un lugar de La Mancha de cuyo nombre no quiero acordarme no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero adarga antigua rocín 
flaco y galgo corredor Una olla de algo más vaca que carnero salpicón las más noches duelos y quebrantos los sábados lentejas los viernes algún palomino 
de añadidura los domingos consumían las tres partes de su hacienda El resto della concluían sayo de velarte calzas de velludo para las fiestas con sus 
pantuflos de lo mismo y los días de entresemana se honraba con su vellorí de lo más fino Tenía en su casa una ama que pasaba de los cuarenta y una 
sobrina que no llegaba a los veinte y un mozo de campo y plaza que así ensillaba el rocín como tomaba la podadera Frisaba la edad de nuestro hidalgo con 
los cincuenta años Era de complexión recia seco de carnes enjuto de rostro gran madrugador y amigo de la caza Quieren decir que tenía el sobrenombre de 
Quijada o Quesada que en esto hay alguna diferencia en los autores aunque por conjeturas verisímiles se deja entender que se llamaba Quijana Pero esto 
importa poco a nuestro cuento basta que en la narración dél no se salga un punto de la verdad del cual no pudiera poner en duda el más acreditado historiador
"""

word_list = list(set(corpus.split()))
word_index = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2}
special_tokens= len(word_index) 
for i, w in enumerate(word_list):
    word_index[w] = i + special_tokens
index_word = {i: w for i, w in enumerate(word_index)}
vocab_size = len(word_index)
print("vocab size: %d" % vocab_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DecoderTransformer(n_embd=n_embd, n_head=n_head, n_layer=n_layer, vocab_size=vocab_size,  
                max_len=max_len, embd_pdrop=embd_pdrop, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # not needed here since we are not padding inputs
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=train_steps, epochs=1, anneal_strategy='cos')

model.train()
step = 0

for inputs, outputs in make_language_modeling_batch(corpus, word_index, max_len, batch_size, device):
    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits.view(-1,logits.size(-1)), outputs.view(-1)) 
    if (step + 1) % eval_steps == 0 or step == 0:
        print(f"Step {(step + 1):5d}, loss {loss.item():.6f}")
        print(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()
    step = step + 1
    if (step==train_steps):
        break

def generate_text(model, prompt, word_index, index_word, max_len, device):
    words = prompt.split()
    input_ids = [word_index.get(word, word_index['[UNK]']) for word in words]
    input = torch.LongTensor(input_ids).view(1, -1).to(device)  # add batch dimension

    with torch.no_grad():
        for _ in range(max_len - len(input_ids)):
            output = model(input)
            last_token_logits = output[0, -1, :]
            predicted_id = torch.argmax(last_token_logits, dim=-1).item()
            input = torch.cat([input, torch.LongTensor([predicted_id]).view(1,-1).to(device)], dim=1)
            predicted_word = index_word[predicted_id]
            words.append(predicted_word)
            if predicted_word == '[EOS]':
                break

    return ' '.join(words)

model.eval()
prompt = "Los días de entresemana en su casa"
generated_text = generate_text(model, prompt, word_index, index_word, max_len, device)
print(generated_text)
