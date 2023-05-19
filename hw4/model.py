# ========================================================
#             Media and Cognition
#             Homework 4  Sequence Modeling
#             model.py - Model definition
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2023
# ========================================================


# Import required libraries
############################################################
import math
import torch
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np

############################################################

# Define the GELU activation function used in OpenAI GPT
############################################################
def gelu(z):
    """
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    0.5z(1 + tanh[(2/π)^(1/2) * (z + 0.044715 z^3)])
    """
    return 0.5 * z * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * torch.pow(z, 3.0))))

############################################################

# Define the Multi-Head SelfAttention module
############################################################
class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        # define there linear layers for q, k, v generation separately
        self.q_layer = nn.Linear(embed_dim, embed_dim)
        self.k_layer = nn.Linear(embed_dim, embed_dim)
        self.v_layer = nn.Linear(embed_dim, embed_dim)

        # define the projection layer for output
        self.proj_layer = nn.Linear(embed_dim, embed_dim)

        # define the dropout layer for attention and output calculation
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.num_head = num_head
        self.head_dim = embed_dim // num_head
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, dim = x.shape

        # >>> TODO 1: complete the forward process of the Multi-Head SelfAttention module.
        # For self attention, MultiHead(Q. K, V) = Concat(head_1, ..., head_h) W^O.
        # where head_i = Attention(q_i, k_i, v_i) = Attention(X W_i^Q, X W_i^K, X W_i^V)
        # Attention(q, k, v) = softmax(qk^T/sqrt(head_dim)) v

        # Step 1: obtain q, k, v via self.q_layer(), self.k_layer(), self.v_layer() respectively.
        # where q = Concat(q_1, ..., q_h), k = Concat(k_1, ..., k_h), q = Concat(v_1, ..., v_h)
        # the shape of q, k, v: (batch_size, seq_len, num_heads * head_dim)
        q, k, v = self.q_layer(x), self.k_layer(x), self.v_layer(x)

        # Step 2: in order to calculate multi-head attention in paralle, reshape q, k, v first.
        # use `Tensor.view()` or `Tensor.reshape()` to reshape q, k, v to: (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_head, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_head, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_head, self.head_dim)

        # Step 3: calculate multi-head attention in paralle: Attention(q, k, v) = softmax(qk^T / sqrt(head_dim)) v
        # Step 3.1: use `Tensor.transpose()` or `Tensor.permute()` to exchange the dim of q, k, v for matrix multiplication
        # the shape of q, v from (batch_size, seq_len, num_heads, head_dim) to (batch_size, num_heads, seq_len, head_dim)
        # the shape of k from (batch_size, seq_len, num_heads, head_dim) to (batch_size, num_heads, head_dim, seq_len)
        q = q.transpose(1, 2)
        k = k.permute(0, 2, 3, 1)
        v = v.transpose(1, 2)
        # Step 3.2: do matrix multiplication via `torch.matmul()`: attn = qk^T / sqrt(head_dim)
        # the shape of `attn`: (batch_size, num_heads, seq_len, seq_len)
        attn = torch.matmul(q, k) / math.sqrt(self.head_dim)
        # Step 3.3: fill the position of `attn` where `attn_mask==True` with value `float('-inf')` via `Tensor.masked_fill()`
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        # Step 3.4: normalize `attn` via softmax funtion: attn = Softmax(attn) = Softmax(qk^T / sqrt(head_dim))
        attn = F.softmax(attn, dim=-1)
        # Step 3.5: apply dropout to `attn` via self.attn_drop()
        attn = self.attn_drop(attn)
        # Step 3.6: multiply v by `attn`: out = Attention(q, k, v) = attn v
        # the shape of `out`: (batch_size, num_heads, seq_len, head_dim)
        out = torch.matmul(attn, v)

        # Step 4: use `Tensor.transpose()` and `Tensor.reshape()' to concatenate output of different heads
        # the shape of `out` from (batch_size, num_heads, seq_len, head_dim) to (batch_size, seq_len, num_heads*head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.num_head * self.head_dim)

        # Step 5: obtain the final results via self.proj_layer() and self.proj_drop(): result = Dropout(MultiHead(Q, K, V)) = Dropout(Concat(head_1, ..., head_h) W^O) = Dropout(out W^O)
        result = self.proj_layer(out)
        result = self.proj_drop(result)

        # <<< TODO 1

        # return the final results `result` and attention weights `attn`
        return result, attn
    
############################################################
    
# Define the feed forward network (FFN)
############################################################
class FFN(nn.Module):
    def __init__(self, embed_dim, feedforward_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
############################################################

# Define the TransformerLayer
############################################################
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, feedforward_dim, dropout, no_res):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_head, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, feedforward_dim, dropout)
        self.no_res = no_res

    def forward(self, x, attn_mask=None):
        t, attn = self.attn(self.norm1(x), attn_mask)
        x = x + t if not self.no_res else t
        x = x + self.ffn(self.norm2(x)) if not self.no_res else self.ffn(self.norm2(x))
        return x, attn
############################################################

# Define the GPT module
############################################################
class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layer, embed_dim, num_head, feedforward_dim, dropout, no_res=False, no_pos=False):
        '''
            vocab_size: the size of vocabulary
            max_seq_len: the maximum length of input texts
            num_layer: the number of transformer layers
            embed_dim: the embedding dimension
            num_head: the number of heads in Multi-Head Self Attention
            feedforward_dim: the dimension in the feed forward network
            dropout: dropout ratio
        '''
        super().__init__()
        self.num_layer = num_layer
        self.max_seq_len = max_seq_len
        self.no_pos = no_pos

        # Define Embedding Layer to transfer input text tokens and positions to embeddings
        self.word_token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.word_pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.drop = nn.Dropout(dropout)
        # Define the transformer layers
        self.transformer = nn.ModuleList([TransformerLayer(embed_dim, num_head, feedforward_dim, dropout, no_res) for _ in range(num_layer)])
        
        # Define the head layer to predict output
        self.norm = nn.LayerNorm(embed_dim)
        self.language_model_head = nn.Linear(embed_dim, vocab_size, bias=False)

        """
        Weight tying improves the performance of language models by tying (sharing) the weights of the embedding and softmax layers.
        Reference: https://paperswithcode.com/method/weight-tying
        """
        self.word_token_embedding.weight = self.language_model_head.weight

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj_layer.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layer))

    
    def forward(self, word_idx, targets=None):
        batch_size, seq_len = word_idx.shape

        # >>> TODO 2: complete the forward process of GPT
        # Step 1: use torch.arange(?, dtype=torch.long, device=word_idx.device) to generate the position sequence `pos` [0, 1, ..., seq_len-1] 
        pos = torch.arange(seq_len, dtype=torch.long, device=word_idx.device)

        # Step 2: use self.word_token_embedding() and self.word_pos_embedding() to transfer `word_idx` and `pos` to embeddings ('token_embed` and `pos_embed`)
        token_embed = self.word_token_embedding(word_idx)
        pos_embed = self.word_pos_embedding(pos)
        # Step 3: use `if` to decide whether to add pos embeddings to token embeddings: x = Dropout(?) if self.no_pos else Dropout(?)       (self.drop())
        x = self.drop(token_embed) if self.no_pos else self.drop(token_embed + pos_embed)

        # Step 4: for Auto-Regressive Language Model, the predictions for position i can depend only on the input at positions less than i.
        # Therefore, a mask is used to prevent positions from attending to subsequent positions
        # attn_mask = (0,1,...,1; 0,0,1,...,0; ...; 0,...,0) is a upper triangular matrix with shape (seq_len, seq_len)
        # Hint:
        # Step 4.1: use torch.ones(?, device=word_idx.device) to generate a matrix filled with value 1 with shape (seq_len, seq_len)
        attn_mask = torch.ones((seq_len, seq_len), device=word_idx.device).byte()

        # Step 4.2: use torch.triu(?, diagonal=1) to obtain a upper triangular matrix where elements on the main diagonal are 0
        attn_mask = torch.triu(attn_mask, diagonal=1)

        # define a list `attention_weights` and append the attention weights of each transformer layer into the list
        # Step 5: use for loop to obtain the output and attention weights of transformer layers
        attention_weights = []
        for i in range(self.num_layer):
            x, attn = self.transformer[i](x, attn_mask)
            attention_weights.append(attn)

        # Step 6: use self.norm() to normalize the output of transformer layers and then use self.language_model_head() to obtain the predictions `logits`
        # Note: do not use softmax here since it is included in the cross entropy loss function
        logits = self.language_model_head(self.norm(x))

        # <<< TODO 2

        # return logits and loss or attention weights
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        assert isinstance(attention_weights, list), "attention_weights must be a list, please check whether to append the attention weights of all transformer layers into it!"
        return logits, attention_weights

    def configure_optimizers(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('language_model_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
############################################################

GPTConfig = {
    'mygpt': dict(num_layer=4, embed_dim=128, num_head=4, feedforward_dim=128*4, dropout=0.0),
}