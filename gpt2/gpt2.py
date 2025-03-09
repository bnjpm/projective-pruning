import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class GPT2Config:
    n_layer: int
    n_head: int
    qk_dim: int
    v_dim: int
    embed_dim: int
    block_size: int
    vocab_size: int
    intermediate_dim: int

    def qkv_dim(self):
        return 2 * self.qk_dim + self.v_dim


class GPT2MHA(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.c_attn = nn.Linear(config.embed_dim, config.n_head * config.qkv_dim())
        self.c_proj = nn.Linear(config.n_head * config.v_dim, config.embed_dim)

    def forward(self, x: torch.Tensor):
        B, L, E = x.shape
        n_head = self.config.n_head
        qk_dim = self.config.qk_dim
        v_dim = self.config.v_dim

        qkv = self.c_attn(x)  # (B, L, n_head * qkv_dim)
        q, k, v = qkv.split((n_head * qk_dim, n_head * qk_dim, n_head * v_dim), dim=-1)
        q = q.view(B, L, n_head, qk_dim).transpose(1, 2)  # (B, n_head, L, qk_dim)
        k = k.view(B, L, n_head, qk_dim).transpose(1, 2)  # (B, n_head, L, qk_dim)
        v = v.view(B, L, n_head, v_dim).transpose(1, 2)  # (B, n_head, L, v_dim)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, L, n_head * v_dim)
        y = self.c_proj(y)
        return y


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.embed_dim, config.intermediate_dim)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.intermediate_dim, config.embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.embed_dim, eps=1e-5, elementwise_affine=True)
        self.attn = GPT2MHA(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim, eps=1e-5, elementwise_affine=True)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.embed_dim)
        self.wpe = nn.Embedding(config.block_size, config.embed_dim)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.embed_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor):
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        pos_embed = self.wpe(pos)
        tok_embed = self.wte(x)
        x = tok_embed + pos_embed
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight sharing

    def forward(self, input_ids, labels=None):
        hidden = self.transformer(input_ids)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return logits, loss


def convert_openai_statedict(sd, sd_openai):
    endswith_transpose = (
        ".attn.c_attn.weight",
        ".attn.c_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight",
    )
    for k, v in sd_openai.items():
        if k.endswith(endswith_transpose):
            sd[k].copy_(v.T)
        else:
            sd[k].copy_(v)
    return sd


def get_config(n_layer=12, n_head=12, embed_dim=768, block_size=1024, vocab_size=50257):
    return lambda: GPT2Config(
        n_layer=n_layer,
        n_head=n_head,
        qk_dim=64,
        v_dim=64,
        embed_dim=embed_dim,
        block_size=block_size,
        vocab_size=vocab_size,
        intermediate_dim=4 * embed_dim,
    )


CONFIG_CLASSES = {
    "gpt2": get_config(),
    "gpt2-medium": get_config(embed_dim=1024, n_layer=24, n_head=16),
    "gpt2-large": get_config(embed_dim=1280, n_layer=36, n_head=20),
    "gpt2-xl": get_config(embed_dim=1600, n_layer=48, n_head=25),
}
