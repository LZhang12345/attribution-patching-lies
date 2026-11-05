"""Exercise real attention hooks on a random GPT-2; no weights to download."""

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from hvp_patching.benchmark import score_heads
from hvp_patching.heads import Head, HeadInterventions


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = GPT2Config(
        vocab_size=32, n_positions=16, n_embd=32, n_layer=2, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config).eval()
    model.requires_grad_(False)
    adapter = HeadInterventions(model)
    clean = torch.tensor([[1, 5, 8, 3]])
    corrupt = torch.tensor([[1, 9, 8, 3]])
    records = score_heads(adapter, clean, corrupt, [Head(0, 0), Head(1, 1)], 4, 7)
    print(json.dumps({"example": "random tiny GPT-2; not a paper benchmark", "records": records}, indent=2))


if __name__ == "__main__":
    main()
