import json
import subprocess
import sys

import pytest
import torch


def test_installed_demo_cli():
    result = subprocess.run(
        [sys.executable, "-m", "hvp_patching", "demo"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(result.stdout)
    assert values["activation_patching"] == pytest.approx(3.2)
    assert values["ms_hvp_k5"] == pytest.approx(3.192)
    assert set(values) == {"activation_patching", "attribution_patching", "hvp", "ms_hvp_k5"}


def test_head_cli_records_metadata_and_does_not_overwrite(tmp_path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(0)
    directory = tmp_path / "model"
    vocab = {"[UNK]": 0, "John": 1, "Mary": 2, "gave": 3, "to": 4, "a": 5}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(directory)
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        attn_implementation="eager",
    )
    GPT2LMHeadModel(config).save_pretrained(directory)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps(
            {
                "clean": "John gave to",
                "corrupt": "Mary gave to",
                "correct": " Mary",
                "counterfactual": " John",
            }
        )
        + "\n"
    )
    output = tmp_path / "scores.json"
    command = [
        sys.executable,
        "-m",
        "hvp_patching",
        "heads",
        "--model",
        str(directory),
        "--pairs",
        str(pairs),
        "--heads",
        "0:0",
        "--steps",
        "2",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(output.read_text())
    assert len(payload["records"]) == 1
    assert payload["metadata"]["attention"] == "eager"
    assert payload["metadata"]["direction"] == "corrupt-minus-clean; expansion at clean"
    assert set(payload["mean_absolute_error"]) == {"atp", "hvp", "ms_hvp"}
    assert set(payload["records"][0]) == {
        "prompt_index",
        "head",
        "activation_patching",
        "atp",
        "quadratic_correction",
        "hvp",
        "ms_hvp",
    }
    before = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "choose a new output path" in repeated.stderr
    assert output.read_bytes() == before
