"""The player: a small instruct LLM with LoRA adapters plus a value head.

The model reads the text state and answers with one digit (1-6). We read its scores for those six tokens at the
last prompt position: that is the policy. One forward pass per decision, no generation, so many games run as one batch.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("Platformer: reach the flag F on the right. Map: # solid, . air, o coin, E enemy (stomp it, never touch "
          "its side), M you; floor _ = pit. Moves: 1 run, 2 run+long jump, 3 run+hop, 4 left, 5 jump, 6 wait. "
          "Reply with one digit.")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tiny_model(tok):
    """A tiny random Qwen2 (for smoke tests on a laptop)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg)


class Policy(nn.Module):
    def __init__(self, model_name: str = DEFAULT_MODEL, adapter: str | None = None, lora_r: int = 16,
                 device: str | None = None, dtype=None, tiny: bool = False, trainable: bool = True):
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device_name = device or pick_device()
        if dtype is None:
            # fp32 on purpose: in bf16 a state's move scores shift with batch/padding (PPO ratios become noise)
            # and small LoRA updates round away. TF32 matmuls keep it fast.
            dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        base = _tiny_model(self.tok) if tiny else AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        base = base.to(dtype)
        if adapter:
            self.model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
        else:
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, target_modules=LORA_TARGETS,
                             task_type="CAUSAL_LM")
            self.model = get_peft_model(base, cfg)
        hidden = base.config.hidden_size
        # value head: small MLP on a *detached* hidden state, so the value loss never moves the LoRA weights
        # (only the policy gradient / imitation loss does). Keeps PPO updates about the policy.
        self.value = nn.Sequential(nn.Linear(hidden, 256), nn.Tanh(), nn.Linear(256, 1))
        nn.init.zeros_(self.value[2].weight)
        nn.init.zeros_(self.value[2].bias)
        vpath = os.path.join(adapter, "value_head.pt") if adapter else ""
        if vpath and os.path.exists(vpath):
            try:
                self.value.load_state_dict(torch.load(vpath, map_location="cpu"))
            except RuntimeError:
                print("value head in checkpoint has another shape: starting a fresh one", flush=True)
        from mario.game.textstate import TEXT_VERSION
        self.text_version = TEXT_VERSION
        mpath = os.path.join(adapter, "meta.json") if adapter else ""
        if mpath and os.path.exists(mpath):
            self.text_version = json.load(open(mpath)).get("text_version", 1)
        self.action_ids = []
        for d in "123456":
            ids = self.tok.encode(d, add_special_tokens=False)
            assert len(ids) == 1, f"digit {d} is not one token"
            self.action_ids.append(ids[0])
        self._prefix_cache: dict[str, list[int]] = {}
        self.use_prefix_cache = True
        self.dtype = dtype
        self.amp = False   # fp16 autocast for supervised training on GPUs without TF32 (not for PPO: ratios get noisy)
        self.to(self.device_name)

    # ---------- text -> tokens ----------
    def prompt(self, text: str) -> str:
        return self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                                            tokenize=False, add_generation_prompt=True)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """Chat-template tokens; the fixed system part is tokenized once (same ids as tokenizing the full prompt)."""
        if not self._prefix_cache:
            marker = "\x00STATE\x00"
            full = self.prompt(marker)
            pre, post = full.split(marker)
            self._prefix_cache = {"pre": self.tok(pre, add_special_tokens=False)["input_ids"],
                                  "post": self.tok(post, add_special_tokens=False)["input_ids"]}
        pre, post = self._prefix_cache["pre"], self._prefix_cache["post"]
        body = self.tok(texts, add_special_tokens=False)["input_ids"]
        if self.use_prefix_cache:   # the shared system part is computed once per batch in forward()
            return [b + post for b in body]
        return [pre + b + post for b in body]

    def pad(self, seqs: list[list[int]]):
        n = max(len(s) for s in seqs)
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = torch.tensor(s)
            mask[i, n - len(s):] = 1
        return ids.to(self.device_name), mask.to(self.device_name)

    # ---------- forward ----------
    def forward(self, ids, mask):
        if self.amp and str(ids.device).startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                logits, value = self._forward(ids, mask)
            return logits.float(), value.float()
        return self._forward(ids, mask)

    def _forward(self, ids, mask):
        """Returns (logits over the 6 moves [B,6] float32, value [B] float32)."""
        base = self.model.get_base_model()
        if self.use_prefix_cache:
            # The system prompt is the same for every game: run it once (with LoRA, so gradients flow through it),
            # repeat its keys/values for the batch, then run only the per-game tokens. Same result, ~2.7x fewer tokens.
            from transformers import DynamicCache
            if not self._prefix_cache:
                self.tokenize(["x"])
            pre = torch.tensor([self._prefix_cache["pre"]], device=ids.device)
            P, B = pre.shape[1], ids.shape[0]
            cache = DynamicCache(config=base.config)
            base.model(input_ids=pre, position_ids=torch.arange(P, device=ids.device)[None], past_key_values=cache,
                       use_cache=True)
            cache.batch_repeat_interleave(B)
            full_mask = torch.cat([torch.ones(B, P, dtype=mask.dtype, device=mask.device), mask], 1)
            pos = P + (mask.cumsum(-1) - 1).clamp(min=0)
            out = base.model(input_ids=ids, attention_mask=full_mask, position_ids=pos, past_key_values=cache,
                             use_cache=True)
        else:
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            out = base.model(input_ids=ids, attention_mask=mask, position_ids=pos)
        h = out.last_hidden_state[:, -1]
        w = base.lm_head.weight[self.action_ids]
        logits = (h @ w.t()).float()
        value = self.value(h.float().detach()).squeeze(-1)
        return logits, value

    def act_texts(self, texts: list[str]):
        ids, mask = self.pad(self.tokenize(texts))
        return self(ids, mask)

    # ---------- checkpoints ----------
    def save(self, path: str, meta: dict | None = None):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        torch.save(self.value.state_dict(), os.path.join(path, "value_head.pt"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"model": self.model_name, "text_version": self.text_version, **(meta or {})}, f, indent=1)

    def trainable_parameters(self):
        lora = [p for n, p in self.model.named_parameters() if p.requires_grad]
        return lora, list(self.value.parameters())
