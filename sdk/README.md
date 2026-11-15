# bloombee-sdk

Reinforcement learning and custom training for LoRA / QLoRA adapters on BloomBee's pooled GPUs. The interface is modelled on
Tinker and OpenPipe ART: you get low-level primitives (`sample`, `forward_backward`, `optim_step`, `logprobs`) plus ready
GRPO, PPO and DPO loops. Your reward code runs on your machine; only token ids and numbers are sent.

```bash
pip install -e sdk/                       # needs only httpx
export BLOOMBEE_API_KEY=syn_key_...         # create one: POST /api-keys (website: account menu)
export BLOOMBEE_BASE_URL=https://your-bloombee-server
```

```python
import json, bloombee_sdk

client = bloombee_sdk.Client()
with client.training_session("TinyLlama-1.1B-Chat", rank=16, weight_bits=4, gpus=2) as tc:   # QLoRA on 2 pooled GPUs
    def reward(prompt, completion, sample):
        try:
            json.loads(completion); return 1.0
        except ValueError:
            return 0.0
    prompts = ["Ticket: my package is late.\nAnswer:", "Ticket: I was charged twice.\nAnswer:"]
    history = bloombee_sdk.rl.grpo(tc, prompts, reward, iterations=20, group_size=8, max_new_tokens=32, lr=1e-5, kl_coef=0.02)
    model_id = tc.save("ticket-router-grpo")   # appears in My models: deploy, compare, evaluate
```

Low level, your own loop:

```python
samples = tc.sample(["Q: 2+2=\nA:"], n=8, max_new_tokens=8, temperature=1.0)
rewards = [1.0 if s.text.strip().startswith("4") else 0.0 for s in samples]
adv, _ = bloombee_sdk.rl.group_advantages(rewards, group_size=8)
batch = [{"tokens": s.tokens, "prompt_len": s.prompt_len, "advantage": a, "old_logprobs": s.logprobs} for s, a in zip(samples, adv)]
tc.forward_backward(batch, loss="ppo", clip=0.2)     # or "importance_sampling", "cross_entropy", "dpo"
tc.optim_step(lr=1e-5)
```

| Call | What it does |
|---|---|
| `client.training_session(base, init_model=, rank=, weight_bits=16 or 4, gpus=)` | Lease GPUs, start a trainer. `init_model` continues from one of your models; that model is also the reference policy. |
| `tc.sample(prompts, n, max_new_tokens, temperature, top_p, top_k, seed)` | Completions with per-token log-probs under the current policy. |
| `tc.logprobs(seqs, reference=False)` | Log-probs of given completions, under the policy or the reference. |
| `tc.forward_backward(batch, loss, **args)` | Accumulates gradients. Losses: `cross_entropy`, `dpo` (beta), `ppo` (clip, kl_coef), `importance_sampling`. |
| `tc.optim_step(lr, weight_decay, max_grad_norm)` | AdamW step on the adapter. |
| `tc.save(name)` / `tc.close(save_as=)` | Store the adapter as a model / release the GPUs (billing stops). |

Behaviour to know:
- **Retries**: every state-changing call carries an idempotency key, so a retry after a network error never applies a step twice.
- **GPU failures**: workers are stateless. A dead GPU is replaced and the call continues; the adapter and optimizer live in the trainer.
- **Idle timeout**: a session with no call for 10 minutes is saved ("… (auto-saved)") and closed, so GPUs are not held by a forgotten script.
- **Rejected requests** (`BadRequest`): wrong shapes or lengths, sequences longer than `max_seq_len`, non-finite rewards or advantages, `optim_step` with nothing accumulated. Nothing changes on the server.
- Groups whose completions all got the same reward have no learning signal; `grpo` skips them and reports `informative_fraction`.
