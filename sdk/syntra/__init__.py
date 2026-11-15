"""Syntra SDK: run reinforcement learning and custom training on Syntra's pooled GPUs.

    import syntra
    client = syntra.Client()                      # SYNTRA_API_KEY and SYNTRA_BASE_URL from the environment
    with client.training_session("TinyLlama-1.1B-Chat", rank=16, weight_bits=4) as tc:
        history = syntra.rl.grpo(tc, prompts, reward_fn, iterations=20, group_size=8)
        tc.save("my-grpo-model")                  # appears in My models; deploy it like any other
"""
from syntra.client import (BadRequest, Client, NotSignedIn, Sample, SessionNotRunning, SyntraError, TrainingSession)
from syntra import rl

__all__ = ["Client", "TrainingSession", "Sample", "SyntraError", "BadRequest", "NotSignedIn", "SessionNotRunning", "rl"]
__version__ = "0.1.0"
