"""BloomBee SDK: run reinforcement learning and custom training on BloomBee's pooled GPUs.

    import bloombee_sdk
    client = bloombee_sdk.Client()                      # BLOOMBEE_API_KEY and BLOOMBEE_BASE_URL from the environment
    with client.training_session("TinyLlama-1.1B-Chat", rank=16, weight_bits=4) as tc:
        history = bloombee_sdk.rl.grpo(tc, prompts, reward_fn, iterations=20, group_size=8)
        tc.save("my-grpo-model")                  # appears in My models; deploy it like any other
"""
from bloombee_sdk.client import (BadRequest, Client, NotSignedIn, Sample, SessionNotRunning, BloomBeeError, TrainingSession)
from bloombee_sdk import rl

__all__ = ["Client", "TrainingSession", "Sample", "BloomBeeError", "BadRequest", "NotSignedIn", "SessionNotRunning", "rl"]
__version__ = "0.1.0"
