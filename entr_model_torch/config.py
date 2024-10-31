from typing import NamedTuple
from enum import Enum
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerateConfig:
    """Configuration class for text generation parameters.
    
    Attributes:
        prompt (str): The input text to generate from.
        max_tokens (int, optional): Maximum number of tokens to generate.
            Defaults to 600. Range: 1-2048.
        debug (bool, optional): Enable debug output during generation.
            Defaults to True.
        stream (bool, optional): Stream tokens as they're generated.
            Defaults to True.
        csv_file (str, optional): Path to CSV file containing prompts.
            Defaults to None.
    """
    prompt: str = "Tell me a joke"
    max_tokens: Optional[int] = 600
    debug: bool = True
    stream: bool = True
    csv_file: Optional[str] = None

    def __post_init__(self):
        """Validate inputs after initialization."""
        if self.csv_file is None:
            if not isinstance(self.prompt, str):
                raise ValueError("prompt must be a string")
            if not self.prompt.strip():
                raise ValueError("prompt cannot be empty")
            
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int):
                raise ValueError("max_tokens must be an integer")
            if self.max_tokens < 1 or self.max_tokens > 2048:
                raise ValueError("max_tokens must be between 1 and 2048")

    @classmethod
    def help(cls) -> str:
        """Return helpful information about using this configuration class."""
        return """
GenerateConfig Usage:
--------------------
Required:
- prompt (str): The text prompt to generate from
    Example: --config.prompt "Once upon a time"
OR
- csv file (str): path to csv file containing string prompts with column header 'prompts'
    Example: --config.csv_file "prompts.csv"

Optional:
- max_tokens (int): How many tokens to generate (1-2048)
    Default: 600
    Usage: --config.max_tokens 1000
- debug: Toggle debug information during generation
    Default: True
    Usage: --config.debug or --config.no-debug
- stream: Toggle output token streaming
    Default: True
    Usage: --config.stream or --config.no-stream

Example usage:
    python3 -m entr_model_torch.main --config.prompt "Which number is larger 9.11 or 9.9? be brief in your response" --config.no-stream --config.debug
    or
    python3 -m entr_model_torch.main --config.csv_file "prompts.csv" --config.stream --config.debug
"""

class EntropixConfig:
    def __init__(self):
        # Sampler state toggles
        ## Low Entropy, Low Varentropy: "flowing with unspoken intent"
        self.state_flowing = True
        ## High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
        self.state_treading = True
        ## Low Entropy, High Varentropy: "exploring forks in the path"
        self.state_exploring = True
        ## High Entropy, High Varentropy: "resampling in the mist"
        self.state_resampling = True

        # Sampler state extras
        self.state_extras_agreement = False
        self.state_extras_interaction_strength = False

        # Adaptive state dynamic top_p, top_k, min_p adjustment toggles (old)
        '''self.state_dynamic_top_p = True
        self.state_dynamic_top_k = True
        self.state_dynamic_min_p = True'''

params = {
    "dim": 960,
    "n_layers": 32,
    "n_heads": 15,
    "n_kv_heads": 5,
    "vocab_size": 49152,
    "norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "use_scaled_rope": False,  # Inferred from "rope_scaling": null
    "max_seq_len": 2048,  # Inferred from "max_position_embeddings"
}


class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool

MODEL_ID = 'HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_PATH = 'weights/360M-Instruct'

SMOLLM_360M_PARAMS = ModelParams(
  n_layers=params["n_layers"],
  n_local_heads=params["n_heads"],
  n_local_kv_heads=params["n_kv_heads"],
  head_dim=params["dim"] // params["n_heads"],
  max_seq_len=params["max_seq_len"],
  rope_theta=params["rope_theta"],
  use_scaled_rope=params["use_scaled_rope"]
)

# Experimental custom config to trigger different sampler states
class SamplerConfig:
    def __init__(self):
        self.temperature = 0.666
        self.top_p = 0.90
        self.top_k = 27
        self.min_p = 0.03

        self.low_logits_entropy_threshold = 0.5
        self.medium_logits_entropy_threshold = 1.484
        self.high_logits_entropy_threshold = 2.07

        self.low_logits_varentropy_threshold = 1.28
        self.medium_logits_varentropy_threshold = 3.75
        self.high_logits_varentropy_threshold = 6.08

        self.low_attention_entropy_threshold = 5.875
        self.medium_attention_entropy_threshold = 6.125
        self.high_attention_entropy_threshold = 6.415

        self.low_attention_varentropy_threshold = 7.125
        self.medium_attention_varentropy_threshold = 7.6
        self.high_attention_varentropy_threshold = 8.25

        self.low_agreement_threshold = 2e-06
        self.medium_agreement_threshold = 4e-06
        self.high_agreement_threshold = 5e-06

        self.low_interaction_strength_threshold = 0.2
        self.medium_interaction_strength_threshold = 0.247
        self.high_interaction_strength_threshold = 0.264

        self.high_entropy_attention_offset = 1.3
        self.high_entropy_attention_coefficient = 0.2

        self.low_entropy_interaction_strength_offset = 1.2
        self.low_entropy_interaction_strength_coefficient = 0.3

        self.high_entropy_varentropy_attention_offset = 2.0
        self.high_entropy_varentropy_attention_coefficient = 0.5

        self.n_adaptive_samples = 5

        self.adaptive_temperature_logits_coefficient = 0.3
        self.adaptive_temperature_attention_coefficient = 0.2
        self.adaptive_temperature_agreement_coefficient = 0.2
        self.adaptive_top_p_coefficient = 0.1
        self.adaptive_top_k_interaction_coefficient = 0.3
        self.adaptive_top_k_agreement_coefficient = 0.2
        self.adaptive_min_p_coefficient = 0.5
        self.adaptive_score_logits_entropy_coefficient = 0.1
        self.adaptive_score_attention_entropy_coefficient = 0.2
        self.adaptive_score_logits_varentropy_coefficient = 0.3
        self.adaptive_score_attention_varentropy_coefficient = 0.4
        self.adaptive_score_agreement_coefficient = 0.5
        self.adaptive_score_interaction_strength_coefficient = 0.6

class SamplerState(Enum):
    FLOWING = "Flowing with unspoken intent"
    TREADING = "Treading carefully, asking clarifying questions"
    EXPLORING = "Exploring forks in the path"
    RESAMPLING = "Resampling in the mist"
    ADAPTIVE = "Adaptive Sampling"
