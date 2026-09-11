from dataclasses import dataclass, field
from typing import Optional, List
from ..base import BaseConfig

@dataclass
class BaseSearchConfig(BaseConfig):
    """Base configuration class for all search algorithms.
    
    Inherits common attributes from BaseConfig:
        - policy_model_name: Primary language model name
        - gpu_device: GPU device identifier
        - max_length: Maximum token length for generation
        - max_steps: Maximum search depth (replaces depth_limit for consistency)
    
    Config Args (via --search-arg):
        n_actions: Number of actions to generate per expansion (default: 3)
        max_steps: Maximum search depth/reasoning steps (default: 10, inherited from BaseConfig)
        force_terminating_on_depth_limit: Force termination when max_steps reached (default: True)
        terminate_on_first_solution: Stop search at the first qualifying terminal (default: True)
        early_stop_reward: Minimum reward for early termination; only used with terminate_on_first_solution (default: None = accept any terminal)
        r_terminating: Transition-only threshold for its 'reward_threshold' constraint (default: None)
        add_continuation: Enable continuation phase for sequential reasoning (default: False)
        reward_alpha: Exponent for fast reward transformation (default: None)
        reward_beta: Confidence threshold for state transition (default: None)
        reward_gamma: Threshold for BN evaluator (default: None)
    """
    # Action generation
    n_actions: int = 3
    runtime_limit_before_iter: int = None

    # LLM models
    eval_model_name: str = None
    enable_think_policy: bool = True
    enable_think_eval: bool = True
    enable_think_terminal_gen: bool = False

    # Terminate parameters
    terminate_constraints: list[str] = field(default_factory=list)
    terminate_ORM_name: str = None
    transition_model_name: str = None
    r_terminating: Optional[float] = None  # transition termination only; not a search-stop threshold
    sample_size_terminate: int = None
    sample_threshold_terminate: float = None
    sample_threshold_verify: float = None
    force_terminating_on_depth_limit: bool = True
    terminate_on_first_solution: bool = True  # stop at the first terminal satisfying early_stop_reward
    early_stop_reward: Optional[float] = None  # if set with terminate_on_first_solution, only stop when terminal reward >= this threshold

    # Continuation parameters
    bn_model_name: str = None
    add_continuation: bool = False
    reward_alpha: float = None # for fast reward
    reward_beta: float = None # for confidence of state transition
    reward_gamma: float = None # for BNEvaluator
    reward_gamma1: float = None
    n_actions_for_bne: int = None
    bn_method: str = None
    only_continuation_at_head: bool = False
    max_new_tokens_for_bn_eval: int = None
    max_try_for_bn_eval: int = 3

    # Fast reward evaluation
    think_for_usefulness: bool = True
    think_for_correctness: bool = True
    n_for_correctness: int = 5
    n_for_usefulness: int = 5
