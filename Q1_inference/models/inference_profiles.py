from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelProfile:
    name: str
    parameters: int  # billions
    context_window: int
    memory_per_token: float  # GB per 1K tokens
    base_latency: float  # ms for first token
    token_latency: float  # ms per token
    cost_per_1k_tokens: float  # USD
    min_gpu_memory: float  # GB
    supported_hardware: List[str]

MODEL_PROFILES = {
    '7B': ModelProfile(
        name='LLaMA-2 7B',
        parameters=7,
        context_window=4096,
        memory_per_token=0.0042,  # Approximate
        base_latency=150,
        token_latency=30,
        cost_per_1k_tokens=0.0002,
        min_gpu_memory=14,
        supported_hardware=['A100', 'A10G', 'T4', 'CPU']
    ),
    '13B': ModelProfile(
        name='LLaMA-2 13B',
        parameters=13,
        context_window=4096,
        memory_per_token=0.0078,
        base_latency=200,
        token_latency=40,
        cost_per_1k_tokens=0.0004,
        min_gpu_memory=26,
        supported_hardware=['A100', 'A10G']
    ),
    'GPT-4': ModelProfile(
        name='GPT-4',
        parameters=100,  # Approximate
        context_window=8192,
        memory_per_token=0.0,  # Not applicable for API
        base_latency=300,
        token_latency=50,
        cost_per_1k_tokens=0.03,
        min_gpu_memory=0.0,  # Not applicable for API
        supported_hardware=['API-only']
    )
}

# Hardware specifications
HARDWARE_SPECS = {
    'A100': {
        'memory': 80,  # GB
        'cost_per_hour': 2.50
    },
    'A10G': {
        'memory': 24,
        'cost_per_hour': 1.20
    },
    'T4': {
        'memory': 16,
        'cost_per_hour': 0.60
    },
    'CPU': {
        'memory': 32,  # System RAM
        'cost_per_hour': 0.10
    }
}