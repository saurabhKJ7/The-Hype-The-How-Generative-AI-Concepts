from dataclasses import dataclass
from typing import Dict, List, Tuple
from models.inference_profiles import HARDWARE_SPECS, MODEL_PROFILES

@dataclass
class InferenceEstimate:
    latency_ms: float
    memory_gb: float
    cost_usd: float
    hardware_compatible: bool
    compatibility_reason: str

class InferenceCalculator:
    def __init__(self):
        self.model_profiles = MODEL_PROFILES
        self.hardware_specs = HARDWARE_SPECS
    
    def estimate_inference(
        self,
        model_size: str,
        tokens: int,
        batch_size: int,
        hardware_type: str,
        deployment_mode: str
    ) -> InferenceEstimate:
        model = self.model_profiles[model_size]
        
        # Check hardware compatibility
        compatible, reason = self._check_hardware_compatibility(
            model, hardware_type, deployment_mode
        )
        
        # Calculate latency
        latency = self._calculate_latency(
            model, tokens, batch_size, deployment_mode
        )
        
        # Calculate memory
        memory = self._calculate_memory(
            model, tokens, batch_size, deployment_mode
        )
        
        # Calculate cost
        cost = self._calculate_cost(
            model, tokens, batch_size, hardware_type, deployment_mode
        )
        
        return InferenceEstimate(
            latency_ms=latency,
            memory_gb=memory,
            cost_usd=cost,
            hardware_compatible=compatible,
            compatibility_reason=reason
        )
    
    def _check_hardware_compatibility(
        self,
        model: 'ModelProfile',
        hardware_type: str,
        deployment_mode: str
    ) -> Tuple[bool, str]:
        if deployment_mode == 'API-hosted':
            if model.name == 'GPT-4':
                return True, 'GPT-4 is only available via API'
            return False, f'{model.name} is designed for local deployment'
        
        if hardware_type not in model.supported_hardware:
            return False, f'{hardware_type} is not supported for {model.name}'
        
        if hardware_type != 'CPU' and \
           self.hardware_specs[hardware_type]['memory'] < model.min_gpu_memory:
            return False, f'Insufficient GPU memory for {model.name}'
        
        return True, 'Hardware is compatible'
    
    def _calculate_latency(
        self,
        model: 'ModelProfile',
        tokens: int,
        batch_size: int,
        deployment_mode: str
    ) -> float:
        # Base latency + per-token latency
        base = model.base_latency
        per_token = model.token_latency * tokens
        
        # Batch processing reduces per-token latency but adds overhead
        if batch_size > 1:
            per_token = (per_token * batch_size) / 1.5  # Parallel processing gain
            base *= 1.2  # Batch overhead
        
        # API deployment adds network latency
        if deployment_mode == 'API-hosted':
            base += 100  # Additional network latency
        
        return base + per_token
    
    def _calculate_memory(
        self,
        model: 'ModelProfile',
        tokens: int,
        batch_size: int,
        deployment_mode: str
    ) -> float:
        if deployment_mode == 'API-hosted':
            return 0.0  # Memory not applicable for API deployment
        
        # Base model memory
        base_memory = model.min_gpu_memory
        
        # Token memory scaling
        token_memory = (tokens / 1000) * model.memory_per_token
        
        # Batch memory scaling
        batch_memory = token_memory * batch_size
        
        # Additional memory overhead (20%)
        overhead = (base_memory + batch_memory) * 0.2
        
        return base_memory + batch_memory + overhead
    
    def _calculate_cost(
        self,
        model: 'ModelProfile',
        tokens: int,
        batch_size: int,
        hardware_type: str,
        deployment_mode: str
    ) -> float:
        if deployment_mode == 'API-hosted':
            # API pricing per 1K tokens
            return (tokens / 1000) * model.cost_per_1k_tokens * batch_size
        
        # Local deployment: hardware cost per hour
        latency_hours = self._calculate_latency(model, tokens, batch_size, deployment_mode) / (3600 * 1000)
        hardware_cost_per_hour = self.hardware_specs[hardware_type]['cost_per_hour']
        
        return latency_hours * hardware_cost_per_hour