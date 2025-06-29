from calculator.engine import InferenceCalculator
from typing import Dict, List

class UseCaseAnalyzer:
    def __init__(self):
        self.calculator = InferenceCalculator()
    
    def analyze_chatbot(self) -> Dict:
        """Analyze chatbot with 1k tokens on GPU"""
        return self.calculator.estimate_inference(
            model_size='7B',
            tokens=1000,
            batch_size=1,
            hardware_type='T4',
            deployment_mode='local'
        ).__dict__
    
    def analyze_batch_summarization(self) -> List[Dict]:
        """Analyze batch summarization for 10 documents"""
        results = []
        
        # Test different batch sizes
        for batch_size in [1, 5, 10]:
            estimate = self.calculator.estimate_inference(
                model_size='13B',
                tokens=2000,  # Assuming 2K tokens per document
                batch_size=batch_size,
                hardware_type='A100',
                deployment_mode='local'
            ).__dict__
            
            results.append({
                'batch_size': batch_size,
                'estimate': estimate
            })
        
        return results
    
    def analyze_code_generation(self) -> Dict:
        """Analyze real-time code generation with 4 users"""
        # Using GPT-4 API for best code generation quality
        single_request = self.calculator.estimate_inference(
            model_size='GPT-4',
            tokens=1500,  # Typical code generation request
            batch_size=1,
            hardware_type='API-only',
            deployment_mode='API-hosted'
        ).__dict__
        
        return {
            'single_request': single_request,
            'hourly_cost_4_users': single_request['cost_usd'] * 4 * 10,  # Assuming 10 requests per hour per user
            'max_concurrent_requests': 'Unlimited (API-based)',
            'recommended_setup': 'API deployment for scalability and latest model access'
        }
    
    def generate_insights(self) -> Dict:
        return {
            'chatbot': {
                **self.analyze_chatbot(),
                'insights': [
                    'T4 GPU provides good cost-performance ratio for chat',
                    'Single batch processing is suitable for real-time responses',
                    'Memory usage allows for multiple model instances'
                ]
            },
            'batch_summarization': {
                'results': self.analyze_batch_summarization(),
                'insights': [
                    'A100 enables efficient batch processing',
                    'Optimal batch size depends on memory vs latency tradeoff',
                    'Cost per document decreases with larger batch sizes'
                ]
            },
            'code_generation': {
                **self.analyze_code_generation(),
                'insights': [
                    'GPT-4 API provides best code quality',
                    'API costs are predictable and scale with usage',
                    'No infrastructure management required'
                ]
            }
        }