import argparse
import json
from calculator.engine import InferenceCalculator
from scenarios.use_case_analysis import UseCaseAnalyzer

def main():
    parser = argparse.ArgumentParser(description='LLM Inference Calculator')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Calculator command
    calc_parser = subparsers.add_parser('calculate', help='Estimate inference metrics')
    calc_parser.add_argument('--model_size', required=True, choices=['7B', '13B', 'GPT-4'])
    calc_parser.add_argument('--tokens', required=True, type=int)
    calc_parser.add_argument('--batch_size', required=True, type=int)
    calc_parser.add_argument('--hardware_type', required=True, choices=['A100', 'A10G', 'T4', 'CPU', 'API-only'])
    calc_parser.add_argument('--deployment_mode', required=True, choices=['local', 'API-hosted'])
    
    # Scenarios command
    subparsers.add_parser('scenarios', help='Analyze real-world use cases')
    
    args = parser.parse_args()
    
    if args.command == 'calculate':
        calculator = InferenceCalculator()
        estimate = calculator.estimate_inference(
            model_size=args.model_size,
            tokens=args.tokens,
            batch_size=args.batch_size,
            hardware_type=args.hardware_type,
            deployment_mode=args.deployment_mode
        )
        print(json.dumps(estimate.__dict__, indent=4))
    
    elif args.command == 'scenarios':
        analyzer = UseCaseAnalyzer()
        insights = analyzer.generate_insights()
        print(json.dumps(insights, indent=4))

if __name__ == '__main__':
    main()