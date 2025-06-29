from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AITool:
    name: str
    capabilities: List[str]
    system_prompt_format: str
    strengths: List[str]
    weaknesses: List[str]
    prompt_tuning_techniques: List[str]

class KnowledgeBase:
    def __init__(self):
        self.tools: Dict[str, AITool] = {
            'github_copilot': AITool(
                name='GitHub Copilot',
                capabilities=[
                    'Real-time code completion',
                    'Context-aware suggestions',
                    'Multi-language support',
                    'Function documentation generation'
                ],
                system_prompt_format='Natural language comments and docstrings',
                strengths=[
                    'Excellent at completing partial code',
                    'Strong pattern recognition',
                    'IDE integration',
                    'Works well with existing codebases'
                ],
                weaknesses=[
                    'May suggest outdated patterns',
                    'Limited context window',
                    'No direct conversation capability'
                ],
                prompt_tuning_techniques=[
                    'Use clear function signatures',
                    'Provide example inputs/outputs',
                    'Break complex tasks into steps',
                    'Include type hints'
                ]
            ),
            'cursor': AITool(
                name='Cursor',
                capabilities=[
                    'Code generation',
                    'Code explanation',
                    'Refactoring assistance',
                    'Bug fixing'
                ],
                system_prompt_format='Markdown with code blocks',
                strengths=[
                    'Strong at explaining code',
                    'Good at complex refactoring',
                    'Supports interactive editing',
                    'Handles larger context'
                ],
                weaknesses=[
                    'May be verbose',
                    'Sometimes overcomplicates solutions',
                    'Performance varies with prompt clarity'
                ],
                prompt_tuning_techniques=[
                    'Use markdown formatting',
                    'Specify desired output format',
                    'Include context and constraints',
                    'Break down complex requirements'
                ]
            ),
            'replit_ghostwriter': AITool(
                name='Replit Ghostwriter',
                capabilities=[
                    'Code completion',
                    'Code generation',
                    'Debugging assistance',
                    'Code explanation'
                ],
                system_prompt_format='Natural language with optional code context',
                strengths=[
                    'Good at web development tasks',
                    'Integrated development environment',
                    'Real-time collaboration support',
                    'Quick prototyping'
                ],
                weaknesses=[
                    'Limited to Replit environment',
                    'May struggle with complex algorithms',
                    'Less effective for large projects'
                ],
                prompt_tuning_techniques=[
                    'Provide clear project context',
                    'Use step-by-step instructions',
                    'Include example code snippets',
                    'Specify output requirements'
                ]
            ),
            'aws_codewhisperer': AITool(
                name='AWS CodeWhisperer',
                capabilities=[
                    'Code suggestions',
                    'Security scanning',
                    'AWS service integration',
                    'Multi-language support'
                ],
                system_prompt_format='Comments with AWS-specific context',
                strengths=[
                    'Excellent for AWS services',
                    'Strong security focus',
                    'Good at cloud patterns',
                    'Supports multiple IDEs'
                ],
                weaknesses=[
                    'AWS-centric suggestions',
                    'Limited general-purpose coding',
                    'Requires AWS knowledge'
                ],
                prompt_tuning_techniques=[
                    'Include AWS service context',
                    'Specify security requirements',
                    'Use clear function purposes',
                    'Reference AWS patterns'
                ]
            ),
            'tabnine': AITool(
                name='Tabnine',
                capabilities=[
                    'Code completion',
                    'Pattern learning',
                    'Team-specific suggestions',
                    'Multi-language support'
                ],
                system_prompt_format='Code context based',
                strengths=[
                    'Fast suggestions',
                    'learns from codebase',
                    'Privacy focused',
                    'Low latency'
                ],
                weaknesses=[
                    'Limited explanation capability',
                    'Shorter context window',
                    'May need training time'
                ],
                prompt_tuning_techniques=[
                    'Use consistent coding style',
                    'Maintain clear structure',
                    'Follow project patterns',
                    'Keep context focused'
                ]
            ),
            'claude': AITool(
                name='Claude',
                capabilities=[
                    'Code generation',
                    'Code explanation',
                    'Problem solving',
                    'Technical writing'
                ],
                system_prompt_format='Natural language with optional system instructions',
                strengths=[
                    'Strong reasoning ability',
                    'Handles complex instructions',
                    'Good at explanations',
                    'Flexible output format'
                ],
                weaknesses=[
                    'May be overly verbose',
                    'No direct IDE integration',
                    'Response time can vary'
                ],
                prompt_tuning_techniques=[
                    'Provide clear constraints',
                    'Use structured instructions',
                    'Include example outputs',
                    'Specify format requirements'
                ]
            )
        }
    
    def get_tool(self, tool_name: str) -> AITool:
        """Get the AI tool information by name."""
        return self.tools.get(tool_name.lower())
    
    def list_tools(self) -> List[str]:
        """Get a list of all available AI tool names."""
        return list(self.tools.keys())