"""Knowledge base containing information about AI coding agents and their capabilities."""

AI_AGENTS = {
    'github_copilot': {
        'name': 'GitHub Copilot',
        'strengths': [
            'Code generation',
            'Code completion',
            'Multi-language support',
            'IDE integration',
            'Context awareness'
        ],
        'best_for': [
            'General coding tasks',
            'Boilerplate code generation',
            'Documentation writing',
            'Test case generation'
        ],
        'languages': ['All major programming languages'],
        'complexity_handling': ['Basic', 'Intermediate', 'Advanced'],
        'special_features': [
            'Real-time suggestions',
            'Natural language understanding',
            'Code pattern recognition'
        ]
    },
    'cursor': {
        'name': 'Cursor',
        'strengths': [
            'Code editing',
            'Code explanation',
            'Debugging assistance',
            'Code refactoring'
        ],
        'best_for': [
            'Code understanding',
            'Bug fixing',
            'Code modernization',
            'Performance optimization'
        ],
        'languages': ['Most modern programming languages'],
        'complexity_handling': ['Intermediate', 'Advanced'],
        'special_features': [
            'Built-in IDE features',
            'Advanced code analysis',
            'Chat interface'
        ]
    },
    'replit_ghostwriter': {
        'name': 'Replit Ghostwriter',
        'strengths': [
            'Code completion',
            'Code generation',
            'Educational support',
            'Collaborative features'
        ],
        'best_for': [
            'Learning programming',
            'Quick prototyping',
            'Educational projects',
            'Team collaboration'
        ],
        'languages': ['Popular programming languages'],
        'complexity_handling': ['Basic', 'Intermediate'],
        'special_features': [
            'Web-based IDE',
            'Real-time collaboration',
            'Educational focus'
        ]
    },
    'aws_codewhisperer': {
        'name': 'AWS CodeWhisperer',
        'strengths': [
            'AWS integration',
            'Security scanning',
            'Code completion',
            'Best practices suggestions'
        ],
        'best_for': [
            'AWS development',
            'Cloud applications',
            'Secure coding',
            'Infrastructure as code'
        ],
        'languages': ['Major programming languages', 'Infrastructure as Code'],
        'complexity_handling': ['Intermediate', 'Advanced'],
        'special_features': [
            'AWS service integration',
            'Security scanning',
            'Best practices enforcement'
        ]
    },
    'tabnine': {
        'name': 'Tabnine',
        'strengths': [
            'Code completion',
            'Pattern recognition',
            'Team learning',
            'Privacy focus'
        ],
        'best_for': [
            'Code completion',
            'Team development',
            'Private codebases',
            'Repetitive coding tasks'
        ],
        'languages': ['All major programming languages'],
        'complexity_handling': ['Basic', 'Intermediate'],
        'special_features': [
            'Local processing option',
            'Team code patterns',
            'Privacy controls'
        ]
    }
}

def get_all_agents():
    """Return all available AI coding agents."""
    return AI_AGENTS

def get_agent(agent_id):
    """Return specific agent information."""
    return AI_AGENTS.get(agent_id)

def get_agent_names():
    """Return list of available agent names."""
    return [agent['name'] for agent in AI_AGENTS.values()]