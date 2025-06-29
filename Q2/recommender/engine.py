"""Recommendation engine for AI coding agents."""

import re
from agents.knowledge_base import get_all_agents

def analyze_task(task_description):
    """Analyze the task description to extract key features."""
    task_type = 'unknown'
    if re.search(r'debug|fix|error', task_description, re.IGNORECASE):
        task_type = 'debugging'
    elif re.search(r'generate|create|build', task_description, re.IGNORECASE):
        task_type = 'code_generation'
    elif re.search(r'refactor|modernize|optimize', task_description, re.IGNORECASE):
        task_type = 'refactoring'

    complexity = 'basic'
    if re.search(r'advanced|complex|performance', task_description, re.IGNORECASE):
        complexity = 'advanced'
    elif re.search(r'intermediate|moderate', task_description, re.IGNORECASE):
        complexity = 'intermediate'

    languages = []
    if re.search(r'python', task_description, re.IGNORECASE):
        languages.append('Python')
    if re.search(r'typescript|ts', task_description, re.IGNORECASE):
        languages.append('TypeScript')
    if re.search(r'javascript|js', task_description, re.IGNORECASE):
        languages.append('JavaScript')

    return {'type': task_type, 'complexity': complexity, 'languages': languages}

def score_agent(agent, task_features):
    """Score an agent based on its suitability for the task."""
    score = 0
    justification = []

    if task_features['type'] in agent['best_for']:
        score += 3
        justification.append(f"Strongly recommended for {task_features['type']}.")

    if task_features['complexity'] in agent['complexity_handling']:
        score += 2
        justification.append(f"Handles {task_features['complexity']} tasks well.")

    if not task_features['languages'] or any(lang in agent['languages'] for lang in task_features['languages']):
        score += 1

    return score, ' '.join(justification)

def recommend_agents(task_description):
    """Recommend the top 3 AI coding agents for a given task."""
    task_features = analyze_task(task_description)
    agents = get_all_agents()
    recommendations = []

    for agent_id, agent_info in agents.items():
        score, justification = score_agent(agent_info, task_features)
        if score > 0:
            recommendations.append({
                'name': agent_info['name'],
                'score': score,
                'justification': justification
            })

    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:3]