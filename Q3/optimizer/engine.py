from tools.knowledge_base import KnowledgeBase, AITool

class OptimizerEngine:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()

    def optimize_prompt(self, base_prompt: str, tool_name: str) -> (str, str):
        tool = self.knowledge_base.get_tool(tool_name)
        if not tool:
            return base_prompt, "Tool not found."

        optimized_prompt, explanation = self._apply_optimization(base_prompt, tool)
        return optimized_prompt, explanation

    def _apply_optimization(self, prompt: str, tool: AITool) -> (str, str):
        """Apply tool-specific optimization strategies."""
        explanation = f"Optimization for {tool.name}:\n"
        optimized_prompt = prompt

        if tool.name == 'GitHub Copilot':
            optimized_prompt = f"# Task: {prompt}\n# Steps:\n# 1. [Step 1]\n# 2. [Step 2]"
            explanation += "- Added structured comments to guide Copilot.\n"
            explanation += "- Broke down the task into clear steps."

        elif tool.name == 'Cursor':
            optimized_prompt = f"""\n**Task:** {prompt}\n\n**Instructions:**\n- Please provide the complete code solution.\n- Use markdown for the final output.\n"""
            explanation += "- Formatted prompt with markdown for clarity.\n"
            explanation += "- Added explicit instructions for output format."

        elif tool.name == 'Replit Ghostwriter':
            optimized_prompt = f"// Objective: {prompt}\n// Provide a concise and efficient solution."
            explanation += "- Used comments to state the objective clearly.\n"
            explanation += "- Requested a concise solution suitable for rapid prototyping."

        elif tool.name == 'AWS CodeWhisperer':
            optimized_prompt = f"# AWS Task: {prompt}\n# Ensure the solution follows AWS best practices."
            explanation += "- Highlighted the task as AWS-specific.\n"
            explanation += "- Added a reminder to follow AWS best practices."

        elif tool.name == 'Tabnine':
            optimized_prompt = f"// To-do: {prompt}\n// Follow existing project patterns."
            explanation += "- Used a common comment format to trigger suggestions.\n"
            explanation += "- Encouraged consistency with the existing codebase."

        elif tool.name == 'Claude':
            optimized_prompt = f"System: You are a helpful coding assistant.\nUser: {prompt}\nAssistant: Here is the optimized code:"
            explanation += "- Used a system prompt to set the context.\n"
            explanation += "- Structured the prompt as a conversation for better interaction."

        return optimized_prompt, explanation