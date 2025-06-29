from flask import Flask, render_template, request
from optimizer.engine import OptimizerEngine
from tools.knowledge_base import KnowledgeBase

app = Flask(__name__)
optimizer_engine = OptimizerEngine()
knowledge_base = KnowledgeBase()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        base_prompt = request.form['prompt']
        tool_name = request.form['tool']
        optimized_prompt, explanation = optimizer_engine.optimize_prompt(base_prompt, tool_name)
        return render_template(
            'index.html',
            tools=knowledge_base.list_tools(),
            base_prompt=base_prompt,
            optimized_prompt=optimized_prompt,
            explanation=explanation,
            selected_tool=tool_name
        )
    return render_template('index.html', tools=knowledge_base.list_tools())

if __name__ == '__main__':
    app.run(debug=True, port=3000)