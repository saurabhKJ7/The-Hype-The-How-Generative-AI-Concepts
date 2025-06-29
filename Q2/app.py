"""Main Flask application for the AI Coding Agent Recommendation System."""

from flask import Flask, render_template, request
from recommender.engine import recommend_agents

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle user input and display recommendations."""
    if request.method == 'POST':
        task_description = request.form['task_description']
        recommendations = recommend_agents(task_description)
        return render_template('index.html', recommendations=recommendations, task_description=task_description)
    return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)