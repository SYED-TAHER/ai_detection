from flask import Flask, render_template, request, jsonify
from model.ai_detector import detect_ai_generated

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        text = request.form['text']
        if not text.strip():
            return jsonify(result={"error": "Text input is empty. Please provide valid input."})

        result = detect_ai_generated(text)
        return jsonify(result=result)
    except Exception as e:
        return jsonify(result={"error": f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
