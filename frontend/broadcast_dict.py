from flask import Flask, jsonify, send_from_directory
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/global_state', methods=['GET'])
def get_global_state():
    """Endpoint to get the global_state."""
    try:
        with open('static/global_state.json', 'r') as f:
            global_state = json.load(f)
        return jsonify(global_state)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=17873)