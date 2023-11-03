from flask import Flask, jsonify
import json

app = Flask(__name__)

@app.route('/global_state', methods=['GET'])
def get_global_state():
    """Endpoint to get the global_state."""
    try:
        with open('global_state.json', 'r') as f:
            global_state = json.load(f)
        return jsonify(global_state)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=17873)