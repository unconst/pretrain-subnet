from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/loss_dict', methods=['GET'])
def get_loss_dict():
    """Endpoint to get the loss_dict."""
    try:
        with open('loss_dict.json', 'r') as f:
            loss_dict = json.load(f)
        return jsonify(loss_dict)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)