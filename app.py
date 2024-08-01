from flask import Flask, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
