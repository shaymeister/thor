from flask import Flask
from Camera import Camera

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

def stream():
    cam = Camera()
if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0')