from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=['POST'])
def analyze_knee():
    data = request.data
    return "Hello World!"

if __name__ == '__main__':
    app.run()