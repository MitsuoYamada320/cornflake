from flask import Flask, render_template
from flask_cors import CORS
from api.get_texts import api  # Blueprint の読み込み

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.json.ensure_ascii = False

CORS(app)
app.register_blueprint(api, url_prefix="/api")  # API の登録

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
