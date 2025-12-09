import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from model_utils import load_model, predict_image

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL = load_model()   # Load once on startup

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "Empty filename!"

    if not allowed(file.filename):
        return "Unsupported file type!"

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    pil_img = Image.open(save_path)
    results = predict_image(MODEL, pil_img)

    return render_template("result.html", results=results, image_path="/" + save_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
