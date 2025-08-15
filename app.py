import os
import time
import shutil
import uuid
import pandas as pd
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import subprocess
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

# ✅ Load YOLO model
model_path = os.path.join("helmet_model", "weights", "best.pt")
model = YOLO(model_path)

# ✅ Load model metrics from results.csv
results_csv = os.path.join("helmet_model", "results.csv")
precision, recall, map50 = 0, 0, 0
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    last_row = df.iloc[-1]
    precision = round(last_row["metrics/precision(B)"] * 100, 2)
    recall = round(last_row["metrics/recall(B)"] * 100, 2)
    map50 = round(last_row["metrics/mAP50(B)"] * 100, 2)

# ✅ Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        unique_input_name = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_input_name)
        file.save(filepath)

        # ✅ Start timer
        start_time = time.time()

        # ✅ Run YOLO prediction
        results = model.predict(
    source=filepath,
    imgsz=640,
    conf=0.25,
    save=True,
    vid_stride=3  # ✅ process every 3rd frame
)


        detection_time = round(time.time() - start_time, 2)

        # ✅ Get YOLO output folder
        last_run_dir = results[0].save_dir

        # ✅ Find predicted file
        predicted_file = None
        for f in os.listdir(last_run_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov")):
                predicted_file = os.path.join(last_run_dir, f)
                break

        if not predicted_file:
            return "Prediction file not found", 500

        # ✅ Save final file
        final_filename = f"{uuid.uuid4().hex}_{os.path.basename(predicted_file)}"
        final_output = os.path.join(app.config['OUTPUT_FOLDER'], final_filename)
        shutil.copy(predicted_file, final_output)
        

        if final_output.lower().endswith(".avi"):
            mp4_output = final_output.replace(".avi", ".mp4")
            ffmpeg_path = r"C:\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"

            subprocess.run([ffmpeg_path, "-i", final_output, "-vcodec", "libx264", "-crf", "18", mp4_output])
            os.remove(final_output)  # remove .avi
            final_filename = os.path.basename(mp4_output)
 


        # ✅ Count detections
        boxes = results[0].boxes
        names = model.names
        helmet_count = 0
        no_helmet_count = 0

        for box in boxes:
            cls_id = int(box.cls)
            label = names[cls_id]
            if label == "h":
                helmet_count += 1
            elif label == "nh":
                no_helmet_count += 1

        # ✅ Delete runs folder
        if os.path.exists("runs"):
            shutil.rmtree("runs")

        return render_template(
            "result.html",
            filename=final_filename,
            detection_time=detection_time,
            helmet_count=helmet_count,
            no_helmet_count=no_helmet_count,
            precision=precision,
            recall=recall,
            map50=map50
        )

    return render_template("index.html", precision=precision, recall=recall, map50=map50)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response

@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
