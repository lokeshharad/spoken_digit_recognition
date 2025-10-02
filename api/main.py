# # app.py
# import os
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# import uvicorn
# import numpy as np
# import tensorflow as tf

# # Import your feature extractor
from src.feature_extractor import extract_mel_spectrogram

# # Load model globally
# MODEL_PATH = "models/best_model.h5"  # adjust path
# model = tf.keras.models.load_model(MODEL_PATH)

# app = FastAPI()

# # Serve uploaded files
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# @app.get("/", response_class=HTMLResponse)
# def index():
#     return """
#     <html>
#         <head>
#             <title>Spoken Digit Recognition</title>
#         </head>
#         <body>
#             <h1>Upload Audio File</h1>
#             <form action="/predict" enctype="multipart/form-data" method="post">
#                 <input name="file" type="file" accept="audio/*"/>
#                 <input type="submit" value="Predict"/>
#             </form>
#             <h3>Uploaded audio will appear below after prediction</h3>
#         </body>
#     </html>
#     """


# @app.post("/predict", response_class=HTMLResponse)
# async def predict(file: UploadFile = File(...)):
#     # Save uploaded file
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Preprocess audio with your feature extractor at 8kHz
#     x = extract_mel_spectrogram(file_path)  # your function should handle sr
#     x = np.expand_dims(x, axis=0)             # add batch dimension

#     # Predict
#     preds = model.predict(x)
#     predicted_class = np.argmax(preds, axis=1)[0]

#     return f"""
#     <html>
#         <head>
#             <title>Prediction Result</title>
#         </head>
#         <body>
#             <h2>Predicted Digit: {predicted_class}</h2>
#             <audio controls>
#                 <source src="/uploads/{file.filename}" type="audio/wav">
#                 Your browser does not support the audio element.
#             </audio>
#             <br><br>
#             <a href="/">Upload another file</a>
#         </body>
#     </html>
#     """


# # if __name__ == "__main__":
# #     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)



# app.py
# import os
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# import uvicorn
# import numpy as np
# import tensorflow as tf
# # from src.features import extract_features  # your feature extractor
# from src.feature_extractor import extract_mel_spectrogram

# # Load model once globally
# MODEL_PATH = "models/best_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# app = FastAPI()

# # Serve uploaded files
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# @app.get("/", response_class=HTMLResponse)
# def index():
#     return """
#     <html>
#     <head>
#         <title>Spoken Digit Recognition</title>
#         <style>
#             body {
#                 font-family: Arial, sans-serif;
#                 background-color: #f7f9fc;
#                 display: flex;
#                 flex-direction: column;
#                 align-items: center;
#                 justify-content: flex-start;
#                 padding-top: 50px;
#             }
#             h1 { color: #333; }
#             form {
#                 background: #fff;
#                 padding: 30px;
#                 border-radius: 10px;
#                 box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
#                 display: flex;
#                 flex-direction: column;
#                 gap: 15px;
#                 width: 300px;
#             }
#             input[type="file"] {
#                 padding: 5px;
#             }
#             input[type="submit"] {
#                 padding: 10px;
#                 background-color: #4CAF50;
#                 color: white;
#                 border: none;
#                 border-radius: 5px;
#                 cursor: pointer;
#                 font-weight: bold;
#             }
#             input[type="submit"]:hover {
#                 background-color: #45a049;
#             }
#             audio {
#                 margin-top: 20px;
#                 width: 300px;
#             }
#             .result {
#                 margin-top: 20px;
#                 font-size: 20px;
#                 color: #222;
#             }
#             a {
#                 margin-top: 20px;
#                 text-decoration: none;
#                 color: #4CAF50;
#                 font-weight: bold;
#             }
#             a:hover { color: #45a049; }
#         </style>
#     </head>
#     <body>
#         <h1>🎤 Spoken Digit Recognition</h1>
#         <form action="/predict" enctype="multipart/form-data" method="post">
#             <input name="file" type="file" accept="audio/*" required/>
#             <input type="submit" value="Predict"/>
#         </form>
#     </body>
#     </html>
#     """


# @app.post("/predict", response_class=HTMLResponse)
# async def predict(file: UploadFile = File(...)):
#     # Save uploaded file
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Preprocess audio
#     x = extract_mel_spectrogram(file_path)  # your feature extractor
#     x = np.expand_dims(x, axis=0)  # add batch dimension

#     # Predict
#     preds = model.predict(x)
#     predicted_class = np.argmax(preds, axis=1)[0]

#     return f"""
#     <html>
#     <head>
#         <title>Prediction Result</title>
#         <style>
#             body {{
#                 font-family: Arial, sans-serif;
#                 background-color: #f7f9fc;
#                 display: flex;
#                 flex-direction: column;
#                 align-items: center;
#                 justify-content: flex-start;
#                 padding-top: 50px;
#             }}
#             .result {{
#                 background: #fff;
#                 padding: 30px;
#                 border-radius: 10px;
#                 box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
#                 text-align: center;
#             }}
#             audio {{
#                 margin-top: 20px;
#                 width: 300px;
#             }}
#             a {{
#                 margin-top: 20px;
#                 text-decoration: none;
#                 color: #4CAF50;
#                 font-weight: bold;
#             }}
#             a:hover {{ color: #45a049; }}
#         </style>
#     </head>
#     <body>
#         <div class="result">
#             <h2>Predicted Digit: {predicted_class}</h2>
#             <audio controls>
#                 <source src="/uploads/{file.filename}" type="audio/wav">
#                 Your browser does not support the audio element.
#             </audio>
#             <br>
#             <a href="/">⬅ Upload another file</a>
#         </div>
#     </body>
#     </html>
#     """


# # if __name__ == "__main__":
# #     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# app.py
# import os
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# import uvicorn
# import numpy as np
# import tensorflow as tf
# # from src.features import extract_features  # your feature extractor
# from src.feature_extractor import extract_mel_spectrogram

# # Load model
# MODEL_PATH = "models/best_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# app = FastAPI()

# # Serve uploaded files
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# @app.get("/", response_class=HTMLResponse)
# def index():
#     return """
#     <html>
#     <head>
#         <title>Spoken Digit Recognition</title>
#         <style>
#             body { font-family: Arial; background: #f7f9fc; display:flex; flex-direction:column; align-items:center; padding-top:50px;}
#             h1 { color:#333;}
#             form { background:#fff; padding:30px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); display:flex; flex-direction:column; gap:15px; width:300px;}
#             input[type="submit"] { padding:10px; background:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer; font-weight:bold;}
#             input[type="submit"]:hover { background:#45a049; }
#             audio { margin-top:20px; width:300px; }
#             canvas { margin-top:20px; border:1px solid #ccc; width:300px; height:100px;}
#             .result { background:#fff; padding:20px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); text-align:center; margin-top:20px;}
#             a { margin-top:20px; text-decoration:none; color:#4CAF50; font-weight:bold;}
#             a:hover { color:#45a049; }
#         </style>
#     </head>
#     <body>
#         <h1>🎤 Spoken Digit Recognition</h1>
#         <form action="/predict" enctype="multipart/form-data" method="post">
#             <input name="file" type="file" accept="audio/*" required/>
#             <input type="submit" value="Predict"/>
#         </form>
#     </body>
#     </html>
#     """


# @app.post("/predict", response_class=HTMLResponse)
# async def predict(file: UploadFile = File(...)):
#     # Save uploaded file
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Preprocess and predict
#     x = extract_mel_spectrogram(file_path)
#     x = np.expand_dims(x, axis=0)
#     preds = model.predict(x)
#     predicted_class = np.argmax(preds, axis=1)[0]

#     return f"""
#     <html>
#     <head>
#         <title>Prediction Result</title>
#     </head>
#     <body style="font-family: Arial; display:flex; flex-direction:column; align-items:center; padding-top:50px; background:#f7f9fc;">
#         <div class="result">
#             <h2>Predicted Digit: {predicted_class}</h2>
#             <audio controls>
#                 <source src="/uploads/{file.filename}" type="audio/wav">
#                 Your browser does not support the audio element.
#             </audio>
#             <canvas id="waveform"></canvas>
#             <br>
#             <a href="/">⬅ Upload another file</a>
#         </div>

#         <script>
#         // Draw waveform using Web Audio API
#         const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
#         const canvas = document.getElementById('waveform');
#         const ctx = canvas.getContext('2d');
#         const audioUrl = '/uploads/{file.filename}';

#         fetch(audioUrl)
#             .then(response => response.arrayBuffer())
#             .then(arrayBuffer => audioCtx.decodeAudioData(arrayBuffer))
#             .then(audioBuffer => {{
#                 const rawData = audioBuffer.getChannelData(0); // first channel
#                 const samples = 500; // number of points in waveform
#                 const blockSize = Math.floor(rawData.length / samples);
#                 const filteredData = [];
#                 for(let i=0; i<samples; i++){{
#                     let sum=0;
#                     for(let j=0;j<blockSize;j++){{
#                         sum += Math.abs(rawData[(i*blockSize)+j]);
#                     }}
#                     filteredData.push(sum/blockSize);
#                 }}

#                 // Normalize
#                 const multiplier = canvas.height / Math.max(...filteredData);

#                 // Draw
#                 ctx.clearRect(0,0,canvas.width,canvas.height);
#                 ctx.beginPath();
#                 ctx.moveTo(0, canvas.height/2);
#                 filteredData.forEach((v,i) => {{
#                     const x = (i/canvas.width)*canvas.width;
#                     const y = (canvas.height/2) - v*multiplier;
#                     ctx.lineTo(x, y);
#                 }});
#                 ctx.strokeStyle = '#4CAF50';
#                 ctx.lineWidth = 2;
#                 ctx.stroke();
#             }})
#             .catch(e => console.error(e));
#         </script>
#     </body>
#     </html>
#     """


# # if __name__ == "__main__":
# #     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# app.py
import os
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
import base64
import librosa
# from src.features import extract_features  # your feature extractor
from src.feature_extractor import extract_mel_spectrogram

# Load model
MODEL_PATH = "models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

# Serve uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def get_waveform_plot(audio_path, sr=8000):
    """Return Plotly waveform HTML div"""
    y, _ = librosa.load(audio_path, sr=sr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode='lines', line=dict(color='#4CAF50')))
    fig.update_layout(
        xaxis=dict(title='Samples'),
        yaxis=dict(title='Amplitude'),
        margin=dict(l=20, r=20, t=20, b=20),
        height=200,
        template='plotly_white'
    )
    return fig.to_html(full_html=False)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>Spoken Digit Recognition</title>
        <style>
            body { font-family: Arial; background:#f7f9fc; display:flex; flex-direction:column; align-items:center; padding:30px; }
            h1 { color:#333; }
            form, .record-box { background:#fff; padding:20px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); margin-bottom:20px;}
            input[type="submit"], button { padding:10px; background:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer; font-weight:bold; }
            input[type="submit"]:hover, button:hover { background:#45a049; }
            audio { margin-top:10px; width:300px; }
        </style>
    </head>
    <body>
        <h1>🎤 Spoken Digit Recognition</h1>
        <div class="record-box">
            <h3>Record Audio</h3>
            <button id="recordButton">Start Recording</button>
            <button id="stopButton" disabled>Stop & Submit</button>
        </div>
        <form action="/predict" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="audio/*" required/>
            <input type="submit" value="Predict"/>
        </form>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            const recordButton = document.getElementById('recordButton');
            const stopButton = document.getElementById('stopButton');

            recordButton.onclick = async () => {{
                audioChunks = [];
                const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.start();
                recordButton.disabled = true;
                stopButton.disabled = false;
            }};

            stopButton.onclick = () => {{
                mediaRecorder.stop();
                mediaRecorder.onstop = async () => {{
                    const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                    const formData = new FormData();
                    formData.append("file", audioBlob, "recorded.wav");
                    const response = await fetch("/predict", {{ method: "POST", body: formData }});
                    const html = await response.text();
                    document.body.innerHTML = html;
                }};
            }};
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Preprocess and predict
    x = extract_mel_spectrogram(file_path)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    predicted_class = np.argmax(preds, axis=1)[0]

    # Generate waveform plot
    waveform_div = get_waveform_plot(file_path, sr=8000)

    return f"""
    <html>
    <head>
        <title>Prediction Result</title>
    </head>
    <body style="font-family:Arial; display:flex; flex-direction:column; align-items:center; padding:30px; background:#f7f9fc;">
        <div class="result" style="background:#fff; padding:20px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            <h2>Predicted Digit: {predicted_class}</h2>
            <audio controls>
                <source src="/uploads/{file.filename}" type="audio/wav">
            </audio>
            <div>{waveform_div}</div>
            <br>
            <a href="/">⬅ Upload or Record Another File</a>
        </div>
    </body>
    </html>
    """


# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
