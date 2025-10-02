import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import moviepy.editor as moviepy
import uvicorn
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
from src.feature_extractor import extract_mel_spectrogram
import librosa
from pydub import AudioSegment

# Load model
MODEL_PATH = "models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

# Serve uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def get_waveform_plot(audio_path, sr=8000):
    y, _ = librosa.load(audio_path, sr=sr)
    x = np.arange(len(y)) / sr  # Convert samples to time in seconds
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#4CAF50')))
    fig.update_layout(
        xaxis=dict(title='Time (s)'),
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
    body { font-family: Arial; background:#f7f9fc; display:flex; flex-direction:column; align-items:center; padding:30px; margin:0;}
    h1 { color:#333; }
    form, .record-box { background:#fff; padding:20px; border-radius:10px; box-shadow:0 10px 20px rgba(0,0,0,0.1); margin-bottom:20px; width:100%; max-width:1200px; box-sizing:border-box;}
    input[type="submit"], button { padding:10px; background:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer; font-weight:bold; }
    input[type="submit"]:hover, button:hover { background:#45a049; }
    audio { margin-top:10px; width:300px; }
    #progressContainer { width:100%; max-width:600px; background:#ddd; border-radius:5px; margin-top:10px; }
    #progressBar { width:0%; height:20px; background:#4CAF50; border-radius:5px; }
    #timer { margin-top:5px; font-weight:bold; }
</style>

    </head>
    <body>
        <h1>🎤 Spoken Digit Recognition</h1>
        <div class="record-box" style="width:100%; max-width:1200px; box-sizing:border-box; display:flex; align-items:center; gap:10px; padding:20px; background:#fff; border-radius:10px; box-shadow:0 10px 20px rgba(0,0,0,0.1);">
    <h3 style="margin:0; flex-shrink:0;">Record Audio:</h3>
    <button id="recordButton">Record</button>
    <button id="stopButton" disabled>Stop</button>
    <div id="progressContainer" style="flex:1; background:#ddd; height:20px; border-radius:5px;">
        <div id="progressBar" style="width:0%; height:100%; background:#4CAF50; border-radius:5px;"></div>
    </div>
    <div id="timer" style="min-width:50px; text-align:center;">00:00</div>
</div>


        <form action="/predict" enctype="multipart/form-data" method="post" 
      style="display:flex; align-items:center; gap:10px; background:#fff; padding:15px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); width:100%; max-width:1200px; box-sizing:border-box;">
    <h3 style="margin:0; flex-shrink:0;">Upload Audio:</h3>
    <input name="file" type="file" accept="audio/*" required 
           style="width:100%; max-width:700px; padding:8px; border:1px solid #ccc; border-radius:5px; cursor:pointer;" />
    <input type="submit" value="Predict" 
           style="padding:10px 20px; background:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer; font-weight:bold;" />
</form>


        <script>
            let mediaRecorder;
            let audioChunks = [];
            let interval;
            let seconds = 0;
            const recordButton = document.getElementById('recordButton');
            const stopButton = document.getElementById('stopButton');
            const progressBar = document.getElementById('progressBar');
            const timer = document.getElementById('timer');

            function updateProgress() {
                seconds++;
                const mins = String(Math.floor(seconds/60)).padStart(2,'0');
                const secs = String(seconds%60).padStart(2,'0');
                timer.textContent = `${mins}:${secs}`;
                progressBar.style.width = Math.min((seconds/60)*100, 100) + '%'; // 1 min max
            }

            recordButton.addEventListener('click', async () => {
                audioChunks = [];
                seconds = 0;
                progressBar.style.width = '0%';
                timer.textContent = '00:00';

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.start();

                    interval = setInterval(updateProgress, 1000);

                    recordButton.disabled = true;
                    stopButton.disabled = false;
                } catch(err) {
                    alert("Microphone access denied or not supported: " + err);
                }
            });

            stopButton.addEventListener('click', () => {
                if (!mediaRecorder) return;
                mediaRecorder.stop();
                clearInterval(interval);

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const formData = new FormData();
                    formData.append("file", audioBlob, "recorded.webm");

                    const response = await fetch("/predict", { method: "POST", body: formData });
                    const html = await response.text();
                    document.open();
                    document.write(html);
                    document.close();
                };

                recordButton.disabled = false;
                stopButton.disabled = true;
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    print("file path  is", file_path)
    wav_path = os.path.join(UPLOAD_DIR, "converted.wav")
    clip = moviepy.AudioFileClip(file_path)
    clip.write_audiofile(wav_path)

    # Preprocess and predict
    x = extract_mel_spectrogram(wav_path)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    predicted_class = np.argmax(preds, axis=1)[0]

    # Generate waveform plot
    waveform_div = get_waveform_plot(wav_path, sr=8000)

    return f"""<html>
<head>
    <title>Prediction Result</title>
</head>
<body style="font-family:Arial; display:flex; flex-direction:column; align-items:center; padding:30px; background:#f7f9fc; margin:0;">
    <div style="background:#fff; padding:50px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); text-align:center; width:100%; max-width:1200px; box-sizing:border-box;">
        <h2>Predicted Digit: {predicted_class}</h2>
        <audio controls style="width:80%; max-width:600px;">
            <source src="/uploads/{file.filename}" type="audio/wav">
        </audio>
        <div style="margin-top:20px;">{waveform_div}</div>
        <br>
        <a href="/">⬅ Upload or Record Another File</a>
    </div>
</body>
</html>
"""
# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
