from flask import Flask, request, render_template, redirect, flash
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Ensure that model files are in the same directory
model_path = 'model.pkl'
label_encoder_path = 'label_encoder.pkl'

if not os.path.exists(model_path) or not os.path.exists(label_encoder_path):
    raise FileNotFoundError("Model files not found. Please run model.py to generate them.")

# Load the trained model and label encoder
model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

# Define the mapping of numerical labels to emotion names
emotion_mapping = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
    '09': 'contempt',
    '10': 'confused',
    '11': 'excited',
    '12': 'bored',
    '13': 'tense',
    '14': 'content'
}

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0)
        ))
        return features
    except Exception as e:
        flash(f"Error extracting features: {str(e)}", 'danger')
        return None

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error in feature extraction"
    try:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        emotion_label = label_encoder.inverse_transform(prediction)[0]
        return emotion_mapping.get(emotion_label, "Unknown")
    except Exception as e:
        flash(f"Error predicting emotion: {str(e)}", 'danger')
        return "Error in prediction"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', file_name=None, emotion=None)

@app.route('/record', methods=['POST'])
def record():
    emotion = None
    file_name = None
    if 'audio' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    file_name = audio_file.filename
    file_ext = os.path.splitext(file_name)[1].lower()
    if file_ext not in ['.txt','.wav', '.mp3', '.flac', '.ogg']:
        flash('Unsupported file type', 'danger')
        return redirect(request.url)
    
    audio_file_path = 'temp_audio' + file_ext
    audio_file.save(audio_file_path)

    try:
        # Display the file name before processing
        if file_ext != '.wav':
            audio = AudioSegment.from_file(audio_file_path)
            audio_file_path = 'temp_audio.wav'
            audio.export(audio_file_path, format='wav')

        # Predict the emotion
        emotion = predict_emotion(audio_file_path)
    
    except Exception as e:
        flash(f"Error processing file: {str(e)}", 'danger')
    
    finally:
        # Clean up temporary files
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
    
    return render_template('index.html', file_name=file_name, emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)
