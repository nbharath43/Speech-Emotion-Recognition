import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib   

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
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_dataset(dataset_path):
    features = []
    labels = []

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path '{dataset_path}' does not exist.")
    
    for actor_folder in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_folder)
        if os.path.isdir(actor_path):
            print(f"Reading folder: {actor_path}")
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    emotion = file.split("-")[2]
                    file_path = os.path.join(actor_path, file)
                    print(f"Extracting features from: {file_path}")
                    feature = extract_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(emotion)
                    else:
                        print(f"Skipping file due to feature extraction error: {file_path}")

    print(f"Total files processed: {len(features)}")
    return np.array(features), np.array(labels)

def train_model():
    dataset_path = 'D:\B1\Batch A12\speech-emotion-recognition-ravdess-data'
    features, labels = load_dataset(dataset_path)

    if features.size == 0 or labels.size == 0:
        raise ValueError("No features or labels found. Please check the dataset path and ensure the audio files are correct.")
    
    print(f"Number of samples: {len(features)}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    joblib.dump(model, 'model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

if __name__ == "__main__":
    train_model()
