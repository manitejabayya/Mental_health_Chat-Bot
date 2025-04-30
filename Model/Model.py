import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

class SpeechEmotionModel:
    def __init__(self, data_path=None, audio_dir='files'):
        # Use the provided data_path or construct a default path
        if data_path:
            self.data_path = data_path
        else:
            # Default path now points to the Model folder
            self.data_path = os.path.join(os.getcwd(), 'Model', 'speech_emotions.csv')
        
        self.audio_dir = audio_dir
        self.emotions = ['euphoric', 'joyfully', 'sad', 'surprised']
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # Print the path being used for debugging
        print(f"Looking for CSV file at: {self.data_path}")
        
    def load_data(self):
        """Load and prepare data from the CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"CSV file not found at: {self.data_path}")
        
        # Load the CSV file into a pandas DataFrame
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} entries from CSV file.")
        return self.df
    
    def extract_audio_features(self, file_path, max_pad_len=174):
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            
            # Pad or truncate to standardize length
            if mfccs.shape[1] < max_pad_len:
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]
                
            return mfccs
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None
    
    def prepare_dataset(self):
        """Prepare dataset for training"""
        features = []
        labels = []
        
        # First check if audio directory exists
        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError(f"Audio directory not found at: {self.audio_dir}")
        
        # Process through folders containing audio files
        for root, dirs, files in os.walk(self.audio_dir):
            for filename in files:
                if filename.endswith('.wav'):
                    # Extract emotion from filename
                    emotion = filename.split('.')[0]
                    if emotion in self.emotions:
                        file_path = os.path.join(root, filename)
                        
                        # Extract features
                        audio_features = self.extract_audio_features(file_path)
                        if audio_features is not None:
                            features.append(audio_features)
                            labels.append(emotion)
        
        if not features:
            raise ValueError("No valid audio files found for training")
        
        # Convert lists to numpy arrays
        X = np.array(features)
        
        # Check if we have any features before reshaping
        if X.shape[0] == 0:
            raise ValueError("No features extracted - check your audio files")
            
        # Reshape for LSTM input (samples, time steps, features)
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        
        # Encode labels
        self.label_encoder.fit(self.emotions)
        y = self.label_encoder.transform(labels)
        y = to_categorical(y)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build LSTM model for speech emotion recognition"""
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(LSTM(128, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(len(self.emotions), activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        # Load and prepare data
        self.load_data()
        X_train, X_test, y_train, y_test = self.prepare_dataset()
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        checkpoint = ModelCheckpoint('best_model.h5', 
                                     monitor='val_accuracy', 
                                     save_best_only=True, 
                                     mode='max', 
                                     verbose=1)
        
        early_stop = EarlyStopping(monitor='val_loss', 
                                   patience=10, 
                                   restore_best_weights=True, 
                                   verbose=1)
        
        # Train model
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(X_test, y_test),
                                 callbacks=[checkpoint, early_stop])
        
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Save model
        self.model.save('speech_emotion_model.h5')
        
        return history
    
    def predict(self, audio_file):
        """Predict emotion from audio file"""
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None
        
        # Extract features
        features = self.extract_audio_features(audio_file)
        if features is None:
            return None
        
        # Reshape for prediction
        features = np.reshape(features, (1, features.shape[1], features.shape[0]))
        
        # Predict
        prediction = self.model.predict(features)[0]
        predicted_index = np.argmax(prediction)
        predicted_emotion = self.label_encoder.inverse_transform([predicted_index])[0]
        
        return {
            'emotion': predicted_emotion,
            'confidence': float(prediction[predicted_index]),
            'all_probabilities': {emotion: float(prob) for emotion, prob in 
                                  zip(self.label_encoder.classes_, prediction)}
        }


if __name__ == "__main__":
    # Adjust the path to the CSV file if needed
    model = SpeechEmotionModel(
    data_path='C:/Users/bayya/OneDrive/Desktop/Chat-Bot/Model/speech_emotions.csv',
    audio_dir='C:/Users/bayya/OneDrive/Desktop/Chat-Bot/Model/files'
)
    try:
        history = model.train()
        print("Model training completed!")
    except Exception as e:
        print(f"Error during training: {e}")