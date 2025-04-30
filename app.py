import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import google.generativeai as genai
from textblob import TextBlob
import time
import speech_recognition as sr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import sounddevice as sd
import soundfile as sf
import tempfile
import random
import tensorflow as tf
from tensorflow import keras
import datetime
import cv2
from PIL import Image
from deepface import DeepFace

# Configure Gemini API Key
api_key = "AIzaSyDVKSewCA6f95z63GPfFBnrA_uO7o8COII"
genai.configure(api_key=api_key)

# Set up the page
st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="💬")
st.title("🧠 AI Mental Health Chatbot")

# Use a current model from the available list
MODEL_NAME = "models/gemini-1.5-flash"

# Load the trained emotion recognition model
@st.cache_resource
def load_emotion_model():
    try:
        model = keras.models.load_model('best_model.h5')
        return model
    except Exception as e:
        st.warning(f"Error loading emotion model: {str(e)}")
        return None

# Load emotion labels
EMOTION_LABELS = ['angry', 'happy', 'neutral', 'sad']

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "chat"  # Default to chat mode
if "temp_input" not in st.session_state:
    st.session_state.temp_input = ""
if "process_input" not in st.session_state:
    st.session_state.process_input = False
if "voice_data" not in st.session_state:
    st.session_state.voice_data = None
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_features" not in st.session_state:
    st.session_state.audio_features = {}
if "recording_started" not in st.session_state:
    st.session_state.recording_started = False
if "audio_file_path" not in st.session_state:
    st.session_state.audio_file_path = None
if "countdown" not in st.session_state:
    st.session_state.countdown = 0
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "sentiment_analysis" not in st.session_state:
    st.session_state.sentiment_analysis = None
if "recording_id" not in st.session_state:
    st.session_state.recording_id = 0
if "emotion_prediction" not in st.session_state:
    st.session_state.emotion_prediction = None
if "face_analysis" not in st.session_state:
    st.session_state.face_analysis = None
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "last_face_update" not in st.session_state:
    st.session_state.last_face_update = 0
# Add user profile data
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "current_feeling": None,
        "feeling_reason": None,
        "wake_time": None,
        "sleep_time": None,
        "occupation": None,
        "age": None
    }
if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False
if "current_question" not in st.session_state:
    st.session_state.current_question = "feeling"  # Start with asking about feeling

# Add global CSS for styling the entire application
st.markdown("""
<style>
    /* Change the main title color to red */
    .st-emotion-cache-10trblm {
        color: #e53e3e !important;
    }
    
    /* Change the subheader and header colors to red */
    .st-emotion-cache-zt5igj {
        color: #e53e3e !important;
    }
    
    /* Specifically target sidebar header */
    .sidebar .st-emotion-cache-10trblm, .sidebar .st-emotion-cache-zt5igj {
        color: #e53e3e !important;
    }
    
    /* Improve text visibility for the entire app */
    body {
        color: #1a202c !important;
    }
    
    /* Make headers more visible */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
    }
    
    /* Improve button visibility */
    .stButton>button {
        background-color: #4299e1 !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Make metrics more readable */
    .css-1qg05tj {
        color: #2d3748 !important;
    }
    
    /* Sidebar content styling */
    .sidebar-content {
        color: #1a202c;
        background-color: #edf2f7;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Voice recording button styling */
    .recording-button {
        background-color: #e53e3e !important;
        color: white !important;
    }
    
    /* Profile card styling */
    .profile-card {
        background-color: #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    /* Progress bar styling */
    .onboarding-progress {
        margin-bottom: 20px;
    }
    
    /* Face analysis container */
    .face-container {
        border: 2px solid #e53e3e;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }
    
    /* Webcam feed styling */
    .webcam-feed {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    /* Emotion indicators */
    .emotion-indicator {
        margin-bottom: 5px;
    }
    
    /* Emotion bar */
    .emotion-bar {
        height: 20px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    /* Dominant emotion highlight */
    .dominant-emotion {
        font-weight: bold;
        color: #e53e3e;
    }
</style>
""", unsafe_allow_html=True)

# Function to analyze facial expressions with enhanced robustness
def analyze_facial_expression(frame):
    try:
        # Create a copy of the frame
        frame_copy = frame.copy()
        
        # Save frame to a temporary file (this approach works better with DeepFace)
        temp_img_path = os.path.join(tempfile.gettempdir(), f'temp_face_{random.randint(1000, 9999)}.jpg')
        cv2.imwrite(temp_img_path, frame_copy)
        
        # Try different backends if one fails
        backends = ['opencv', 'ssd', 'mtcnn', 'retinaface']
        
        for backend in backends:
            try:
                # Analyze face for emotions
                analysis = DeepFace.analyze(
                    img_path=temp_img_path,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend=backend
                )
                
                # If analysis succeeded, break the loop
                if analysis:
                    break
            except:
                continue
        
        # Clean up the temporary file
        try:
            os.remove(temp_img_path)
        except:
            pass
        
        if not analysis:
            return {
                'emotions': {},
                'dominant_emotion': None,
                'face_detected': False
            }
            
        if isinstance(analysis, list):
            # Get the first face if multiple detected
            analysis = analysis[0]
            
        # Get emotion scores
        emotion_scores = analysis.get('emotion', {})
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else None
        
        return {
            'emotions': emotion_scores,
            'dominant_emotion': dominant_emotion,
            'face_detected': True if emotion_scores else False
        }
        
    except Exception as e:
        # Clean up any temporary files that might have been created
        try:
            if 'temp_img_path' in locals():
                os.remove(temp_img_path)
        except:
            pass
            
        return {
            'emotions': {},
            'dominant_emotion': None,
            'face_detected': False,
            'error': str(e)
        }

# Function to extract audio features for emotion analysis
def extract_audio_features(audio_data, sample_rate):
    # Extract features that can indicate emotional state
    features = {}
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Basic audio statistics
    features["amplitude_mean"] = np.mean(np.abs(audio_data))
    features["amplitude_std"] = np.std(audio_data)
    features["energy"] = np.sum(audio_data**2) / len(audio_data)
    
    # Spectral features using librosa
    try:
        # Extract MFCC features (related to voice tone and timbre)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features["mfcc_mean"] = np.mean(mfccs)
        features["mfcc_std"] = np.std(mfccs)
        
        # Extract pitch (fundamental frequency) - related to emotional arousal
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        features["pitch_mean"] = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Speech rate estimation (syllables/phonemes per second)
        # This is a simplified approximation
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
        features["speech_rate"] = np.sum(librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)) / (len(audio_data) / sample_rate)
        
        # Spectral contrast (related to articulation and clarity)
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        features["spectral_contrast_mean"] = np.mean(contrast)
        
        # Save the raw audio data for potential model processing
        features["raw_audio"] = audio_data
        features["sample_rate"] = sample_rate
    except Exception as e:
        st.warning(f"Some advanced audio features could not be extracted: {str(e)}")
        
    return features

# Function to extract MFCC features in the format expected by the model
def extract_mfcc_features_for_model(audio_data, sample_rate, num_mfcc=13, n_fft=2048, hop_length=512):
    try:
        # Make sure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Extract MFCCs with num_mfcc=13 to match model expectations
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sample_rate, 
            n_mfcc=num_mfcc,
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        # Transpose to get time frames in the first dimension
        mfccs = mfccs.T
        
        # Ensure consistent shape (pad or truncate)
        if mfccs.shape[0] < 216:  # If shorter than expected
            pad_width = 216 - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        elif mfccs.shape[0] > 216:  # If longer than expected
            mfccs = mfccs[:216, :]
        
        # Add batch dimension for model prediction
        mfccs = np.expand_dims(mfccs, axis=0)
        
        return mfccs
    except Exception as e:
        st.error(f"Error extracting MFCC features: {str(e)}")
        return None

# Function to predict emotion from audio using the trained model
def predict_emotion(audio_data, sample_rate):
    model = load_emotion_model()
    if model is None:
        return {"emotion": "unknown", "confidence": 0}
    
    try:
        # Extract features in the format expected by the model
        features = extract_mfcc_features_for_model(audio_data, sample_rate)
        
        if features is None:
            return {"emotion": "unknown", "confidence": 0}
        
        # Make prediction
        predictions = model.predict(features)
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # Map the index to emotion label
        emotion = EMOTION_LABELS[predicted_class_index]
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": {EMOTION_LABELS[i]: float(predictions[0][i]) for i in range(len(EMOTION_LABELS))}
        }
    except Exception as e:
        st.error(f"Error predicting emotion: {str(e)}")
        return {"emotion": "unknown", "confidence": 0}

# Improved function to analyze voice for mental health indicators with more variability
def analyze_voice_emotion(audio_features, emotion_prediction=None):
    # This is a simplified model for demonstration
    # In a real application, you would use a trained machine learning model
    
    results = {}
    
    # Add a small amount of randomness to ensure variations between recordings
    # Note: This is only for demonstration purposes
    random_factor = random.uniform(0.8, 1.2)
    
    # Estimate emotional arousal (low=depressed, high=anxious/manic)
    if "amplitude_mean" in audio_features and "pitch_mean" in audio_features:
        # Make arousal calculation more sensitive to amplitude variations
        arousal = (audio_features["amplitude_mean"] * 8 * random_factor + 
                  (audio_features["pitch_mean"] / 150 if audio_features["pitch_mean"] > 0 else 0))
        results["arousal"] = min(max(arousal, 0), 1)  # Normalize between 0-1
    else:
        results["arousal"] = 0.5 * random_factor  # Default middle value with variation
    
    # Estimate emotional valence (negative vs positive emotion)
    if "spectral_contrast_mean" in audio_features and "mfcc_std" in audio_features:
        # Adjust valence calculation to be more sensitive
        valence = 0.5 + (audio_features["spectral_contrast_mean"] / 25 * random_factor) - (audio_features["mfcc_std"] / 15)
        results["valence"] = min(max(valence, -1), 1)  # Normalize between -1 and 1
    else:
        results["valence"] = 0 * random_factor  # Default neutral value with variation
    
    # Speech rate can indicate anxiety (high) or depression (low)
    if "speech_rate" in audio_features:
        speech_rate_normalized = ((audio_features["speech_rate"] - 2) / 3) * random_factor  # Normalize around average rate
        results["speech_rate_indicator"] = min(max(speech_rate_normalized, -1), 1)
    else:
        results["speech_rate_indicator"] = 0 * random_factor
    
    # Interpret the voice features into mental health indicators
    mental_health_indicators = {}
    
    # Initialize indicators with default values
    mental_health_indicators["anxiety"] = 25
    mental_health_indicators["depression"] = 25
    mental_health_indicators["stress"] = 25
    mental_health_indicators["general_wellbeing"] = 75
    
    # Incorporate trained model emotion predictions if available
    if emotion_prediction and emotion_prediction.get("emotion") != "unknown":
        emotions = emotion_prediction.get("all_emotions", {})
        
        # Use the predicted emotions to influence mental health indicators
        # Anxiety is often related to fear and sometimes anger
        anxiety_from_model = (
            emotions.get("fear", 0) * 0.7 + 
            emotions.get("angry", 0) * 0.2 + 
            emotions.get("sad", 0) * 0.1
        ) * 100
        
        # Depression is often related to sadness and lack of happiness
        depression_from_model = (
            emotions.get("sad", 0) * 0.7 + 
            (1 - emotions.get("happy", 0)) * 0.3
        ) * 100
        
        # Stress can be related to multiple emotions
        stress_from_model = (
            emotions.get("angry", 0) * 0.3 + 
            emotions.get("fear", 0) * 0.3 +
            emotions.get("sad", 0) * 0.2 +
            (1 - emotions.get("neutral", 0)) * 0.2
        ) * 100
        
        # Weight model predictions higher than feature-based analysis
        model_weight = 0.7
        feature_weight = 0.3
    else:
        # If no model prediction, rely only on feature-based analysis
        anxiety_from_model = 0
        depression_from_model = 0
        stress_from_model = 0
        model_weight = 0
        feature_weight = 1
    
    # Anxiety indicator (high arousal, negative valence, faster speech)
    anxiety_from_features = (
        results["arousal"] * 0.5 + 
        (0.5 - results["valence"]/2) * 0.3 + 
        (results["speech_rate_indicator"] if results["speech_rate_indicator"] > 0 else 0) * 0.2
    ) * 100
    
    mental_health_indicators["anxiety"] = (
        anxiety_from_model * model_weight + 
        anxiety_from_features * feature_weight
    )
    
    # Depression indicator (low arousal, negative valence, slower speech)
    depression_from_features = (
        (1 - results["arousal"]) * 0.4 + 
        (0.5 - results["valence"]/2) * 0.4 + 
        (-results["speech_rate_indicator"] if results["speech_rate_indicator"] < 0 else 0) * 0.2
    ) * 100
    
    mental_health_indicators["depression"] = (
        depression_from_model * model_weight + 
        depression_from_features * feature_weight
    )
    
    # Stress indicator (high arousal, variable pitch)
    stress_from_features = 0
    if "mfcc_std" in audio_features:
        stress_from_features = (
            results["arousal"] * 0.6 + 
            audio_features["mfcc_std"] * 0.4
        ) * 100
    else:
        stress_from_features = results["arousal"] * 0.8 * 100
    
    mental_health_indicators["stress"] = (
        stress_from_model * model_weight + 
        stress_from_features * feature_weight
    )
    
    # Scale all indicators to 0-100% and apply small randomness for natural variation
    for key in mental_health_indicators:
        mental_health_indicators[key] = min(max(mental_health_indicators[key] * random.uniform(0.9, 1.1), 0), 100)
    
    return mental_health_indicators

# Function to analyze facial expressions for mental health indicators
def analyze_face_for_mental_health(face_analysis):
    if not face_analysis or not face_analysis.get('face_detected'):
        return None
    
    emotions = face_analysis.get('emotions', {})
    dominant_emotion = face_analysis.get('dominant_emotion', 'neutral')
    
    # Initialize mental health indicators
    indicators = {
        'anxiety': 25,
        'depression': 25,
        'stress': 25,
        'general_wellbeing': 75
    }
    
    # Map emotions to mental health indicators
    emotion_weights = {
        'angry': {'stress': 0.8, 'anxiety': 0.2},
        'disgust': {'stress': 0.6, 'anxiety': 0.4},
        'fear': {'anxiety': 0.9, 'stress': 0.1},
        'happy': {'general_wellbeing': 1.0, 'depression': -0.5},
        'sad': {'depression': 0.9, 'general_wellbeing': -0.5},
        'surprise': {'stress': 0.5, 'anxiety': 0.5},
        'neutral': {}  # Neutral doesn't strongly indicate anything
    }
    
    # Apply weights based on emotion scores
    for emotion, score in emotions.items():
        weights = emotion_weights.get(emotion, {})
        for indicator, weight in weights.items():
            # Normalize score to 0-100 and apply weight
            normalized_score = (score / 100) * weight * 100
            indicators[indicator] += normalized_score
    
    # Ensure values stay within bounds
    for key in indicators:
        indicators[key] = min(max(indicators[key], 0), 100)
    
    return indicators

# Function to combine multiple analysis results (voice, face, profile)
def combine_analysis_results(voice_analysis=None, face_analysis=None, profile_analysis=None):
    # Initialize combined results with default values
    combined = {
        'anxiety': 25,
        'depression': 25,
        'stress': 25,
        'general_wellbeing': 75
    }
    
    # Count how many analyses we have
    analysis_count = 0
    
    # Add voice analysis if available
    if voice_analysis:
        analysis_count += 1
        for key in combined:
            if key in voice_analysis:
                combined[key] += voice_analysis.get(key, 0) - 25  # Adjust from absolute to delta
    
    # Add face analysis if available
    if face_analysis:
        analysis_count += 1
        for key in combined:
            if key in face_analysis:
                combined[key] += face_analysis.get(key, 0) - 25  # Adjust from absolute to delta
    
    # Add profile analysis if available
    if profile_analysis:
        analysis_count += 1
        for key in combined:
            if key in profile_analysis:
                combined[key] += profile_analysis.get(key, 0) - 25  # Adjust from absolute to delta
    
    # Average the results if we have multiple analyses
    if analysis_count > 0:
        for key in combined:
            combined[key] = combined[key] / analysis_count
    
    # Ensure values stay within bounds
    for key in combined:
        combined[key] = min(max(combined[key], 0), 100)
    
    return combined

# Function to analyze user profile data for mental health insights
def analyze_user_profile():
    profile = st.session_state.user_profile
    health_insights = {}
    
    # Default scores
    health_insights["anxiety"] = 25
    health_insights["depression"] = 25
    health_insights["stress"] = 25
    health_insights["general_wellbeing"] = 75
    
    # Sleep duration analysis
    if profile["wake_time"] and profile["sleep_time"]:
        try:
            # Convert to datetime objects for calculation
            wake_time = datetime.datetime.strptime(profile["wake_time"], "%H:%M").time()
            sleep_time = datetime.datetime.strptime(profile["sleep_time"], "%H:%M").time()
            
            # Create datetime objects for calculation
            today = datetime.datetime.today().date()
            wake_dt = datetime.datetime.combine(today, wake_time)
            
            # Handle sleep time potentially being past midnight
            sleep_dt = datetime.datetime.combine(today, sleep_time)
            if sleep_time < wake_time:  # If sleep time is before wake time (e.g. 23:00 - 07:00)
                sleep_dt = sleep_dt + datetime.timedelta(days=1)
            
            # Calculate sleep duration in hours
            sleep_duration = (wake_dt - sleep_dt).total_seconds() / 3600
            if sleep_duration < 0:  # If calculation went wrong (negative hours)
                sleep_duration = 24 + sleep_duration
                
            # Analyze sleep duration
            if sleep_duration < 6:
                # Insufficient sleep associated with higher anxiety and stress
                health_insights["anxiety"] += 15
                health_insights["stress"] += 15
                health_insights["general_wellbeing"] -= 10
            elif sleep_duration > 9:
                # Oversleeping can be associated with depression
                health_insights["depression"] += 10
                health_insights["general_wellbeing"] -= 5
            else:
                # Healthy sleep duration
                health_insights["general_wellbeing"] += 10
                health_insights["stress"] -= 5
                
            # Very early wake time might indicate anxiety
            if wake_time.hour < 5:
                health_insights["anxiety"] += 10
                
            # Very late sleep time might indicate stress or anxiety
            if sleep_time.hour > 1:
                health_insights["stress"] += 10
                health_insights["anxiety"] += 5
                
        except Exception as e:
            st.warning(f"Error analyzing sleep patterns: {str(e)}")
    
    # Age-based analysis
    if profile["age"]:
        try:
            age = int(profile["age"])
            
            # Different age groups face different challenges
            if 13 <= age <= 19:
                # Teens often face more stress and anxiety
                health_insights["stress"] += 10
                health_insights["anxiety"] += 5
            elif 20 <= age <= 29:
                # Young adults often face career and relationship stress
                health_insights["stress"] += 5
            elif 30 <= age <= 45:
                # Middle-aged often face work-life balance issues
                health_insights["stress"] += 10
                health_insights["general_wellbeing"] -= 5
            elif age >= 65:
                # Elderly might face more depression
                health_insights["depression"] += 5
                
        except Exception as e:
            st.warning(f"Error analyzing age data: {str(e)}")
    
    # Occupation analysis - simple keyword-based approach
    if profile["occupation"]:
        occupation = profile["occupation"].lower()
        
        # High-stress occupations
        high_stress_jobs = ["doctor", "nurse", "healthcare", "lawyer", "police", "firefighter", 
                          "emergency", "executive", "manager", "teacher", "professor"]
        
        # Isolating occupations
        isolating_jobs = ["remote", "writer", "programmer", "developer", "researcher", 
                        "analyst", "night shift", "truck driver"]
        
        # Check for high stress jobs
        for job in high_stress_jobs:
            if job in occupation:
                health_insights["stress"] += 15
                health_insights["anxiety"] += 10
                break
                
        # Check for potentially isolating jobs
        for job in isolating_jobs:
            if job in occupation:
                health_insights["depression"] += 10
                break
    
    # Current feeling analysis
    if profile["current_feeling"]:
        feeling = profile["current_feeling"].lower()
        
        # Analyze negative emotions
        negative_emotions = ["sad", "depress", "anxious", "worry", "stress", "overwhelm", 
                           "exhaust", "tired", "angry", "frustrat", "lonely", "alone"]
                           
        for emotion in negative_emotions:
            if emotion in feeling:
                # Increase relevant mental health indicators
                if emotion in ["sad", "depress", "lonely", "alone"]:
                    health_insights["depression"] += 20
                elif emotion in ["anxious", "worry"]:
                    health_insights["anxiety"] += 20
                elif emotion in ["stress", "overwhelm", "exhaust", "tired", "angry", "frustrat"]:
                    health_insights["stress"] += 20
                    
                health_insights["general_wellbeing"] -= 15
                break
    
    # Ensure all values are within range
    for key in health_insights:
        health_insights[key] = min(max(health_insights[key], 0), 100)
    
    return health_insights

def start_recording():
    """Set the recording flag to True to start recording"""
    st.session_state.recording = True
    st.session_state.recording_started = True
    st.session_state.countdown = 5  # 5 seconds countdown
    st.session_state.input_mode = "voice"
    st.session_state.recording_id += 1  # Increment recording ID to ensure fresh analysis

# Modified toggle_camera function
def toggle_camera():
    """Toggle the camera on/off"""
    if st.session_state.camera_active:
        # If camera is active, deactivate it
        st.session_state.camera_active = False
        # Clear face analysis data
        st.session_state.face_analysis = None
    else:
        # If camera is inactive, activate it
        st.session_state.camera_active = True
        st.session_state.last_face_update = 0  # Reset face analysis timer

# Handle face analysis if in face mode
if st.session_state.input_mode == "Face + Voice":
    if st.button("Toggle Face Analysis Camera", on_click=toggle_camera):
        pass
    
    if st.session_state.camera_active:
        # Create a placeholder for camera status
        camera_status = st.empty()
        
        # Create a placeholder for the webcam feed
        webcam_placeholder = st.empty()
        
        # Use a file uploader as a fallback if the camera doesn't work
        uploaded_image = st.file_uploader("Or upload a photo for face analysis:", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            # Process the uploaded image instead
            try:
                # Convert uploaded image to OpenCV format
                image_bytes = uploaded_image.getvalue()
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Display the image
                webcam_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Analyze facial expression
                face_analysis = analyze_facial_expression(frame)
                st.session_state.face_analysis = face_analysis
                
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
        else:
            # Try using webcam
            try:
                # Single image capture approach rather than continuous capture
                camera_status.info("Attempting to take a snapshot from webcam...")
                
                # Initialize webcam
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                
                if not cap.isOpened():
                    camera_status.error("Could not access webcam. Please check your camera connection or try uploading an image.")
                else:
                    # Capture a single frame
                    ret, frame = cap.read()
                    
                    # Immediately release the camera
                    cap.release()
                    
                    if ret:
                        # Display the captured frame
                        webcam_placeholder.image(frame, channels="BGR", use_container_width=True)
                        camera_status.success("Snapshot taken!")
                        
                        # Analyze facial expression
                        face_analysis = analyze_facial_expression(frame)
                        st.session_state.face_analysis = face_analysis
                    else:
                        camera_status.warning("Could not capture frame from webcam. Try uploading an image instead.")
            except Exception as e:
                camera_status.error(f"Camera error: {str(e)}. Try uploading an image instead.")
                # Clear any partial camera handles
                try:
                    if 'cap' in locals() and cap is not None:
                        cap.release()
                except:
                    pass

def process_audio():
    """Process the recorded audio file"""
    if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
        try:
            # Load the audio file
            audio_data, sample_rate = librosa.load(st.session_state.audio_file_path, sr=None)
            
            # Extract audio features
            features = extract_audio_features(audio_data, sample_rate)
            st.session_state.audio_features = features
            
            # Predict emotion using the trained model
            emotion_prediction = predict_emotion(audio_data, sample_rate)
            st.session_state.emotion_prediction = emotion_prediction
            
            # Convert to speech
            recognizer = sr.Recognizer()
            with sr.AudioFile(st.session_state.audio_file_path) as source:
                audio = recognizer.record(source)
                
            # Recognize speech
            try:
                text = recognizer.recognize_google(audio)
                st.session_state.temp_input = text
                st.session_state.process_input = True
            except sr.UnknownValueError:
                st.warning("Could not understand audio, please try again")
                st.session_state.temp_input = ""
                st.session_state.process_input = False
            except sr.RequestError:
                st.error("Could not request results; check your network connection")
                st.session_state.temp_input = ""
                st.session_state.process_input = False
            
            # Clean up
            st.session_state.recording = False
            
            # Remove the temporary file
            try:
                os.remove(st.session_state.audio_file_path)
            except:
                pass
            
            return True
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            st.session_state.recording = False
            return False
    else:
        st.error("No audio file found to process.")
        st.session_state.recording = False
        return False

# Handle user profile onboarding first
if not st.session_state.onboarding_complete:
    # Display a welcome message and onboarding progress
    st.markdown("## Welcome to Mental Health Assistant")
    st.markdown("Let's start by collecting some information to better understand your situation.")
    
    # Calculate progress percentage
    progress_questions = ["feeling", "reason", "wake_time", "sleep_time", "occupation", "age"]
    current_index = progress_questions.index(st.session_state.current_question) if st.session_state.current_question in progress_questions else 0
    progress_percentage = int((current_index / len(progress_questions)) * 100)
    
    # Progress bar
    st.progress(progress_percentage)
    
    # Container for the current question
    with st.container():
        if st.session_state.current_question == "feeling":
            st.markdown("### How are you feeling today?")
            feeling = st.text_input("Describe your current mood or emotion (e.g., happy, anxious, stressed, sad):")
            
            if st.button("Next"):
                if feeling.strip():
                    st.session_state.user_profile["current_feeling"] = feeling
                    st.session_state.current_question = "reason"
                else:
                    st.warning("Please describe how you're feeling")
        
        elif st.session_state.current_question == "reason":
            st.markdown("### What's contributing to how you're feeling?")
            reason = st.text_area("(Optional) Is there a specific reason for how you're feeling?")
            
            if st.button("Next"):
                st.session_state.user_profile["feeling_reason"] = reason if reason.strip() else None
                st.session_state.current_question = "wake_time"
        
        elif st.session_state.current_question == "wake_time":
            st.markdown("### What time do you usually wake up?")
            wake_time = st.time_input("Select your typical wake-up time:")
            
            if st.button("Next"):
                st.session_state.user_profile["wake_time"] = wake_time.strftime("%H:%M")
                st.session_state.current_question = "sleep_time"
        
        elif st.session_state.current_question == "sleep_time":
            st.markdown("### What time do you usually go to sleep?")
            sleep_time = st.time_input("Select your typical bedtime:")
            
            if st.button("Next"):
                st.session_state.user_profile["sleep_time"] = sleep_time.strftime("%H:%M")
                st.session_state.current_question = "occupation"
        
        elif st.session_state.current_question == "occupation":
            st.markdown("### What is your occupation or main daily activity?")
            occupation = st.text_input("(Optional) For example: Student, Software Developer, Nurse, etc.")
            
            if st.button("Next"):
                st.session_state.user_profile["occupation"] = occupation if occupation.strip() else None
                st.session_state.current_question = "age"
        
        elif st.session_state.current_question == "age":
            st.markdown("### How old are you?")
            age = st.number_input("Enter your age (optional)", min_value=13, max_value=120, step=1)
            
            if st.button("Finish Setup"):
                st.session_state.user_profile["age"] = str(age) if age else None
                st.session_state.onboarding_complete = True
                st.session_state.current_question = None
                st.rerun()
    
    # Allow skipping the onboarding
    if st.button("Skip Setup"):
        st.session_state.onboarding_complete = True
        st.rerun()
    
    # Don't proceed with the rest of the app until onboarding is complete
    st.stop()

# Main chat interface
with st.sidebar:
    st.markdown("## 💬 Chat Settings")
    
    # Input mode selection
    options = ["Chat", "Voice", "Face + Voice"]
    input_mode = st.radio(
    "Input Mode:",
    options,
    index=next((i for i, opt in enumerate(options) if opt.lower() == st.session_state.input_mode.lower()), 0),
    key="input_mode_selector"
)
    
    # Update the session state
    st.session_state.input_mode = input_mode
    
    # Display user profile card
    st.markdown("## 👤 Your Profile")
    with st.expander("View/Edit Profile"):
        # Current feeling
        current_feeling = st.text_input(
            "How are you feeling?",
            value=st.session_state.user_profile.get("current_feeling", ""),
            key="profile_feeling"
        )
        
        # Feeling reason
        feeling_reason = st.text_area(
            "Reason for feeling this way:",
            value=st.session_state.user_profile.get("feeling_reason", ""),
            key="profile_reason"
        )
        
        # Sleep schedule
        col1, col2 = st.columns(2)
        with col1:
            wake_time = st.text_input(
                "Wake time (HH:MM)",
                value=st.session_state.user_profile.get("wake_time", "07:00"),
                key="profile_wake"
            )
        with col2:
            sleep_time = st.text_input(
                "Sleep time (HH:MM)",
                value=st.session_state.user_profile.get("sleep_time", "23:00"),
                key="profile_sleep"
            )
        
        # Occupation and age
        occupation = st.text_input(
            "Occupation",
            value=st.session_state.user_profile.get("occupation", ""),
            key="profile_occupation"
        )
        
        age = st.text_input(
            "Age",
            value=st.session_state.user_profile.get("age", ""),
            key="profile_age"
        )
        
        if st.button("Update Profile"):
            st.session_state.user_profile = {
                "current_feeling": current_feeling if current_feeling.strip() else None,
                "feeling_reason": feeling_reason if feeling_reason.strip() else None,
                "wake_time": wake_time if wake_time.strip() else None,
                "sleep_time": sleep_time if sleep_time.strip() else None,
                "occupation": occupation if occupation.strip() else None,
                "age": age if age.strip() else None
            }
            st.success("Profile updated successfully!")
    
    # Display analysis results if available
    if st.session_state.analysis_results:
        st.markdown("## 📊 Your Mental Health Indicators")
        
        # Anxiety indicator
        st.metric(
            label="Anxiety Level",
            value=f"{st.session_state.analysis_results.get('anxiety', 0):.0f}%",
            delta=None,
            delta_color="inverse"
        )
        st.progress(st.session_state.analysis_results.get('anxiety', 0) / 100)
        
        # Depression indicator
        st.metric(
            label="Depression Level",
            value=f"{st.session_state.analysis_results.get('depression', 0):.0f}%",
            delta=None,
            delta_color="inverse"
        )
        st.progress(st.session_state.analysis_results.get('depression', 0) / 100)
        
        # Stress indicator
        st.metric(
            label="Stress Level",
            value=f"{st.session_state.analysis_results.get('stress', 0):.0f}%",
            delta=None,
            delta_color="inverse"
        )
        st.progress(st.session_state.analysis_results.get('stress', 0) / 100)
        
        # Wellbeing indicator
        st.metric(
            label="General Wellbeing",
            value=f"{st.session_state.analysis_results.get('general_wellbeing', 75):.0f}%",
            delta=None
        )
        st.progress(st.session_state.analysis_results.get('general_wellbeing', 75) / 100)

# Main chat area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle voice recording if in voice mode
if st.session_state.input_mode in ["Voice", "Face + Voice"]:
    if st.session_state.recording:
        if st.session_state.countdown > 0:
            st.warning(f"Recording will start in {st.session_state.countdown} seconds...")
            time.sleep(1)
            st.session_state.countdown -= 1
            st.rerun()
        else:
            # Start actual recording
            st.warning("Recording in progress... Speak now (5 seconds maximum)")
            
            # Record audio
            sample_rate = 44100
            duration = 5  # seconds
            
            # Record using sounddevice
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                st.session_state.audio_file_path = tmp_file.name
                sf.write(tmp_file.name, audio_data, sample_rate)
            
            # Process the audio
            if process_audio():
                st.rerun()
    
    # Voice recording button
    if not st.session_state.recording:
        if st.button("🎤 Start Voice Recording", key="start_recording", on_click=start_recording):
            pass

# Handle face analysis if in face mode
if st.session_state.input_mode == "Face + Voice":
    if st.button("Toggle Face Analysis Camera", on_click=toggle_camera):
        pass
    
    if st.session_state.camera_active:
        # Create a placeholder for the webcam feed
        webcam_placeholder = st.empty()
        
        try:
            # Initialize webcam with specific settings
            cap = cv2.VideoCapture(0)
            
            # Set camera properties for better detection
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Increase brightness
            
            if not cap.isOpened():
                st.error("Could not access webcam. Please check your camera connection.")
            else:
                # Capture frame
                for _ in range(3):  # Skip a few frames to let camera adjust
                    ret, frame = cap.read()
                    
                ret, frame = cap.read()  # Get the actual frame we'll use
                
                if ret:
                    # Display the raw frame first
                    webcam_placeholder.image(frame, channels="BGR", use_container_width=True)
                    
                    # Check if it's time to update face analysis
                    current_time = time.time()
                    if current_time - st.session_state.last_face_update > 2:
                        # Analyze facial expression
                        face_analysis = analyze_facial_expression(frame)
                        st.session_state.face_analysis = face_analysis
                        
                        # Update timestamp
                        st.session_state.last_face_update = current_time
                else:
                    st.warning("Could not capture frame from webcam")
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
        finally:
            # Make sure to release the camera properly
            if 'cap' in locals() and cap is not None:
                cap.release()
                cv2.destroyAllWindows()  # Clean up OpenCV windows
        
        # Display face analysis results if available
        if st.session_state.face_analysis:
            with st.expander("Facial Expression Analysis"):
                if st.session_state.face_analysis.get('face_detected'):
                    st.markdown("### Detected Emotions:")
                    
                    emotions = st.session_state.face_analysis.get('emotions', {})
                    dominant_emotion = st.session_state.face_analysis.get('dominant_emotion')
                    
                    for emotion, score in emotions.items():
                        # Create a progress bar for each emotion
                        st.markdown(f"**{emotion.capitalize()}**")
                        progress_value = min(max(score / 100, 0), 1)
                        st.progress(progress_value)
                        
                        # Highlight the dominant emotion
                        if emotion == dominant_emotion:
                            st.markdown(f"<span class='dominant-emotion'>Dominant Emotion</span>", 
                                        unsafe_allow_html=True)
                else:
                    st.warning("No face detected in the frame")

# Chat input (either text or processed voice)
if st.session_state.input_mode == "Chat" or (st.session_state.process_input and st.session_state.temp_input):
    if st.session_state.process_input:
        # Use the processed voice input
        user_input = st.session_state.temp_input
        st.session_state.process_input = False
        st.session_state.temp_input = ""
    else:
        # Regular text input
        user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Perform sentiment analysis on user input
        blob = TextBlob(user_input)
        sentiment = blob.sentiment
        st.session_state.sentiment_analysis = {
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity
        }
        
        # Analyze user profile
        profile_analysis = analyze_user_profile()
        
        # Analyze voice if available
        voice_analysis = None
        if st.session_state.audio_features:
            voice_analysis = analyze_voice_emotion(
                st.session_state.audio_features,
                st.session_state.emotion_prediction
            )
        
        # Analyze face if available
        face_analysis = None
        if st.session_state.face_analysis:
            face_analysis = analyze_face_for_mental_health(st.session_state.face_analysis)
        
        # Combine all analysis results
        st.session_state.analysis_results = combine_analysis_results(
            voice_analysis,
            face_analysis,
            profile_analysis
        )
        
        # Prepare context for the AI
        context = f"""
        You are a compassionate mental health assistant. The user has shared the following:
        
        Current feeling: {st.session_state.user_profile.get('current_feeling', 'not specified')}
        Feeling reason: {st.session_state.user_profile.get('feeling_reason', 'not specified')}
        Sleep schedule: {st.session_state.user_profile.get('wake_time', 'not specified')} to {st.session_state.user_profile.get('sleep_time', 'not specified')}
        Occupation: {st.session_state.user_profile.get('occupation', 'not specified')}
        Age: {st.session_state.user_profile.get('age', 'not specified')}
        
        Current message sentiment: 
        - Polarity: {sentiment.polarity:.2f} (-1 = negative, 1 = positive)
        - Subjectivity: {sentiment.subjectivity:.2f} (0 = objective, 1 = subjective)
        
        Mental health indicators:
        - Anxiety: {st.session_state.analysis_results.get('anxiety', 0):.0f}%
        - Depression: {st.session_state.analysis_results.get('depression', 0):.0f}%
        - Stress: {st.session_state.analysis_results.get('stress', 0):.0f}%
        - Wellbeing: {st.session_state.analysis_results.get('general_wellbeing', 75):.0f}%
        
        Voice analysis: {st.session_state.emotion_prediction.get('emotion', 'unknown') if st.session_state.emotion_prediction else 'not available'}
        Face analysis: {st.session_state.face_analysis.get('dominant_emotion', 'not available') if st.session_state.face_analysis else 'not available'}
        
        The user said: "{user_input}"
        """
        
        # Generate response from Gemini
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            
            # Start a new chat or continue existing one
            if "chat" not in st.session_state:
                st.session_state.chat = model.start_chat(history=[])
            
            # Send message to Gemini
            response = st.session_state.chat.send_message(context)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response.text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        # Force a rerun to update the UI
        st.rerun()