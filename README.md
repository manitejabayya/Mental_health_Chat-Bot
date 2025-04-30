# 🧠 AI Mental Health Chatbot

This project is an AI-powered mental health chatbot designed to provide insights into mental health indicators such as anxiety, depression, stress, and general wellbeing. It uses a combination of voice, facial expression, and text-based inputs to analyze and provide feedback to users.

## Features

- **Voice Emotion Analysis**: Extracts audio features and predicts emotions using a trained LSTM model.
- **Facial Expression Analysis**: Analyzes facial expressions to infer emotional states using DeepFace.
- **Text Sentiment Analysis**: Uses TextBlob to analyze the sentiment of user inputs.
- **User Profile Insights**: Provides mental health insights based on user-provided data such as sleep schedule, occupation, and age.
- **Interactive Chat Interface**: Allows users to interact with the chatbot via text, voice, or a combination of both.
- **Real-Time Feedback**: Displays mental health indicators such as anxiety, depression, stress, and general wellbeing.

## Project Structure

```
├── data/                   # Dataset for training and testing models
├── models/                 # Pre-trained models for emotion and sentiment analysis
├── src/                    # Source code for the chatbot
│   ├── audio_analysis.py   # Voice emotion analysis module
│   ├── facial_analysis.py  # Facial expression analysis module
│   ├── text_analysis.py    # Text sentiment analysis module
│   ├── chatbot_interface.py # Chat interface logic
│   └── utils.py            # Utility functions
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── LICENSE                 # License information
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/ai-mental-health-chatbot.git
    cd ai-mental-health-chatbot
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained models and place them in the `models/` directory.

## Usage

1. Run the chatbot interface:
    ```bash
    python src/chatbot_interface.py
    ```

2. Interact with the chatbot using text, voice, or facial expressions.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes and push to your fork:
    ```bash
    git commit -m "Add feature-name"
    git push origin feature-name
    ```
4. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This chatbot is not a substitute for professional mental health care. If you are experiencing a mental health crisis, please seek help from a qualified professional or contact emergency services in your area.
