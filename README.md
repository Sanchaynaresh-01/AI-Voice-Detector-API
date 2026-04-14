# 🎙️ AI-Generated Voice Detection (Multi-Language)
>> Note : This project has issues since the model has gone overfitting, so will be updated in some days
🚀 **India AI Impact Buildathon – GUVI x HCL**

---

## 📌 Problem Statement

Build an **API-based system** that determines whether a given voice sample is:

- 🎤 **Human-generated**
- 🤖 **AI-generated**

The system must:

- Accept **Base64-encoded MP3 audio**
- Support **multiple languages**:
  - 🇮🇳 Hindi  
  - 🇮🇳 Tamil  
  - 🇮🇳 Telugu  
  - 🇮🇳 Malayalam  
  - 🌍 English  
- Return a structured JSON response with:
  - Classification
  - Confidence score
  - Explanation

---

## 🧠 Solution Overview

This project uses a **hybrid AI + ML approach**:

### 🔍 Feature Extraction
- Uses **Wav2Vec2 (Facebook AI)** for deep speech representations
- Extracts **MFCC features** for classical audio characteristics

### 🧮 Classification
- Lightweight **Logistic Regression classifier**
- Fast, efficient, and suitable for real-time APIs

---

## ⚙️ System Architecture

Audio Input (Base64 MP3)
↓
Decode & Convert to WAV (16kHz)
↓
Feature Extraction
├── Wav2Vec2 Embeddings
└── MFCC Features
↓
Feature Fusion
↓
Classifier (Logistic Regression)
↓
Prediction Output (JSON)

---

## 📊 Model Performance

| Metric        | Score |
|--------------|------|
| Accuracy     | 100% |
| Precision    | 1.00 |
| Recall       | 1.00 |

⚠️ *Note:* High accuracy is achieved on a controlled dataset. Real-world performance may vary with more diverse inputs.

---

## 📁 Project Structure
AI Voice/
│
├── data/
│ ├── human/
│ └── ai/
│
├── extract_features.py
├── train.py
├── model.pkl
│
├── api.py (FastAPI server)
└── README.md


---

## Setup and Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Sanchaynaresh-01/AI-Voice-Detector-API
cd AI Voice
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers librosa soundfile scikit-learn fastapi uvicorn
```

### Step 4: Prepare Dataset

Organize your dataset in the following structure:

- Place real human voice files in `data/human/`
- Place AI-generated voice files in `data/ai/`

**Dataset Requirements:**
- File format: WAV
- Sample rate: 16kHz
- Channel: Mono
- Minimum duration: 3 seconds

### Step 5: Train the Model
```bash
python train.py
```

This will:
- Extract features from all audio files in the dataset
- Train the Logistic Regression classifier
- Save the trained model as `model.pkl`

### Step 6: Run API Server
```bash
uvicorn api:app --reload
```

The API server will start and be accessible at `http://localhost:8000`

### Step 7: Test the API

**Endpoint:**
POST /predict
**Request Body:**
```json
{
  "audio_base64": "<BASE64_MP3_AUDIO>"
}
```

**Response:**
```json
{
  "prediction": "AI or Human",
  "confidence": 0.85,
  "explanation": "Short reasoning"
}
```

## API Usage

### Making a Prediction Request

1. Encode your MP3 audio file to Base64
2. Send a POST request to `/predict` endpoint
3. Include the Base64-encoded audio in the request body
4. Receive the prediction result with confidence score and explanation

### Example Request
```python
import requests
import base64

# Read and encode audio file
with open("sample.mp3", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

# Make API request
response = requests.post(
    "http://localhost:8000/predict",
    json={"audio_base64": audio_base64}
)

# Get result
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Explanation: {result['explanation']}")
```

## Key Features

- **Multi-Language Support**: Handles voice samples in Tamil, English, Hindi, Malayalam, and Telugu
- **Dual Feature Extraction**: Combines Wav2Vec2 embeddings with MFCC features for robust representation
- **GPU Acceleration**: Uses GPU for feature extraction to improve processing speed
- **Lightweight Classification**: Logistic Regression ensures fast inference
- **RESTful API**: Easy integration with FastAPI-based REST interface
- **Flexible Input Format**: Accepts Base64-encoded MP3 files
- **Structured Output**: Returns prediction, confidence score, and explanation in JSON format
- **Trained on Diverse Data**: Uses both human and AI-generated voice datasets

## Conclusion

This AI-Generated Voice Detection system provides a reliable and efficient solution for identifying AI-generated voice samples across multiple Indian languages. By combining modern deep learning feature extraction with traditional signal processing techniques and a lightweight classifier, the system achieves a balance between accuracy and speed, making it suitable for real-time applications and API-based deployments.
