# 🎙️ AI-Generated Voice Detection (Multi-Language)

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
