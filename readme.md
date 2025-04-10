# 🧠 Voice-Based Cognitive Screening with Isolation Forest

## 🔍 Why Isolation Forest?

We selected **Isolation Forest** for the following reasons:

- ✅ **Unsupervised**: No need for labeled "healthy" vs "impaired" voice samples.
- ✅ **Interpretable**: Works by isolating outliers through random partitioning—easy to explain clinically.
- ✅ **Efficient for small datasets**: Works well even with 5–10 samples, making it ideal for early-stage prototyping.
- ✅ **Multivariate**: Handles a mix of audio + linguistic features (e.g., speech rate, hesitation, pitch) robustly.

---

## ⚙️ What the Code Does (Step-by-Step)

### 1. 🎙️ Audio Preprocessing + Feature Extraction
- Loads `.wav` voice clips using `librosa`
- Extracts key **acoustic features**:
  - `speech_rate`: words/sec
  - `avg_silence`: average pause duration
  - `pitch_std`: voice pitch variability
- Transcribes speech using **OpenAI Whisper**
- Uses regex to detect **hesitations** ("uh", "um", etc.)
- Matches transcripts against `.meta` files to evaluate **cognitive tasks**:
  - `recall`, `naming`, or `completion` tasks using keyword match

### 2. 🧠 Unsupervised Modeling with Isolation Forest
- Features are standardized using `StandardScaler`
- `IsolationForest` is trained with `contamination=0.2` to flag outliers
- Risk scores (`anomaly_score`) and flags (`risk_flag`) are generated:
  - `-1`: Anomalous (at-risk)
  - `1`: Normal

### 3. 📤 Output + Export
- Features and predictions are stored in a `pandas.DataFrame`
- Saved to `voice_analysis_results.csv` for analysis

---

## 📊 Example Output

| file         | speech_rate | avg_silence | pitch_std | hesitations | task_score | anomaly_score | risk_flag |
|--------------|-------------|-------------|-----------|-------------|------------|----------------|-----------|
| sample_3.wav | 2.96        | 0.35        | 898.1     | 2           | 0.33       | -0.042         | -1        |
| sample_7.wav | 3.21        | 0.00        | 1015.5    | 0           | 1.00       | 0.19           | 1         |

---

## 🧠 Clinical Relevance

- Captures subtle vocal cues linked to cognitive stress or decline
- Models simulated clinical tasks like object naming, memory recall
- Prepares system for future use with real clinical or longitudinal data
- Simple to integrate into REST APIs or real-time tools

---

> Built with: Python, Whisper, Librosa, Scikit-learn, NLTK, Isolation Forest
