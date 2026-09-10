# Voice-to-Prescription — Medhant Lite

AI-powered Voice-to-Prescription system that converts doctor-dictated prescription audio into a structured, validated prescription JSON, followed by doctor review and approval.

## Overview

Medhant Lite is a local-first clinical documentation pipeline designed to support prescription drafting from voice input.

```text
Voice / Audio
     ↓
Audio Preprocessing
     ↓
Faster-Whisper STT
     ↓
Raw Transcript
     ↓
Medical Entity Extraction
     ↓
Normalization
     ↓
Medical Terminology Validation
     ↓
Structured Prescription JSON
     ↓
Doctor Review / Edit
     ↓
Doctor Approval
```

> The AI-generated prescription is treated as a draft. Finalization requires doctor review and approval.

## Key Features

### Speech-to-Text

- Faster-Whisper for local speech recognition
- Current configured model: `base`
- CPU inference with `int8` compute
- Audio preprocessing and resampling
- Voice activity detection
- Medical-domain prompting
- STT confidence reporting

### Medical Entity Extraction

The pipeline extracts prescription information such as:

- Medicine name
- Dosage / strength
- Dosage form
- Frequency
- Duration
- Route
- Instructions / remarks
- Unit per time

The current local LLM configuration uses **Ollama + `ministral-3:3b`**. spaCy is available as a local fallback/NER component.

### Normalization

Prescription values are normalized into the application's structured format, including common frequency representations such as:

| Code | Meaning |
|---|---|
| `1-0-0` | Once daily |
| `1-0-1` | Twice daily |
| `1-1-1` | Thrice daily |
| `1-1-1-1` | Four times daily |
| `SOS` | As needed |

### Medical Validation

- Medical terminology database
- Medicine-name normalization
- Fuzzy matching
- Transcript evidence checks
- Dosage / form / frequency evidence checks
- Missing-field detection
- Duplicate detection
- Unsupported-medicine / hallucination checks
- Validation issues with severity levels

### Doctor Review and Approval

```text
AI Draft
   ↓
Doctor Review / Edit
   ↓
Doctor Changes Saved
   ↓
Doctor Approval
   ↓
Final Approved Prescription
```

### Evaluation Metrics

The project includes an evaluation module for comparing system predictions against doctor-verified reference data.

- **WER (Word Error Rate)** for transcription quality
- **Precision** for extraction correctness
- **Recall** for extraction completeness
- **F1 Score** for the precision/recall balance
- **Hallucination Rate** for unsupported medicine predictions
- Field-level Precision / Recall / F1
- True Positive / False Positive / False Negative counts
- Edit distance details

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Speech-to-Text | Faster-Whisper |
| STT Model | `base` |
| Local LLM | Ollama |
| LLM Model | `ministral-3:3b` |
| NLP / Fallback | spaCy |
| API | FastAPI |
| ASGI Server | Uvicorn |
| Validation | Python + Medical Terminology DB + Fuzzy Matching |
| Data Models | Pydantic |
| Frontend | Streamlit |
| Audio Processing | Pydub / NumPy / Librosa |
| Database | SQLite |

## Project Architecture

```text
                    ┌──────────────────────┐
                    │ Doctor Voice / Text  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Audio Preprocessing  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Faster-Whisper Base  │
                    │      CPU / int8      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Raw Transcript       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Ministral 3:3b       │
                    │ via local Ollama     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Normalization        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Medical Validation  │
                    │ + Evidence Checks   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Structured JSON      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Doctor Review        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Doctor Approval      │
                    └──────────────────────┘
```

## Project Structure

```text
voice-to-prescription/
│
├── app/
│   ├── api.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   └── pipeline/
│       ├── pipeline.py
│       ├── stt.py
│       ├── ollama_extractor.py
│       ├── validator.py
│       └── medical_terminology.py
│
├── data/
│   └── ...
│
├── evaluation_dataset/
│   ├── case_001.json
│   ├── case_002.json
│   └── ...
│
├── tests/
│   └── ...
│
├── streamlit_app.py
├── run.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── medical_terminology.db
├── .env.example
├── .gitignore
└── README.md
```

> Evaluation audio files such as `.mp3` and `.wav` are excluded by `.gitignore` and are therefore intended to remain local. Do not commit private patient or identifiable clinical recordings to a public repository.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ashutosh2207/voice-to-prescription.git
cd voice-to-prescription
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

### 4. Install the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. Install and start Ollama

Install Ollama for your operating system, then pull the configured model:

```bash
ollama pull ministral-3:3b
```

Start the local Ollama service:

```bash
ollama serve
```

The application expects Ollama at:

```text
http://localhost:11434
```

## Run the Project

### API server

```bash
python run.py api
```

Or:

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

### Streamlit UI

In a second terminal:

```bash
python run.py streamlit
```

Or:

```bash
streamlit run streamlit_app.py
```

### Run API + Streamlit

```bash
python run.py all
```

## Access the Application

FastAPI:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Streamlit:

```text
http://localhost:8501
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/process-audio` | Process uploaded audio |
| `POST` | `/process-audio/transformed` | Process audio and return transformed prescription |
| `POST` | `/process-base64` | Process Base64 audio |
| `POST` | `/process-base64/transformed` | Process Base64 audio and return transformed prescription |
| `POST` | `/process-text` | Process direct text input |
| `POST` | `/process-text/transformed` | Process text and return transformed prescription |
| `POST` | `/configure` | Configure pipeline |
| `GET` | `/schema` | Retrieve schema information |
| `POST` | `/evaluate` | Evaluate a prediction against ground truth |

## Example: Process Audio

```bash
curl -X POST "http://localhost:8000/process-audio" \
  -F "file=@prescription.mp3" \
  -F "language=en"
```

## Example: Process Text

```bash
curl -X POST "http://localhost:8000/process-text" \
  -H "Content-Type: application/json" \
  -d '{"text":"Paracetamol 500 mg 1-0-1 for 5 days after food"}'
```

## Output Format

The transformed output uses the Medhant Lite prescription representation.

```json
{
  "success": true,
  "rawTranscript": "Paracetamol 500 mg tablet morning and evening after food for 5 days.",
  "data": {
    "prescription": [
      {
        "name": "PARACETAMOL",
        "form": "TABLET",
        "dosage": "500MG",
        "unitPerTime": "1",
        "numberOfTime": "1-0-1",
        "remarks": "AFTER FOOD",
        "timesPerDay": "5 DAYS"
      }
    ]
  },
  "validation": {
    "isValid": true,
    "issues": [],
    "missingFields": []
  },
  "processingTimeMs": 1250,
  "sttConfidence": 0.95
}
```

> Only information supported by the input should be extracted. Missing clinical information should not be invented.

## Supported Routes

```text
oral
topical
injection
inhalation
sublingual
rectal
ophthalmic
otic
nasal
transdermal
unknown
```

## Evaluation Framework

The evaluation framework is designed to compare system predictions with **doctor-verified reference data**.

### Dataset structure

```text
evaluation_dataset/
├── case_001.mp3     # local only; ignored by Git
├── case_001.json    # verified reference data
├── case_002.mp3
├── case_002.json
└── ...
```

Each JSON file contains the verified transcript and prescription for the matching audio case:

```json
{
  "referenceTranscript": "Verified transcript for this recording.",
  "referencePrescription": {
    "prescription": [
      {
        "name": "ASPIRIN",
        "form": "TABLET",
        "dosage": "100MG",
        "numberOfTime": "1-0-1",
        "remarks": "AFTER FOOD",
        "timesPerDay": "5 DAYS",
        "unitPerTime": "1"
      }
    ]
  }
}
```

### Evaluation flow

```text
Evaluation Audio
      ↓
System Prediction
      ↓
Predicted Transcript + Prescription
      ↓
Doctor-Verified Ground Truth
      ↓
Automatic Comparison
      ↓
WER / Precision / Recall / F1
      ↓
Hallucination Analysis
```

### Metric definitions

**WER**

```text
WER = (Substitutions + Deletions + Insertions) / Reference Words
```

Lower WER is better.

**Precision**

The proportion of predicted medicines/fields that are correct.

**Recall**

The proportion of ground-truth medicines/fields successfully extracted.

**F1 Score**

The harmonic mean of precision and recall.

**Hallucination Rate**

The rate of unsupported medicines predicted by the system when compared with the supplied reference.

## Configuration

The current tested configuration is centered on local inference:

```python
PipelineConfig(
    whisper_model="base",
    use_ollama_extraction=False,
    ollama_base_url="http://localhost:11434",
    ollama_model="ministral-3:3b",
    use_spacy_fallback=True,
    spacy_model="en_core_web_sm",
    language="en"
)
```

## Doctor Review Workflow

The application separates AI generation from final clinical approval:

```text
Voice Input
   ↓
AI Draft
   ↓
Validation
   ↓
Doctor Review / Edit
   ↓
Doctor Approval
   ↓
Final Prescription
```

This human-in-the-loop approach is intended to prevent the AI draft from being treated as a finalized clinical order without review.

## Development / Troubleshooting

Check API health:

```bash
curl http://localhost:8000/health
```

Python syntax check:

```bash
python -m py_compile streamlit_app.py
```

Check Git changes:

```bash
git status
```

## Security and Privacy

This repository is public. Do not commit:

- API keys or credentials
- `.env` files
- Private patient information
- Identifiable clinical recordings
- Other confidential healthcare data

The repository `.gitignore` excludes virtual environments, environment files, model weights, audio recordings and logs.

## Limitations

- STT quality depends on audio quality, accent, pronunciation, noise and microphone conditions.
- Local CPU inference may be slower than GPU inference.
- Medical terminology coverage depends on the maintained terminology database.
- Evaluation quality depends on the correctness of human-verified reference data.
- The system should not replace professional medical judgment.

## Future Improvements

- Larger multi-speaker evaluation datasets
- Batch evaluation and aggregate benchmark reports
- Expanded medical terminology coverage
- GPU-accelerated inference
- Authentication and role-based access control
- Audit logging and versioned prescription history
- Persistent production database
- Hospital/mobile integration
- Expanded multilingual support

## License

Internal use — Medhant Lite project.

## Repository

Public GitHub repository:

https://github.com/Ashutosh2207/voice-to-prescription
