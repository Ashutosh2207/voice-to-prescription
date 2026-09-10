# Voice-to-Prescription — Medhant Lite

AI-powered voice-to-prescription pipeline for doctors. Dictate prescriptions on mobile, get structured JSON.

## Features

- **Speech-to-Text**: Local Whisper (tiny/base/small/medium/large) — no cloud API needed
- **Entity Extraction**: 
  - spaCy NER + rule-based matching (free, local)
  - Optional: Ollama LLM (mistral-3b/ministral-3:3b) for higher accuracy
- **Normalization**: Medicine names, dosages, frequencies, durations, routes, instructions
- **Validation**: Field-level checks, confidence scores, missing field detection
- **REST API**: FastAPI with CORS for mobile integration
- **Frontend**: Streamlit demo UI for testing

## Pipeline

```
Voice → Audio Preprocessing → Whisper STT → Text Normalization 
  → NLP Extraction (spaCy/Ollama) → JSON Mapping → Validation 
  → Doctor Review → Final Prescription
```

## Quick Start

### 1. Install Dependencies

```bash
# Using the runner script
python run.py install

# Or manually
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. (Optional) Install Ollama for LLM Extraction

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull ministral-3:3b

# Start Ollama server
ollama serve
```

### 3. Run the System

```bash
# Run everything (API + Streamlit)
python run.py all

# Or run separately:
# Terminal 1: API server
python run.py api

# Terminal 2: Streamlit UI
python run.py streamlit
```

### 4. Access

- **API**: http://localhost:8000 (docs at `/docs`)
- **Streamlit UI**: http://localhost:8501

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process-audio` | POST | Upload audio file (multipart) |
| `/process-base64` | POST | Base64 encoded audio |
| `/process-text` | POST | Direct text input (bypass STT) |
| `/configure` | POST | Change pipeline settings |
| `/schema` | GET | Get JSON schema |
| `/health` | GET | Health check |

### Example: Process Audio

```bash
curl -X POST "http://localhost:8000/process-audio" \
  -F "file=@prescription.wav" \
  -F "language=en"
```

### Example: Process Text

```bash
curl -X POST "http://localhost:8000/process-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Paracetamol 500 mg 1-0-1 for 5 days after food"}'
```

## Output Format

```json
{
  "success": true,
  "data": {
    "prescription": {
      "medicines": [
        {
          "medicineName": "Paracetamol",
          "dosage": "500 mg",
          "frequency": "1-0-1",
          "duration": "5 days",
          "route": "oral",
          "instructions": "After food",
          "confidence": 0.9
        }
      ],
      "notes": null,
      "rawTranscript": "Paracetamol 500 mg 1-0-1 for 5 days after food"
    },
    "validation": {
      "isValid": true,
      "issues": [],
      "missingFields": []
    },
    "processingTimeMs": 1250,
    "sttConfidence": 0.95
  }
}
```

## Frequency Codes

| Code | Meaning |
|------|---------|
| `1-0-0` | Once daily (OD) |
| `1-0-1` | Twice daily (BD) |
| `1-1-1` | Thrice daily (TID) |
| `1-1-1-1` | Four times daily (QID) |
| `SOS` | As needed (PRN) |

## Routes

`oral`, `topical`, `injection`, `inhalation`, `sublingual`, `rectal`, `ophthalmic`, `otic`, `nasal`, `transdermal`, `unknown`

## Configuration

Configure via API `/configure` or environment variables:

```python
PipelineConfig(
    whisper_model="base",           # tiny, base, small, medium, large
    use_ollama_extraction=False,    # Use Ollama LLM
    ollama_config=OllamaConfig(
        base_url="http://localhost:11434",
        model="ministral-3:3b"
    ),
    use_spacy_fallback=True,        # Fallback to spaCy
    spacy_model="en_core_web_sm",
    language="en"
)
```

## Project Structure

```
├── app/
│   ├── api.py              # FastAPI endpoints
│   ├── models/
│   │   └── schemas.py      # Pydantic models
│   ├── pipeline/
│   │   ├── stt.py          # Whisper STT
│   │   ├── normalizer.py   # Text normalization
│   │   ├── nlp_extractor.py # spaCy extraction
│   │   ├── ollama_extractor.py # Ollama LLM extraction
│   │   ├── validator.py    # Validation
│   │   └── pipeline.py     # Main orchestrator
│   └── utils/
│       └── audio_utils.py  # Audio helpers
├── data/samples/           # Sample prescriptions
├── streamlit_app.py        # Streamlit frontend
├── run.py                  # Entry point
└── requirements.txt
```

## Validation Rules

- **Required**: medicineName
- **Warnings**: Missing dosage, frequency, duration, route
- **Errors**: No medicines, duplicate medicines, unknown medicines
- **Info**: Non-standard frequency format, unclear duration

## Mobile Integration

The API is designed for mobile apps:
- CORS enabled for all origins
- Accepts base64 audio for easy mobile upload
- Returns structured JSON ready for Medhant Lite
- Lightweight response with confidence scores

## Accuracy Tips

1. Use `base` or `small` Whisper model for balance
2. Enable Ollama with `ministral-3:3b` for best extraction
3. Speak clearly: "MedicineName Dosage Frequency Duration Route Instructions"
4. Pause slightly between multiple medicines

## License

Internal use — Medhant Lite project.
