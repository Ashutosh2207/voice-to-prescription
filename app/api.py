from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import tempfile
import os
import asyncio
from datetime import datetime, timezone

from app.models.schemas import (
    PrescriptionResponse,
    AudioProcessRequest,
    Prescription,
    PrescriptionStatus,
    DoctorReviewRequest,
    DoctorApprovalRequest,
)

from app.pipeline import (
    get_pipeline,
    create_pipeline,
    PipelineConfig,
    OllamaConfig,
)

from app.evaluation.metrics import (
    evaluate_pipeline,
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Voice-to-Prescription API",
    description=(
        "AI-powered voice to prescription conversion "
        "with clinical review and approval workflow "
        "for Medhant Lite"
    ),
    version="1.1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class TextProcessRequest(BaseModel):
    text: str
    language: str = "auto"


class PipelineConfigRequest(BaseModel):
    whisper_model: str = "base"
    use_ollama_extraction: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "ministral-3:3b"
    use_spacy_fallback: bool = True
    spacy_model: str = "en_core_web_sm"
    language: str = "auto"


class EvaluationRequest(BaseModel):
    referenceTranscript: str
    predictedTranscript: str
    referencePrescription: Dict[str, Any]
    predictedPrescription: Dict[str, Any]


class PrescriptionReviewResponse(BaseModel):
    success: bool
    prescription: Optional[Prescription] = None
    status: Optional[PrescriptionStatus] = None
    doctorId: Optional[str] = None
    editReason: Optional[str] = None
    approvalNote: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None


# =========================================================
# GLOBAL PIPELINE
# =========================================================

pipeline = get_pipeline()


# =========================================================
# PRESCRIPTION REVIEW STORE
# =========================================================
#
# Development implementation.
#
# Production:
# Replace this in-memory store with a persistent,
# authenticated database-backed prescription store.
#
# =========================================================

reviewed_prescriptions: Dict[str, Dict[str, Any]] = {}


# =========================================================
# HELPER
# =========================================================

async def run_in_thread(func, *args):
    """
    Run synchronous pipeline functions outside
    FastAPI's active event loop.
    """

    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        None,
        lambda: func(*args)
    )


def get_raw_transcript(result) -> str:
    """
    Safely get Whisper raw transcript
    from pipeline result.
    """

    try:

        if result is None:
            return ""

        prescription = getattr(
            result,
            "prescription",
            None
        )

        if prescription is None:
            return ""

        transcript = getattr(
            prescription,
            "rawTranscript",
            ""
        )

        if transcript is None:
            return ""

        return str(
            transcript
        ).strip()

    except Exception:
        return ""


def get_validation(result):
    """
    Safely convert validation object to dictionary.
    """

    try:

        validation = getattr(
            result,
            "validation",
            None
        )

        if validation is None:
            return None

        if hasattr(
            validation,
            "model_dump"
        ):
            return validation.model_dump()

        if hasattr(
            validation,
            "dict"
        ):
            return validation.dict()

        if isinstance(
            validation,
            dict
        ):
            return validation

        return None

    except Exception:
        return None


def get_current_timestamp() -> str:
    """
    Return UTC timestamp in ISO-8601 format.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "service": "Voice-to-Prescription API",
        "version": "1.1.0",
        "status": "running",
        "workflow": (
            "AI Draft -> Doctor Review/Edit -> Doctor Approval"
        ),
        "endpoints": [
            "/process-audio",
            "/process-base64",
            "/process-text",
            "/process-audio/transformed",
            "/process-base64/transformed",
            "/process-text/transformed",
            "/configure",
            "/schema",
            "/health",
            "/evaluate",
            "/prescription/review/{prescription_id}",
            "/prescription/approve/{prescription_id}",
            "/prescription/{prescription_id}"
        ]
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "pipeline": "loaded"
    }


# =========================================================
# PROCESS UPLOADED AUDIO
# =========================================================

@app.post(
    "/process-audio",
    response_model=PrescriptionResponse
)
async def process_audio(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_transcript: bool = Form(True)
):

    tmp_path = None

    try:

        suffix = os.path.splitext(
            file.filename or ".wav"
        )[1]

        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            content = await file.read()

            tmp.write(content)

            tmp_path = tmp.name

        result = await run_in_thread(
            pipeline.process_audio_file,
            tmp_path,
            language
        )

        if not return_transcript:

            result.prescription.rawTranscript = None

        # New workflow status
        try:
            result.status = (
                PrescriptionStatus.AI_DRAFT
            )
        except Exception:
            pass

        return PrescriptionResponse(
            success=True,
            data=result
        )

    except Exception as e:

        return PrescriptionResponse(
            success=False,
            error=str(e)
        )

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            try:
                os.unlink(tmp_path)

            except Exception:
                pass


# =========================================================
# PROCESS BASE64 AUDIO
# =========================================================

@app.post(
    "/process-base64",
    response_model=PrescriptionResponse
)
async def process_base64(
    request: AudioProcessRequest
):

    try:

        if not request.audioBase64:

            raise HTTPException(
                status_code=400,
                detail="audioBase64 is required"
            )

        result = await run_in_thread(
            pipeline.process_base64_audio,
            request.audioBase64,
            request.language
        )

        if not request.returnTranscript:

            result.prescription.rawTranscript = None

        # New workflow status
        try:
            result.status = (
                PrescriptionStatus.AI_DRAFT
            )
        except Exception:
            pass

        return PrescriptionResponse(
            success=True,
            data=result
        )

    except HTTPException:
        raise

    except Exception as e:

        return PrescriptionResponse(
            success=False,
            error=str(e)
        )


# =========================================================
# PROCESS TEXT
# =========================================================

@app.post(
    "/process-text",
    response_model=PrescriptionResponse
)
async def process_text(
    request: TextProcessRequest
):

    try:

        result = await run_in_thread(
            pipeline.process_text,
            request.text
        )

        # New workflow status
        try:
            result.status = (
                PrescriptionStatus.AI_DRAFT
            )
        except Exception:
            pass

        return PrescriptionResponse(
            success=True,
            data=result
        )

    except Exception as e:

        return PrescriptionResponse(
            success=False,
            error=str(e)
        )


# =========================================================
# CONFIGURE PIPELINE
# =========================================================

@app.post(
    "/configure",
    response_model=PrescriptionResponse
)
async def configure_pipeline(
    config: PipelineConfigRequest
):

    global pipeline

    try:

        ollama_config = None

        if config.use_ollama_extraction:

            ollama_config = OllamaConfig(
                base_url=config.ollama_base_url,
                model=config.ollama_model
            )

        new_pipeline = await run_in_thread(
            create_pipeline,
            PipelineConfig(
                whisper_model=config.whisper_model,
                use_ollama_extraction=(
                    config.use_ollama_extraction
                ),
                ollama_config=ollama_config,
                use_spacy_fallback=(
                    config.use_spacy_fallback
                ),
                spacy_model=config.spacy_model,
                language=config.language
            )
        )

        pipeline = new_pipeline

        return PrescriptionResponse(
            success=True,
            data=None
        )

    except Exception as e:

        return PrescriptionResponse(
            success=False,
            error=str(e)
        )


# =========================================================
# SCHEMA
# =========================================================

@app.get("/schema")
async def get_schema():

    return {

        "prescriptionStatus": {
            "values": [
                "AI_DRAFT",
                "DOCTOR_EDITED",
                "APPROVED"
            ]
        },

        "prescription": {

            "medicines": [

                {
                    "medicineName": (
                        "string (required)"
                    ),

                    "dosage": (
                        "string "
                        "(e.g., '500 mg')"
                    ),

                    "frequency": (
                        "string "
                        "(e.g., '1-0-1', 'SOS')"
                    ),

                    "duration": (
                        "string "
                        "(e.g., '5 days')"
                    ),

                    "route": (
                        "enum: oral|topical|"
                        "injection|inhalation|"
                        "sublingual|rectal|"
                        "ophthalmic|otic|nasal|"
                        "transdermal|unknown"
                    ),

                    "instructions": (
                        "string "
                        "(e.g., 'After food')"
                    ),

                    "confidence": (
                        "float 0-1"
                    )
                }

            ],

            "notes": "string",

            "rawTranscript": "string"
        },

        "validation": {

            "isValid": "boolean",

            "issues": [

                {
                    "field": "string",

                    "medicineIndex": "int",

                    "issue": "string",

                    "severity": (
                        "error|warning|info"
                    )
                }

            ],

            "missingFields": [
                "string"
            ]
        }
    }


# =========================================================
# TRANSFORMED - UPLOADED AUDIO
# =========================================================

@app.post(
    "/process-audio/transformed"
)
async def process_audio_transformed(
    file: UploadFile = File(...),
    language: str = Form("auto"),
):

    tmp_path = None

    try:

        suffix = os.path.splitext(
            file.filename or ".wav"
        )[1]

        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            content = await file.read()

            tmp.write(content)

            tmp_path = tmp.name

        # -------------------------------------------------
        # Audio -> Whisper -> Extraction
        # -------------------------------------------------

        result = await run_in_thread(
            pipeline.process_audio_file,
            tmp_path,
            language
        )

        # -------------------------------------------------
        # Raw Whisper transcript
        # -------------------------------------------------

        raw_transcript = (
            get_raw_transcript(
                result
            )
        )

        # -------------------------------------------------
        # Transform to Medhant Lite format
        # -------------------------------------------------

        transformed = (
            pipeline.transform_prescription(
                result.prescription
            )
        )

        return {

            "success": True,

            "status": "AI_DRAFT",

            "rawTranscript": raw_transcript,

            "data": transformed,

            "validation": (
                get_validation(
                    result
                )
            ),

            "processingTimeMs": getattr(
                result,
                "processingTimeMs",
                None
            ),

            "sttConfidence": getattr(
                result,
                "sttConfidence",
                None
            )
        }

    except Exception as e:

        import traceback

        print(
            "\n========== PROCESSING ERROR =========="
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        print(
            "======================================\n"
        )

        return {

            "success": False,

            "error": (
                f"{type(e).__name__}: {e}"
            ),

            "rawTranscript": ""
        }

    finally:

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            try:
                os.unlink(
                    tmp_path
                )

            except Exception:
                pass


# =========================================================
# TRANSFORMED - BASE64 / MICROPHONE
# =========================================================

@app.post(
    "/process-base64/transformed"
)
async def process_base64_transformed(
    request: AudioProcessRequest
):

    try:

        if not request.audioBase64:

            raise HTTPException(
                status_code=400,
                detail="audioBase64 is required"
            )

        # -------------------------------------------------
        # Mic -> Base64 -> Whisper -> Extraction
        # -------------------------------------------------

        result = await run_in_thread(
            pipeline.process_base64_audio,
            request.audioBase64,
            request.language
        )

        # -------------------------------------------------
        # Raw Whisper transcript
        # -------------------------------------------------

        raw_transcript = (
            get_raw_transcript(
                result
            )
        )

        # -------------------------------------------------
        # Transform
        # -------------------------------------------------

        transformed = (
            pipeline.transform_prescription(
                result.prescription
            )
        )

        return {

            "success": True,

            "status": "AI_DRAFT",

            "rawTranscript": raw_transcript,

            "data": transformed,

            "validation": (
                get_validation(
                    result
                )
            ),

            "processingTimeMs": getattr(
                result,
                "processingTimeMs",
                None
            ),

            "sttConfidence": getattr(
                result,
                "sttConfidence",
                None
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        import traceback

        print(
            "\n========== PROCESSING ERROR =========="
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        print(
            "======================================\n"
        )

        return {

            "success": False,

            "error": (
                f"{type(e).__name__}: {e}"
            ),

            "rawTranscript": ""
        }


# =========================================================
# TRANSFORMED - TEXT
# =========================================================

@app.post(
    "/process-text/transformed"
)
async def process_text_transformed(
    request: TextProcessRequest
):

    try:

        result = await run_in_thread(
            pipeline.process_text,
            request.text
        )

        transformed = (
            pipeline.transform_prescription(
                result.prescription
            )
        )

        return {

            "success": True,

            "status": "AI_DRAFT",

            "rawTranscript": request.text,

            "data": transformed,

            "validation": (
                get_validation(
                    result
                )
            ),

            "processingTimeMs": getattr(
                result,
                "processingTimeMs",
                None
            ),

            "sttConfidence": None
        }

    except Exception as e:

        import traceback

        print(
            "\n========== TEXT PROCESSING ERROR =========="
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        print(
            "===========================================\n"
        )

        return {

            "success": False,

            "error": (
                f"{type(e).__name__}: {e}"
            ),

            "rawTranscript": ""
        }


# =========================================================
# DOCTOR REVIEW / EDIT PRESCRIPTION
# =========================================================

@app.post(
    "/prescription/review/{prescription_id}",
    response_model=PrescriptionReviewResponse
)
async def review_prescription(
    prescription_id: str,
    request: DoctorReviewRequest
):
    """
    Save a doctor-reviewed/edited prescription.

    Status after successful review:
        DOCTOR_EDITED
    """

    try:

        timestamp = get_current_timestamp()

        reviewed_prescriptions[
            prescription_id
        ] = {

            "prescription": (
                request.prescription
            ),

            "status": (
                PrescriptionStatus.DOCTOR_EDITED
            ),

            "doctorId": request.doctorId,

            "editReason": request.editReason,

            "approvalNote": None,

            "createdAt": timestamp,

            "updatedAt": timestamp,

            "approvedAt": None
        }

        return PrescriptionReviewResponse(

            success=True,

            prescription=(
                request.prescription
            ),

            status=(
                PrescriptionStatus.DOCTOR_EDITED
            ),

            doctorId=(
                request.doctorId
            ),

            editReason=(
                request.editReason
            ),

            timestamp=timestamp
        )

    except Exception as e:

        import traceback

        traceback.print_exc()

        return PrescriptionReviewResponse(
            success=False,
            error=(
                f"{type(e).__name__}: {e}"
            )
        )


# =========================================================
# DOCTOR APPROVE PRESCRIPTION
# =========================================================

@app.post(
    "/prescription/approve/{prescription_id}",
    response_model=PrescriptionReviewResponse
)
async def approve_prescription(
    prescription_id: str,
    request: DoctorApprovalRequest
):
    """
    Approve a doctor-reviewed prescription.

    Only AI_DRAFT or DOCTOR_EDITED records can move
    into APPROVED state.

    In this development implementation, the record
    must already exist in the in-memory store.
    """

    try:

        # -------------------------------------------------
        # Check prescription exists
        # -------------------------------------------------

        if prescription_id not in reviewed_prescriptions:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Prescription not found "
                    "for approval"
                )
            )

        record = reviewed_prescriptions[
            prescription_id
        ]

        # -------------------------------------------------
        # Current status
        # -------------------------------------------------

        current_status = record.get(
            "status"
        )

        if current_status not in [

            PrescriptionStatus.AI_DRAFT,

            PrescriptionStatus.DOCTOR_EDITED

        ]:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Prescription cannot be approved "
                    "from its current status"
                )
            )

        # -------------------------------------------------
        # Timestamp
        # -------------------------------------------------

        timestamp = get_current_timestamp()

        # -------------------------------------------------
        # Update record
        # -------------------------------------------------

        record["status"] = (
            PrescriptionStatus.APPROVED
        )

        record["doctorId"] = (
            request.doctorId
        )

        record["approvalNote"] = (
            request.approvalNote
        )

        record["approvedAt"] = (
            timestamp
        )

        record["updatedAt"] = (
            timestamp
        )

        reviewed_prescriptions[
            prescription_id
        ] = record

        return PrescriptionReviewResponse(

            success=True,

            prescription=(
                record["prescription"]
            ),

            status=(
                PrescriptionStatus.APPROVED
            ),

            doctorId=(
                request.doctorId
            ),

            approvalNote=(
                request.approvalNote
            ),

            timestamp=timestamp
        )

    except HTTPException:
        raise

    except Exception as e:

        import traceback

        traceback.print_exc()

        return PrescriptionReviewResponse(
            success=False,
            error=(
                f"{type(e).__name__}: {e}"
            )
        )


# =========================================================
# GET PRESCRIPTION REVIEW STATUS
# =========================================================

@app.get(
    "/prescription/{prescription_id}"
)
async def get_prescription(
    prescription_id: str
):

    try:

        if prescription_id not in reviewed_prescriptions:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Prescription not found"
                )
            )

        record = (
            reviewed_prescriptions[
                prescription_id
            ]
        )

        prescription = (
            record.get(
                "prescription"
            )
        )

        if hasattr(
            prescription,
            "model_dump"
        ):

            prescription_data = (
                prescription.model_dump()
            )

        elif hasattr(
            prescription,
            "dict"
        ):

            prescription_data = (
                prescription.dict()
            )

        else:

            prescription_data = prescription

        return {

            "success": True,

            "prescriptionId":
                prescription_id,

            "status":
                record.get("status"),

            "doctorId":
                record.get("doctorId"),

            "editReason":
                record.get("editReason"),

            "approvalNote":
                record.get("approvalNote"),

            "createdAt":
                record.get("createdAt"),

            "updatedAt":
                record.get("updatedAt"),

            "approvedAt":
                record.get("approvedAt"),

            "prescription":
                prescription_data
        }

    except HTTPException:
        raise

    except Exception as e:

        return {

            "success": False,

            "error": (
                f"{type(e).__name__}: {e}"
            )
        }


# =========================================================
# EVALUATE PIPELINE METRICS
# =========================================================

@app.post("/evaluate")
async def evaluate_endpoint(
    request: EvaluationRequest
):
    """
    Evaluate Voice-to-Prescription performance.

    Metrics:
        - Word Error Rate (WER)
        - Precision
        - Recall
        - F1 Score
        - Hallucination Rate
    """

    try:

        evaluation = evaluate_pipeline(

            reference_transcript=(
                request.referenceTranscript
            ),

            predicted_transcript=(
                request.predictedTranscript
            ),

            reference_prescription=(
                request.referencePrescription
            ),

            predicted_prescription=(
                request.predictedPrescription
            )
        )

        return {

            "success": True,

            "evaluation": evaluation

        }

    except Exception as e:

        import traceback

        print(
            "\n========== EVALUATION ERROR =========="
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        print(
            "======================================\n"
        )

        return {

            "success": False,

            "error": (
                f"{type(e).__name__}: {e}"
            )
        }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )