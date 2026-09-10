import time
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass

import numpy as np

from app.models.schemas import (
    Prescription,
    Medicine,
    ExtractionResult,
    ValidationResult,
    Route,
)

from app.pipeline.stt import get_stt

from app.pipeline.ollama_extractor import (
    get_ollama_extractor,
    OllamaPrescriptionExtractor,
    OllamaConfig,
)

from app.pipeline.validator import get_validator
from app.pipeline.transformer import get_transformer

from app.pipeline.medical_terminology import (
    get_medical_terminology_service,
)

from app.utils.audio_utils import (
    decode_base64_audio,
    load_audio_file,
    is_audio_valid,
)


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

@dataclass
class PipelineConfig:

    # Local Whisper model
    whisper_model: str = "base"

    # Ollama is ALWAYS enabled
    use_ollama_extraction: bool = True

    # Default local Ollama configuration
    ollama_config: Optional[OllamaConfig] = None

    # spaCy is disabled by default.
    # We do NOT want spaCy replacing the LLM.
    use_spacy_fallback: bool = False

    spacy_model: str = "en_core_web_sm"

    # Language
    language: str = "en"


# ============================================================
# MAIN PIPELINE
# ============================================================

class VoiceToPrescriptionPipeline:
    """
    Voice-to-Prescription pipeline.

    Flow:

        Audio
          ↓
        faster-whisper BASE
          ↓
        Raw Transcript
          ↓
        Ollama / ministral-3:3b
          ↓
        Medicine extraction
          ↓
        Medical terminology normalization
          ↓
        Validation
          ↓
        Medhant Lite transformer
          ↓
        Final JSON
    """

    def __init__(
        self,
        config: PipelineConfig = None
    ):

        self.config = config or PipelineConfig()

        # faster-whisper does not accept "auto" as a language code.
        # Normalize automatic/empty language values to English.
        if (
            not self.config.language
            or self.config.language.lower() == "auto"
        ):
            self.config.language = "en"

        # ----------------------------------------------------
        # Speech-to-text
        # ----------------------------------------------------

        # Always use BASE Whisper.
        self.config.whisper_model = "base"

        self.stt = get_stt(
            self.config.whisper_model
        )

        # ----------------------------------------------------
        # Validator
        # ----------------------------------------------------

        self.validator = get_validator()

        # ----------------------------------------------------
        # Medical terminology service
        # ----------------------------------------------------

        self.medical_terminology = (
            get_medical_terminology_service()
        )

        # ----------------------------------------------------
        # Ollama
        # ----------------------------------------------------

        self.ollama_extractor = None

        # Force Ollama
        ollama_cfg = (
            self.config.ollama_config
            or OllamaConfig(
                model="ministral-3:3b",
                base_url="http://localhost:11434"
            )
        )

        self.ollama_extractor = (
            get_ollama_extractor(
                ollama_cfg
            )
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        self.transformer = get_transformer()

        # ----------------------------------------------------
        # Seed known medical vocabulary
        # ----------------------------------------------------

        self._seed_known_medicines()

    # ========================================================
    # SEED KNOWN MEDICINES
    # ========================================================

    def _seed_known_medicines(self):
        """
        Seed the terminology database with medicines already
        maintained by the extraction layer.

        This is NOT the final medical database.
        It is a local cache/bootstrap vocabulary.
        """

        try:

            known_medicines = (
                self.ollama_extractor.KNOWN_MEDICINES
            )

            for medicine in known_medicines:

                self.medical_terminology.save_term(
                    term=medicine,
                    normalized_name=medicine,
                    rxcui=None,
                    source="local_known_medicine"
                )

        except Exception as e:

            print(
                "Medical terminology bootstrap warning:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

    # ========================================================
    # NORMALIZE MEDICINE NAME
    # ========================================================

    def _normalize_medicine_name(
        self,
        medicine_name: str
    ) -> str:
        """
        Resolve an extracted medicine name against the local
        terminology cache and standardized terminology.

        IMPORTANT:
        - Only medicine names are resolved.
        - No arbitrary transcript words are converted.
        - Low-confidence matches are NOT automatically accepted.
        """

        if not medicine_name:
            return medicine_name

        original_name = (
            str(medicine_name).strip()
        )

        if not original_name:
            return original_name

        try:

            # ------------------------------------------------
            # First: local exact/fuzzy lookup
            # ------------------------------------------------

            exact = (
                self.medical_terminology
                .local_exact_lookup(
                    original_name
                )
            )

            if exact and exact.matched:

                print(
                    "Medical terminology exact match: "
                    f"{original_name} -> "
                    f"{exact.normalized_name}"
                )

                return (
                    exact.normalized_name
                    or original_name
                )

            fuzzy = (
                self.medical_terminology
                .local_fuzzy_lookup(
                    original_name,
                    threshold=0.90
                )
            )

            if fuzzy and fuzzy.matched:

                print(
                    "Medical terminology fuzzy match: "
                    f"{original_name} -> "
                    f"{fuzzy.normalized_name} "
                    f"(score={fuzzy.confidence:.3f})"
                )

                return (
                    fuzzy.normalized_name
                    or original_name
                )

            # ------------------------------------------------
            # External RxNorm approximate lookup
            # ------------------------------------------------

            try:

                rxnorm_result = asyncio.run(
                    self.medical_terminology
                    .rxnorm_lookup(
                        original_name
                    )
                )

            except Exception as e:

                print(
                    "RxNorm lookup skipped: "
                    f"{type(e).__name__}: {e}"
                )

                rxnorm_result = None

            if (
                rxnorm_result
                and rxnorm_result.matched
                and rxnorm_result.normalized_name
            ):

                print(
                    "Medical terminology RxNorm match: "
                    f"{original_name} -> "
                    f"{rxnorm_result.normalized_name} "
                    f"(score={rxnorm_result.confidence:.3f})"
                )

                return (
                    rxnorm_result.normalized_name
                )

        except Exception as e:

            print(
                "Medical terminology normalization failed "
                f"for '{original_name}': "
                f"{type(e).__name__}: {e}"
            )

        # ----------------------------------------------------
        # No reliable match
        # ----------------------------------------------------

        return original_name

    # ========================================================
    # EXTRACT USING OLLAMA
    # ========================================================

    def _extract_with_ollama(
        self,
        text: str
    ):

        if not text or not text.strip():

            return []

        if self.ollama_extractor is None:

            raise RuntimeError(
                "Ollama extractor is not initialized."
            )

        medicines = (
            self.ollama_extractor.extract(
                text
            )
        )

        if not medicines:

            raise RuntimeError(
                "Ollama did not extract any medicines "
                "from the transcript."
            )

        return medicines

    # ========================================================
    # MEDICAL TERMINOLOGY NORMALIZATION
    # ========================================================

    def _normalize_extracted_medicines(
        self,
        extracted_medicines
    ):
        """
        Normalize extracted medicine names using the
        medical terminology service.

        Other prescription fields are preserved exactly
        as extracted.
        """

        normalized_medicines = []

        for ext_med in extracted_medicines:

            try:

                original_name = getattr(
                    ext_med,
                    "name",
                    ""
                )

                normalized_name = (
                    self._normalize_medicine_name(
                        original_name
                    )
                )

                # Preserve all existing fields.
                ext_med.name = normalized_name

                normalized_medicines.append(
                    ext_med
                )

            except Exception as e:

                print(
                    "Medicine normalization error:"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                normalized_medicines.append(
                    ext_med
                )

        return normalized_medicines

    # ========================================================
    # CONVERT EXTRACTED MEDICINES TO SCHEMA
    # ========================================================

    def _build_prescription(
        self,
        extracted_medicines,
        transcript: str
    ) -> Prescription:

        medicines = []

        for ext_med in extracted_medicines:

            med = Medicine(

                medicineName=(
                    ext_med.name
                    if ext_med.name
                    else ""
                ),

                dosage=(
                    ext_med.dosage
                    if ext_med.dosage
                    else None
                ),

                frequency=(
                    ext_med.frequency
                    if ext_med.frequency
                    else None
                ),

                duration=(
                    ext_med.duration
                    if ext_med.duration
                    else None
                ),

                route=(
                    ext_med.route
                    if ext_med.route
                    else Route.UNKNOWN
                ),

                instructions=(
                    ext_med.instructions
                    if ext_med.instructions
                    else None
                ),

                confidence=(
                    ext_med.confidence
                    if ext_med.confidence is not None
                    else 1.0
                )
            )

            medicines.append(
                med
            )

        return Prescription(

            medicines=medicines,

            notes=None,

            rawTranscript=transcript
        )

    # ========================================================
    # PROCESS AUDIO
    # ========================================================

    def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: str = "en"
    ) -> ExtractionResult:

        if (
            not language
            or language.lower() == "auto"
        ):
            language = "en"

        start_time = time.time()

        # ----------------------------------------------------
        # Validate audio
        # ----------------------------------------------------

        valid, msg = is_audio_valid(
            audio_data,
            sample_rate
        )

        if not valid:

            return ExtractionResult(

                prescription=Prescription(),

                validation=ValidationResult(
                    isValid=False,
                    issues=[],
                    missingFields=[]
                ),

                processingTimeMs=(
                    time.time() - start_time
                ) * 1000,

                sttConfidence=0.0
            )

        # ----------------------------------------------------
        # STEP 1
        # Speech -> Text
        # ----------------------------------------------------

        stt_result = self.stt.transcribe(
            audio_data,
            sample_rate,
            language
        )

        transcript = getattr(
            stt_result,
            "text",
            ""
        )

        stt_confidence = getattr(
            stt_result,
            "confidence",
            None
        )

        transcript = (
            transcript.strip()
            if transcript
            else ""
        )

        if not transcript:

            return ExtractionResult(

                prescription=Prescription(
                    rawTranscript=""
                ),

                validation=ValidationResult(
                    isValid=False,
                    issues=[],
                    missingFields=[]
                ),

                processingTimeMs=(
                    time.time() - start_time
                ) * 1000,

                sttConfidence=stt_confidence
            )

        # ----------------------------------------------------
        # STEP 2
        # Transcript -> Ollama
        # ----------------------------------------------------

        try:

            extracted_medicines = (
                self._extract_with_ollama(
                    transcript
                )
            )

        except Exception as e:

            return ExtractionResult(

                prescription=Prescription(
                    medicines=[],
                    notes=(
                        "Ollama extraction failed: "
                        f"{str(e)}"
                    ),
                    rawTranscript=transcript
                ),

                validation=ValidationResult(
                    isValid=False,
                    issues=[],
                    missingFields=[]
                ),

                processingTimeMs=(
                    time.time() - start_time
                ) * 1000,

                sttConfidence=stt_confidence
            )

        # ----------------------------------------------------
        # STEP 3
        # Medical terminology normalization
        # ----------------------------------------------------

        extracted_medicines = (
            self._normalize_extracted_medicines(
                extracted_medicines
            )
        )

        # ----------------------------------------------------
        # STEP 4
        # Build internal prescription schema
        # ----------------------------------------------------

        prescription = self._build_prescription(
            extracted_medicines,
            transcript
        )

        # ----------------------------------------------------
        # STEP 5
        # Validate
        # ----------------------------------------------------

        validation = self.validator.validate(
            prescription
        )

        # ----------------------------------------------------
        # STEP 6
        # Return
        # ----------------------------------------------------

        processing_time = (
            time.time() - start_time
        ) * 1000

        return ExtractionResult(

            prescription=prescription,

            validation=validation,

            processingTimeMs=processing_time,

            sttConfidence=stt_confidence
        )

    # ========================================================
    # TRANSFORM PRESCRIPTION
    # ========================================================

    def transform_prescription(
        self,
        prescription: Prescription
    ) -> Dict[str, Any]:
        """
        Convert internal prescription schema
        to exact Medhant Lite JSON structure.
        """

        return self.transformer.transform(
            prescription
        )

    # ========================================================
    # PROCESS AUDIO FILE
    # ========================================================

    def process_audio_file(
        self,
        file_path: str,
        language: str = "en"
    ) -> ExtractionResult:

        if (
            not language
            or language.lower() == "auto"
        ):
            language = "en"

        audio_data, sample_rate = (
            load_audio_file(
                file_path
            )
        )

        return self.process_audio(
            audio_data,
            sample_rate,
            language
        )

    # ========================================================
    # PROCESS BASE64 AUDIO
    # ========================================================

    def process_base64_audio(
        self,
        base64_audio: str,
        language: str = "en"
    ) -> ExtractionResult:

        if (
            not language
            or language.lower() == "auto"
        ):
            language = "en"

        audio_data, sample_rate = (
            decode_base64_audio(
                base64_audio
            )
        )

        return self.process_audio(
            audio_data,
            sample_rate,
            language
        )

    # ========================================================
    # PROCESS TEXT
    # ========================================================

    def process_text(
        self,
        text: str
    ) -> ExtractionResult:

        start_time = time.time()

        text = (
            text.strip()
            if text
            else ""
        )

        if not text:

            return ExtractionResult(

                prescription=Prescription(
                    rawTranscript=""
                ),

                validation=ValidationResult(
                    isValid=False,
                    issues=[],
                    missingFields=[]
                ),

                processingTimeMs=0,

                sttConfidence=None
            )

        # ----------------------------------------------------
        # TEXT -> OLLAMA
        # ----------------------------------------------------

        try:

            extracted_medicines = (
                self._extract_with_ollama(
                    text
                )
            )

        except Exception as e:

            return ExtractionResult(

                prescription=Prescription(

                    medicines=[],

                    notes=(
                        "Ollama extraction failed: "
                        f"{str(e)}"
                    ),

                    rawTranscript=text
                ),

                validation=ValidationResult(
                    isValid=False,
                    issues=[],
                    missingFields=[]
                ),

                processingTimeMs=(
                    time.time() - start_time
                ) * 1000,

                sttConfidence=None
            )

        # ----------------------------------------------------
        # MEDICAL TERMINOLOGY NORMALIZATION
        # ----------------------------------------------------

        extracted_medicines = (
            self._normalize_extracted_medicines(
                extracted_medicines
            )
        )

        # ----------------------------------------------------
        # BUILD PRESCRIPTION
        # ----------------------------------------------------

        prescription = self._build_prescription(
            extracted_medicines,
            text
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        validation = self.validator.validate(
            prescription
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return ExtractionResult(

            prescription=prescription,

            validation=validation,

            processingTimeMs=(
                time.time() - start_time
            ) * 1000,

            sttConfidence=None
        )


# ============================================================
# GLOBAL PIPELINE
# ============================================================

_pipeline_instance: Optional[
    VoiceToPrescriptionPipeline
] = None


# ============================================================
# GET PIPELINE
# ============================================================

def get_pipeline(
    config: PipelineConfig = None
) -> VoiceToPrescriptionPipeline:

    global _pipeline_instance

    if _pipeline_instance is None:

        # Always create with Ollama enabled

        if config is None:

            config = PipelineConfig(
                whisper_model="base",
                use_ollama_extraction=True,
                use_spacy_fallback=False,
                language="en",
                ollama_config=OllamaConfig(
                    model="ministral-3:3b",
                    base_url="http://localhost:11434"
                )
            )

        else:

            # Force local LLM
            config.use_ollama_extraction = True

            # Disable spaCy fallback
            config.use_spacy_fallback = False

            # Default language
            if not config.language:
                config.language = "en"

            # Always use BASE Whisper
            config.whisper_model = "base"

            if config.ollama_config is None:

                config.ollama_config = OllamaConfig(
                    model="ministral-3:3b",
                    base_url="http://localhost:11434"
                )

        _pipeline_instance = (
            VoiceToPrescriptionPipeline(
                config
            )
        )

    return _pipeline_instance


# ============================================================
# CREATE NEW PIPELINE
# ============================================================

def create_pipeline(
    config: PipelineConfig
) -> VoiceToPrescriptionPipeline:

    # Force Ollama
    config.use_ollama_extraction = True

    # Disable spaCy fallback
    config.use_spacy_fallback = False

    # Use a valid faster-whisper language code.
    if (
        not config.language
        or config.language.lower() == "auto"
    ):
        config.language = "en"

    # Always use BASE Whisper.
    config.whisper_model = "base"

    # Force Ministral
    if config.ollama_config is None:

        config.ollama_config = OllamaConfig(
            model="ministral-3:3b",
            base_url="http://localhost:11434"
        )

    return VoiceToPrescriptionPipeline(
        config
    )