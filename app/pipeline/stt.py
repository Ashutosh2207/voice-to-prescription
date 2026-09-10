from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


@dataclass
class STTResult:
    text: str
    confidence: Optional[float]
    segments: List[Dict[str, Any]]


class WhisperSTT:
    """
    Speech-to-Text engine using faster-whisper.

    Current configuration:
    - Model: base
    - Device: CPU
    - Compute type: int8

    The raw Whisper transcript is preserved exactly so that
    downstream medical normalization/correction can be audited separately.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: str = "en",
    ):
        self.model_size = model_size
        self.language = language

        # Safe default for laptop/CPU environments.
        if device is None:
            device = "cpu"

        self.device = device

        if compute_type is None:
            compute_type = "int8" if self.device == "cpu" else "float16"

        self.compute_type = compute_type

        print(
            f"Loading faster-whisper model: "
            f"{self.model_size} on {self.device} ({self.compute_type})"
        )

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        print("faster-whisper model loaded successfully.")

    # ------------------------------------------------------------------
    # AUDIO PREPARATION
    # ------------------------------------------------------------------

    def _prepare_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Prepare audio for Whisper:
        - float32
        - mono
        - remove NaN/Inf
        - resample to 16 kHz
        - normalize safely
        - reduce extremely tiny background values
        - contiguous memory
        """

        if audio is None or len(audio) == 0:
            raise ValueError("Audio is empty.")

        audio = np.asarray(audio, dtype=np.float32)

        # Stereo -> mono
        if audio.ndim == 2:
            # soundfile usually returns shape [samples, channels]
            audio = np.mean(audio, axis=1)

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # --------------------------------------------------------------
        # RESAMPLE TO 16 kHz
        # --------------------------------------------------------------
        target_sr = 16000

        if sample_rate != target_sr:
            try:
                import librosa

                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=target_sr,
                ).astype(np.float32)

            except ImportError:
                # Fallback linear interpolation if librosa is unavailable.
                duration = len(audio) / float(sample_rate)

                if duration <= 0:
                    raise ValueError("Invalid audio duration.")

                new_length = max(
                    1,
                    int(round(duration * target_sr)),
                )

                old_positions = np.linspace(
                    0.0,
                    1.0,
                    num=len(audio),
                    endpoint=False,
                )

                new_positions = np.linspace(
                    0.0,
                    1.0,
                    num=new_length,
                    endpoint=False,
                )

                audio = np.interp(
                    new_positions,
                    old_positions,
                    audio,
                ).astype(np.float32)

        # --------------------------------------------------------------
        # NORMALIZE
        # --------------------------------------------------------------
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0

        if peak > 0:
            # Prevent clipping and bring quiet recordings into a useful
            # range without making them excessively loud.
            audio = audio / peak

        # --------------------------------------------------------------
        # REMOVE VERY SMALL VALUES
        # --------------------------------------------------------------
        audio[np.abs(audio) < 0.005] = 0.0

        # --------------------------------------------------------------
        # FINAL SAFETY
        # --------------------------------------------------------------
        audio = np.clip(audio, -1.0, 1.0)
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        if len(audio) == 0:
            raise ValueError("Prepared audio is empty.")

        return audio

    # ------------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        segments: List[Dict[str, Any]]
    ) -> Optional[float]:
        """
        Convert Whisper avg_logprob values into a bounded 0..1
        confidence estimate.

        This is NOT a medically validated probability.
        It is only an STT quality signal.
        """

        log_probs = []

        for segment in segments:
            value = segment.get("avg_logprob")

            if value is None:
                continue

            try:
                value = float(value)

                if math.isfinite(value):
                    log_probs.append(value)

            except (TypeError, ValueError):
                continue

        if not log_probs:
            return None

        mean_logprob = float(np.mean(log_probs))

        try:
            confidence = math.exp(mean_logprob)
        except OverflowError:
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        return round(confidence, 4)

    # ------------------------------------------------------------------
    # TRANSCRIBE NUMPY AUDIO
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe an in-memory audio array.
        """

        prepared_audio = self._prepare_audio(
            audio,
            sample_rate,
        )

        active_language = language or self.language

        # ------------------------------------------------------------------
        # MEDICAL INITIAL PROMPT
        # ------------------------------------------------------------------

        medical_prompt = (
            "Medical prescription dictation. "
            "Medicine names may include: "
            "aspirin, paracetamol, acetaminophen, amoxicillin, "
            "azithromycin, ibuprofen, cetirizine, levocetirizine, "
            "omeprazole, pantoprazole, rabeprazole, esomeprazole, "
            "metformin, glimepiride, gliclazide, atorvastatin, "
            "rosuvastatin, amlodipine, losartan, telmisartan, "
            "ramipril, atenolol, doxycycline, cefixime, cefuroxime, "
            "cefpodoxime, ciprofloxacin, levofloxacin, "
            "dextromethorphan, montelukast, ondansetron, "
            "domperidone, pantoprazole. "
            "Prescription terms include tablet, capsule, syrup, "
            "injection, cream, ointment, drops, milligram, mg, "
            "milliliter, ml, once daily, twice daily, three times daily, "
            "morning, afternoon, evening, night, after food, before food, "
            "for five days, for seven days."
        )

        # ------------------------------------------------------------------
        # WHISPER SETTINGS
        # ------------------------------------------------------------------

        segments_iter, info = self.model.transcribe(
            prepared_audio,
            language=active_language,
            task="transcribe",

            # Better decoding than greedy decoding.
            beam_size=5,
            best_of=5,

            # Deterministic decoding.
            temperature=0.0,

            # Important for independent prescription dictation.
            condition_on_previous_text=False,

            # Medical vocabulary context.
            initial_prompt=medical_prompt,

            # Voice activity detection.
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 300,
            },

            # Preserve punctuation/case as much as Whisper can.
            word_timestamps=False,
        )

        segments: List[Dict[str, Any]] = []
        transcript_parts: List[str] = []

        for segment in segments_iter:
            text = (segment.text or "").strip()

            if not text:
                continue

            transcript_parts.append(text)

            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                    "avg_logprob": (
                        float(segment.avg_logprob)
                        if segment.avg_logprob is not None
                        else None
                    ),
                }
            )

        raw_text = " ".join(transcript_parts).strip()

        confidence = self._calculate_confidence(segments)

        return STTResult(
            text=raw_text,
            confidence=confidence,
            segments=segments,
        )

    # ------------------------------------------------------------------
    # TRANSCRIBE FILE
    # ------------------------------------------------------------------

    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
    ) -> STTResult:
        """
        Read an audio file and transcribe it.

        SoundFile handles WAV/FLAC and similar formats.
        For MP3/M4A, your existing pydub/ffmpeg pipeline can convert
        the file before calling this method.
        """

        audio, sample_rate = sf.read(
            file_path,
            dtype="float32",
        )

        return self.transcribe(
            audio=audio,
            sample_rate=sample_rate,
            language=language,
        )


# ==========================================================================
# SINGLETON
# ==========================================================================

_stt_instance: Optional[WhisperSTT] = None


def get_stt(
    model_size: str = "base",
    language: str = "en",
) -> WhisperSTT:
    """
    Return a shared STT model instance.

    The model is loaded only once per process.
    """

    global _stt_instance

    if _stt_instance is None:
        _stt_instance = WhisperSTT(
            model_size=model_size,
            device="cpu",
            compute_type="int8",
            language=language,
        )

    return _stt_instance