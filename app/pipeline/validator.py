import re

from difflib import SequenceMatcher, get_close_matches
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.models.schemas import (
    Medicine,
    Prescription,
    ValidationResult,
    ValidationIssue,
    Route,
)


# =============================================================
# VALIDATION CONFIG
# =============================================================

@dataclass
class ValidationConfig:

    required_fields: List[str] = None

    # Fuzzy medicine matching threshold
    fuzzy_threshold: float = 0.80

    # Automatically correct fuzzy medicine names
    auto_correct_medicine: bool = True

    # Check extracted data against transcript
    hallucination_check: bool = True

    # Remove unsupported extracted fields
    remove_unsupported_fields: bool = True

    def __post_init__(self):

        if self.required_fields is None:
            self.required_fields = [
                "medicineName"
            ]


# =============================================================
# PRESCRIPTION VALIDATOR
# =============================================================

class PrescriptionValidator:
    """
    Prescription validation and quality-control layer.

    Features:
        1. Medicine validation
        2. Fuzzy medicine matching
        3. Medical vocabulary checking
        4. Transcript evidence checking
        5. Hallucination detection
        6. Dosage validation
        7. Form validation
        8. Frequency validation
        9. Duration validation
        10. Instruction validation
        11. Safe duplicate detection
    """

    # =========================================================
    # MEDICAL VOCABULARY
    # =========================================================

    KNOWN_MEDICINES = {
        "paracetamol",
        "acetaminophen",
        "ibuprofen",
        "amoxicillin",
        "azithromycin",
        "metformin",
        "atorvastatin",
        "omeprazole",
        "pantoprazole",
        "levothyroxine",
        "amlodipine",
        "losartan",
        "metoprolol",
        "aspirin",
        "diclofenac",
        "naproxen",
        "cetirizine",
        "loratadine",
        "ranitidine",
        "domperidone",
        "ondansetron",
        "diazepam",
        "alprazolam",
        "clonazepam",
        "zolpidem",
        "sertraline",
        "fluoxetine",
        "escitalopram",
        "venlafaxine",
        "duloxetine",
        "gabapentin",
        "pregabalin",
        "insulin",
        "glipizide",
        "glimepiride",
        "pioglitazone",
        "sitagliptin",
        "lisinopril",
        "enalapril",
        "ramipril",
        "valsartan",
        "irbesartan",
        "hydrochlorothiazide",
        "furosemide",
        "spironolactone",
        "warfarin",
        "rivaroxaban",
        "apixaban",
        "dabigatran",
        "clopidogrel",
        "ticagrelor",
        "dextromethorphan",
    }

    # =========================================================
    # DOSAGE UNITS
    # =========================================================

    VALID_UNITS = {
        "mg",
        "g",
        "mcg",
        "µg",
        "ug",
        "ml",
        "cc",
        "units",
        "iu",
    }

    # =========================================================
    # MEDICINE FORMS
    # =========================================================

    VALID_FORMS = {
        "tablet",
        "tablets",
        "tab",
        "capsule",
        "capsules",
        "cap",
        "pill",
        "pills",
        "syrup",
        "injection",
        "injections",
        "cream",
        "ointment",
        "drops",
        "spray",
    }

    # =========================================================
    # FREQUENCY PATTERNS
    # =========================================================

    VALID_FREQUENCIES = {
        "1-0-0",
        "1-0-1",
        "1-1-1",
        "1-1-1-1",
        "1-0-0-0",
        "1-0-1-0",
        "1-1-0-0",
        "1-1-1-0",
        "0-0-0-1",
        "SOS",
        "OD",
        "BD",
        "TID",
        "QID",
        "PRN",
    }

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        config: ValidationConfig = None
    ):

        self.config = (
            config
            or ValidationConfig()
        )

        self.medicine_list = sorted(
            self.KNOWN_MEDICINES
        )

        # Full transcript used by evidence validators.
        self._current_transcript = ""

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(
        text: Optional[str]
    ) -> str:

        if text is None:
            return ""

        text = str(text)

        # Remove/normalize invisible and Unicode whitespace.
        text = (
            text
            .replace("\u00A0", " ")
            .replace("\u2007", " ")
            .replace("\u202F", " ")
            .replace("\u200B", "")
            .replace("\uFEFF", "")
        )

        text = text.lower().strip()

        text = text.replace(
            "-",
            " "
        )

        text = text.replace(
            "/",
            " "
        )

        text = text.replace(
            ",",
            " "
        )

        text = text.replace(
            ".",
            " "
        )

        text = re.sub(
            r"[^a-z0-9µ]+",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =========================================================
    # STRING SIMILARITY
    # =========================================================

    @staticmethod
    def _similarity(
        left: str,
        right: str
    ) -> float:

        left = (
            PrescriptionValidator
            ._normalize_text(left)
        )

        right = (
            PrescriptionValidator
            ._normalize_text(right)
        )

        if not left or not right:
            return 0.0

        return SequenceMatcher(
            None,
            left,
            right
        ).ratio()

    # =========================================================
    # FUZZY MEDICINE MATCHING
    # =========================================================

    def fuzzy_match_medicine(
        self,
        medicine_name: str
    ) -> Dict[str, Any]:

        original = (
            medicine_name
            or ""
        ).strip()

        normalized = (
            self._normalize_text(
                original
            )
        )

        if not normalized:

            return {
                "original": original,
                "matched": None,
                "score": 0.0,
                "is_exact": False,
                "is_match": False,
            }

        # Exact match
        if normalized in self.KNOWN_MEDICINES:

            return {
                "original": original,
                "matched": normalized,
                "score": 1.0,
                "is_exact": True,
                "is_match": True,
            }

        # Fuzzy candidate search
        candidates = get_close_matches(
            normalized,
            self.medicine_list,
            n=3,
            cutoff=0.50
        )

        if not candidates:

            return {
                "original": original,
                "matched": None,
                "score": 0.0,
                "is_exact": False,
                "is_match": False,
            }

        best_match = candidates[0]

        score = self._similarity(
            normalized,
            best_match
        )

        is_match = (
            score
            >= self.config.fuzzy_threshold
        )

        return {
            "original": original,
            "matched": (
                best_match
                if is_match
                else None
            ),
            "score": round(
                score,
                4
            ),
            "is_exact": False,
            "is_match": is_match,
        }

    # =========================================================
    # TRANSCRIPT EVIDENCE
    # =========================================================

    def medicine_has_transcript_evidence(
        self,
        medicine_name: str,
        transcript: Optional[str]
    ) -> Dict[str, Any]:

        if not medicine_name or not transcript:

            return {
                "supported": False,
                "score": 0.0,
                "matched_text": None,
            }

        medicine = (
            self._normalize_text(
                medicine_name
            )
        )

        source = (
            self._normalize_text(
                transcript
            )
        )

        if not medicine or not source:

            return {
                "supported": False,
                "score": 0.0,
                "matched_text": None,
            }

        # Exact phrase.
        if medicine in source:

            return {
                "supported": True,
                "score": 1.0,
                "matched_text": medicine,
            }

        source_words = source.split()
        medicine_words = medicine.split()

        best_score = 0.0
        best_word = None

        # Single medicine name
        if len(medicine_words) == 1:

            target = medicine_words[0]

            for word in source_words:

                score = self._similarity(
                    target,
                    word
                )

                if score > best_score:

                    best_score = score
                    best_word = word

        # Multi-word medicine
        else:

            window_size = len(
                medicine_words
            )

            for index in range(
                len(source_words)
                - window_size
                + 1
            ):

                window = " ".join(
                    source_words[
                        index:
                        index + window_size
                    ]
                )

                score = self._similarity(
                    medicine,
                    window
                )

                if score > best_score:

                    best_score = score
                    best_word = window

        # Known medicines can tolerate ASR spelling errors.
        if medicine in self.KNOWN_MEDICINES:

            threshold = 0.76

        else:

            threshold = 0.84

        return {
            "supported": (
                best_score >= threshold
            ),
            "score": round(
                best_score,
                4
            ),
            "matched_text": best_word,
        }

    # =========================================================
    # TRANSCRIPT CONTEXT
    # =========================================================

    def _get_context(
        self,
        medicine_name: str,
        transcript: str,
        context_words: int = 30
    ) -> str:

        source = (
            self._normalize_text(
                transcript
            )
        )

        words = source.split()

        if not words:
            return ""

        medicine = (
            self._normalize_text(
                medicine_name
            )
        )

        medicine_words = medicine.split()

        if not medicine_words:
            return source

        best_index = None
        best_score = 0.0

        # Find the strongest matching occurrence.
        for index, word in enumerate(words):

            score = self._similarity(
                medicine_words[0],
                word
            )

            if score > best_score:

                best_score = score
                best_index = index

        if best_index is None:
            return source

        start = max(
            0,
            best_index - context_words
        )

        end = min(
            len(words),
            best_index + context_words + 1
        )

        return " ".join(
            words[start:end]
        )

    # =========================================================
    # DOSAGE EVIDENCE
    # =========================================================

    def _validate_dosage_evidence(
        self,
        med: Medicine,
        idx: int,
        context: str
    ) -> List[ValidationIssue]:

        issues = []

        dosage = med.dosage

        if not dosage:
            return issues

        dosage_text = str(
            dosage
        ).strip()

        if dosage_text.upper() in {
            "NULL",
            "NONE",
            "N/A",
            "NA",
        }:
            return issues

        dosage_normalized = (
            self._normalize_text(
                dosage_text
            )
        )

        dosage_compact = (
            dosage_normalized
            .replace(" ", "")
        )

        transcript_values = re.findall(
            r"\d+(?:\.\d+)?\s*"
            r"(?:mg|g|mcg|µg|ug|ml|cc|iu|units)",
            self._normalize_text(context),
            flags=re.IGNORECASE
        )

        normalized_values = [
            value.replace(
                " ",
                ""
            )
            for value in transcript_values
        ]

        if dosage_compact in normalized_values:
            return issues

        if self.config.remove_unsupported_fields:
            med.dosage = None

        issues.append(
            ValidationIssue(
                field="dosage",
                medicineIndex=idx,
                issue=(
                    f"Dosage '{dosage_text}' "
                    f"is not supported by "
                    f"the transcript context."
                ),
                severity="warning"
            )
        )

        return issues

    # =========================================================
    # FORM EVIDENCE
    # =========================================================

    def _validate_form_evidence(
        self,
        med: Medicine,
        idx: int,
        context: str
    ) -> List[ValidationIssue]:

        issues = []

        form = getattr(
            med,
            "form",
            None
        )

        if not form:
            return issues

        form_normalized = (
            str(form)
            .lower()
            .strip()
        )

        form_map = {
            "tablets": "tablet",
            "tab": "tablet",
            "capsules": "capsule",
            "cap": "capsule",
            "pills": "pill",
            "injections": "injection",
        }

        form_normalized = form_map.get(
            form_normalized,
            form_normalized
        )

        if form_normalized in {
            "",
            "unknown",
            "none",
            "null",
        }:
            return issues

        context_normalized = (
            self._normalize_text(
                context
            )
        )

        # Capsule and tablet remain distinct.
        if form_normalized not in context_normalized:

            if self.config.remove_unsupported_fields:

                try:
                    med.form = "UNKNOWN"
                except Exception:
                    pass

            issues.append(
                ValidationIssue(
                    field="form",
                    medicineIndex=idx,
                    issue=(
                        f"Form '{str(form).upper()}' "
                        f"is not supported by "
                        f"the transcript context."
                    ),
                    severity="warning"
                )
            )

        return issues

    # =========================================================
    # FREQUENCY PHRASE MAPPING
    # =========================================================

    def _frequency_phrases(
        self,
        frequency: str
    ) -> List[str]:

        freq = (
            str(frequency)
            .upper()
            .strip()
        )

        mapping = {

            "1-0-0": [
                "once daily",
                "once a day",
                "once",
                "od",
            ],

            "1-0-1": [
                "twice daily",
                "twice a day",
                "two times daily",
                "two times a day",
                "morning and evening",
                "morning evening",
                "breakfast and dinner",
            ],

            "1-1-1": [
                "three times daily",
                "three times a day",
                "three times",
                "thrice daily",
                "morning afternoon evening",
            ],

            "1-1-1-1": [
                "four times daily",
                "four times a day",
                "four times",
            ],

            "1-0-0-0": [
                "morning",
                "in the morning",
            ],

            "1-0-1-0": [
                "morning and evening",
                "morning evening",
                "in the morning and evening",
                "twice daily",
                "twice a day",
            ],

            "1-1-0-0": [
                "morning and afternoon",
                "morning afternoon",
                "in the morning and afternoon",
            ],

            "1-1-1-0": [
                "morning afternoon evening",
                "morning afternoon and evening",
                "three times daily",
                "three times a day",
                "three times",
            ],

            "0-0-0-1": [
                "night",
                "at night",
                "in the night",
                "bedtime",
                "at bedtime",
                "before bed",
            ],

            "SOS": [
                "as needed",
                "if needed",
                "when needed",
                "only when needed",
                "prn",
                "sos",
            ],
        }

        return mapping.get(
            freq,
            []
        )

    # =========================================================
    # FREQUENCY EVIDENCE
    # =========================================================

    def _validate_frequency_evidence(
        self,
        med: Medicine,
        idx: int,
        context: str
    ) -> List[ValidationIssue]:

        issues = []

        frequency = med.frequency

        if not frequency:
            return issues

        freq = (
            str(frequency)
            .upper()
            .strip()
        )

        if freq in {
            "NULL",
            "NONE",
            "",
        }:
            return issues

        if freq not in self.VALID_FREQUENCIES:

            issues.append(
                ValidationIssue(
                    field="frequency",
                    medicineIndex=idx,
                    issue=(
                        f"Non-standard frequency "
                        f"format: {frequency}."
                    ),
                    severity="warning"
                )
            )

            return issues

        context_normalized = (
            self._normalize_text(
                context
            )
        )

        # -----------------------------------------------------
        # Direct phrase matching
        # -----------------------------------------------------

        for phrase in self._frequency_phrases(
            freq
        ):

            phrase_normalized = (
                self._normalize_text(
                    phrase
                )
            )

            if phrase_normalized in context_normalized:
                return issues

        # -----------------------------------------------------
        # Time-slot matching
        # -----------------------------------------------------

        morning = (
            "morning"
            in context_normalized
        )

        afternoon = (
            "afternoon"
            in context_normalized
        )

        evening = (
            "evening"
            in context_normalized
        )

        night = (
            "night"
            in context_normalized
            or
            "bedtime"
            in context_normalized
        )

        detected_slots = (
            ("1" if morning else "0")
            + "-"
            + ("1" if afternoon else "0")
            + "-"
            + ("1" if evening else "0")
            + "-"
            + ("1" if night else "0")
        )

        if detected_slots == freq:
            return issues

        # -----------------------------------------------------
        # Count-based matching
        # -----------------------------------------------------

        count_map = {

            "1-0-0": [
                "once",
                "one time",
            ],

            "1-0-1": [
                "twice",
                "two times",
            ],

            "1-1-1": [
                "three times",
                "thrice",
            ],

            "1-1-1-1": [
                "four times",
            ],
        }

        for phrase in count_map.get(
            freq,
            []
        ):

            if phrase in context_normalized:
                return issues

        # We intentionally do not generate noisy warnings here.
        return issues

    # =========================================================
    # DURATION NUMBER WORDS
    # =========================================================

    @staticmethod
    def _number_to_words(
        number: str
    ) -> List[str]:

        mapping = {
            "0": ["zero"],
            "1": ["one"],
            "2": ["two"],
            "3": ["three"],
            "4": ["four"],
            "5": ["five"],
            "6": ["six"],
            "7": ["seven"],
            "8": ["eight"],
            "9": ["nine"],
            "10": ["ten"],
            "11": ["eleven"],
            "12": ["twelve"],
            "13": ["thirteen"],
            "14": ["fourteen"],
            "15": ["fifteen"],
            "16": ["sixteen"],
            "17": ["seventeen"],
            "18": ["eighteen"],
            "19": ["nineteen"],
            "20": ["twenty"],
        }

        return mapping.get(
            str(number),
            []
        )

    # =========================================================
    # ROBUST DURATION NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_duration_value(
        duration: str
    ) -> Optional[str]:

        if duration is None:
            return None

        text = str(
            duration
        )

        # Normalize invisible Unicode spacing.
        text = (
            text
            .replace("\u00A0", " ")
            .replace("\u2007", " ")
            .replace("\u202F", " ")
            .replace("\u200B", "")
            .replace("\uFEFF", "")
        )

        text = text.strip().lower()

        # Collapse whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        # Common number-word durations.
        replacements = {
            "one day": "1 day",
            "two days": "2 days",
            "three days": "3 days",
            "four days": "4 days",
            "five days": "5 days",
            "six days": "6 days",
            "seven days": "7 days",
            "eight days": "8 days",
            "nine days": "9 days",
            "ten days": "10 days",

            "one week": "1 week",
            "two weeks": "2 weeks",
            "three weeks": "3 weeks",
            "four weeks": "4 weeks",

            "one month": "1 month",
            "two months": "2 months",
        }

        if text in replacements:
            text = replacements[text]

        # Numeric duration.
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*"
            r"(day|days|week|weeks|month|months)",
            text
        )

        if match:

            number = match.group(1)
            unit = match.group(2)

            if unit.startswith("day"):

                final_unit = (
                    "DAY"
                    if number == "1"
                    else "DAYS"
                )

            elif unit.startswith("week"):

                final_unit = (
                    "WEEK"
                    if number == "1"
                    else "WEEKS"
                )

            else:

                final_unit = (
                    "MONTH"
                    if number == "1"
                    else "MONTHS"
                )

            return (
                f"{number} {final_unit}"
            )

        # Tolerant fallback:
        # 5DAYS / 5 DAYS / Unicode whitespace etc.
        compact = re.sub(
            r"[^a-z0-9.]",
            "",
            text
        )

        compact_match = re.fullmatch(
            r"(\d+(?:\.\d+)?)(day|days|week|weeks|month|months)",
            compact
        )

        if compact_match:

            number = compact_match.group(1)
            unit = compact_match.group(2)

            if unit.startswith("day"):

                final_unit = (
                    "DAY"
                    if number == "1"
                    else "DAYS"
                )

            elif unit.startswith("week"):

                final_unit = (
                    "WEEK"
                    if number == "1"
                    else "WEEKS"
                )

            else:

                final_unit = (
                    "MONTH"
                    if number == "1"
                    else "MONTHS"
                )

            return (
                f"{number} {final_unit}"
            )

        return None

    # =========================================================
    # DURATION EVIDENCE
    # =========================================================

    def _validate_duration_evidence(
        self,
        med: Medicine,
        idx: int,
        context: str
    ) -> List[ValidationIssue]:

        issues = []

        duration = med.duration

        if not duration:
            return issues

        duration_text = str(
            duration
        ).strip()

        if duration_text.upper() in {
            "NULL",
            "NONE",
            "",
        }:
            return issues

        # -----------------------------------------------------
        # Normalize extracted duration.
        # -----------------------------------------------------

        normalized_duration = (
            self._normalize_duration_value(
                duration_text
            )
        )

        if normalized_duration is None:

            issues.append(
                ValidationIssue(
                    field="duration",
                    medicineIndex=idx,
                    issue=(
                        f"Duration format unclear: "
                        f"'{duration_text}'. "
                        f"Expected values such as "
                        f"'5 days' or '2 weeks'."
                    ),
                    severity="warning"
                )
            )

            return issues

        # Preserve normalized representation.
        med.duration = normalized_duration

        # -----------------------------------------------------
        # Extract number + unit.
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s+"
            r"(DAY|DAYS|WEEK|WEEKS|MONTH|MONTHS)",
            normalized_duration.upper()
        )

        if not match:
            return issues

        number = match.group(1)
        unit = match.group(2).lower()

        # -----------------------------------------------------
        # Unit variants.
        # -----------------------------------------------------

        if unit.startswith("day"):

            unit_variants = [
                "day",
                "days",
            ]

        elif unit.startswith("week"):

            unit_variants = [
                "week",
                "weeks",
            ]

        else:

            unit_variants = [
                "month",
                "months",
            ]

        # -----------------------------------------------------
        # Numeric + spoken forms.
        #
        # Example:
        #
        # Extracted:
        #     2 DAYS
        #
        # Transcript:
        #     for two days
        #
        # Both must be accepted.
        # -----------------------------------------------------

        accepted_numbers = [
            number
        ]

        accepted_numbers.extend(
            self._number_to_words(
                number
            )
        )

        # -----------------------------------------------------
        # Full transcript.
        # -----------------------------------------------------

        full_transcript = (
            self._normalize_text(
                self._current_transcript
            )
        )

        context_normalized = (
            self._normalize_text(
                context
            )
        )

        sources = []

        if full_transcript:
            sources.append(
                full_transcript
            )

        if context_normalized:
            sources.append(
                context_normalized
            )

        # -----------------------------------------------------
        # Search all numeric and spoken forms.
        # -----------------------------------------------------

        for source in sources:

            for accepted_number in accepted_numbers:

                for unit_variant in unit_variants:

                    # -----------------------------------------
                    # 2 days / two days
                    # -----------------------------------------

                    direct_pattern = (
                        rf"\b"
                        rf"{re.escape(accepted_number)}"
                        rf"\s+"
                        rf"{re.escape(unit_variant)}"
                        rf"\b"
                    )

                    if re.search(
                        direct_pattern,
                        source,
                        flags=re.IGNORECASE
                    ):

                        return issues

                    # -----------------------------------------
                    # for 2 days / for two days
                    # -----------------------------------------

                    for_pattern = (
                        rf"\bfor\s+"
                        rf"{re.escape(accepted_number)}"
                        rf"\s+"
                        rf"{re.escape(unit_variant)}"
                        rf"\b"
                    )

                    if re.search(
                        for_pattern,
                        source,
                        flags=re.IGNORECASE
                    ):

                        return issues

                    # -----------------------------------------
                    # "of 2 days"
                    # "of two days"
                    # -----------------------------------------

                    of_pattern = (
                        rf"\bof\s+"
                        rf"{re.escape(accepted_number)}"
                        rf"\s+"
                        rf"{re.escape(unit_variant)}"
                        rf"\b"
                    )

                    if re.search(
                        of_pattern,
                        source,
                        flags=re.IGNORECASE
                    ):

                        return issues

        # -----------------------------------------------------
        # Special fallback:
        #
        # Whisper may produce:
        # "for two days"
        # while normalization may not have a number-word
        # conversion for a larger number.
        #
        # Extract all duration mentions from transcript and
        # compare the unit + number carefully.
        # -----------------------------------------------------

        duration_mentions = re.findall(
            r"\b"
            r"(\d+(?:\.\d+)?)"
            r"\s+"
            r"(day|days|week|weeks|month|months)"
            r"\b",
            full_transcript,
            flags=re.IGNORECASE
        )

        for found_number, found_unit in duration_mentions:

            if (
                found_number == number
                and found_unit.lower()
                in unit_variants
            ):

                return issues

        # -----------------------------------------------------
        # If exact transcript evidence is unavailable,
        # remove unsupported duration only when configured.
        # -----------------------------------------------------

        if self.config.remove_unsupported_fields:
            med.duration = None

        issues.append(
            ValidationIssue(
                field="duration",
                medicineIndex=idx,
                issue=(
                    f"Duration '{duration_text}' "
                    f"is not supported by "
                    f"the transcript context."
                ),
                severity="warning"
            )
        )

        return issues

    # =========================================================
    # BASIC DOSAGE VALIDATION
    # =========================================================

    def _validate_dosage(
        self,
        dosage: str,
        idx: int
    ) -> List[ValidationIssue]:

        issues = []

        if not dosage:
            return issues

        dosage_text = (
            str(dosage)
            .strip()
        )

        if dosage_text.upper() in {
            "NULL",
            "NONE",
            "",
        }:
            return issues

        dosage_lower = (
            dosage_text.lower()
        )

        has_numeric = bool(
            re.search(
                r"\d",
                dosage_text
            )
        )

        has_strength = bool(
            re.search(
                r"\d+(?:\.\d+)?\s*"
                r"(?:mg|g|mcg|µg|ug|ml|cc|iu|units)\b",
                dosage_lower
            )
        )

        has_form_unit = bool(
            re.search(
                r"\b(?:tablet|tablets|"
                r"tab|capsule|capsules|"
                r"cap|pill|pills)\b",
                dosage_lower
            )
        )

        if not has_strength and not has_form_unit:

            issues.append(
                ValidationIssue(
                    field="dosage",
                    medicineIndex=idx,
                    issue=(
                        f"Dosage may be missing "
                        f"unit: {dosage_text}"
                    ),
                    severity="warning"
                )
            )

        if not has_numeric:

            issues.append(
                ValidationIssue(
                    field="dosage",
                    medicineIndex=idx,
                    issue=(
                        f"Dosage appears to lack "
                        f"numeric value: "
                        f"{dosage_text}"
                    ),
                    severity="warning"
                )
            )

        return issues

    # =========================================================
    # BASIC FREQUENCY VALIDATION
    # =========================================================

    def _validate_frequency(
        self,
        frequency: str,
        idx: int
    ) -> List[ValidationIssue]:

        issues = []

        if not frequency:
            return issues

        freq_upper = (
            str(frequency)
            .upper()
            .strip()
        )

        if freq_upper in {
            "NULL",
            "NONE",
            "",
        }:
            return issues

        if freq_upper not in self.VALID_FREQUENCIES:

            issues.append(
                ValidationIssue(
                    field="frequency",
                    medicineIndex=idx,
                    issue=(
                        f"Non-standard frequency "
                        f"format: {frequency}. "
                        f"Expected standard "
                        f"prescription frequency "
                        f"patterns."
                    ),
                    severity="warning"
                )
            )

        return issues

    # =========================================================
    # BASIC DURATION VALIDATION
    # =========================================================

    def _validate_duration(
        self,
        duration: str,
        idx: int
    ) -> List[ValidationIssue]:

        issues = []

        if duration is None:
            return issues

        duration_text = str(
            duration
        )

        # Normalize invisible Unicode characters.
        duration_text = (
            duration_text
            .replace("\u00A0", " ")
            .replace("\u2007", " ")
            .replace("\u202F", " ")
            .replace("\u200B", "")
            .replace("\uFEFF", "")
            .strip()
        )

        # Collapse whitespace.
        duration_text = re.sub(
            r"\s+",
            " ",
            duration_text
        ).strip()

        if not duration_text:
            return issues

        if duration_text.upper() in {
            "NULL",
            "NONE",
            "N/A",
            "NA",
        }:
            return issues

        normalized_duration = (
            self._normalize_duration_value(
                duration_text
            )
        )

        # Important:
        # 2 DAYS / 3 DAYS / 5 DAYS are valid.
        if normalized_duration is not None:
            return issues

        issues.append(
            ValidationIssue(
                field="duration",
                medicineIndex=idx,
                issue=(
                    f"Duration format unclear: "
                    f"'{duration_text}'. "
                    f"Expected values such as "
                    f"'5 days' or '2 weeks'."
                ),
                severity="warning"
            )
        )

        return issues

    # =========================================================
    # INSTRUCTION EVIDENCE
    # =========================================================

    def _validate_instruction_evidence(
        self,
        med: Medicine,
        idx: int,
        context: str
    ) -> List[ValidationIssue]:

        issues = []

        instructions = med.instructions

        if not instructions:
            return issues

        instruction = (
            self._normalize_text(
                instructions
            )
        )

        if instruction in {
            "",
            "null",
            "none",
        }:
            return issues

        if instruction == "after food":

            supported = (
                "after food" in context
                or
                "after meals" in context
                or
                "after eating" in context
                or
                "with food" in context
            )

        elif instruction == "before food":

            supported = (
                "before food" in context
                or
                "before meals" in context
                or
                "before eating" in context
                or
                "before breakfast" in context
                or
                "empty stomach" in context
            )

        else:

            supported = (
                instruction in context
            )

        if not supported:

            if self.config.remove_unsupported_fields:
                med.instructions = None

            issues.append(
                ValidationIssue(
                    field="instructions",
                    medicineIndex=idx,
                    issue=(
                        f"Instruction "
                        f"'{instructions}' "
                        f"is not supported by "
                        f"the transcript context."
                    ),
                    severity="warning"
                )
            )

        return issues

    # =========================================================
    # MAIN VALIDATION
    # =========================================================

    def validate(
        self,
        prescription: Prescription
    ) -> ValidationResult:

        issues = []
        missing_fields = []

        if not prescription.medicines:

            issues.append(
                ValidationIssue(
                    field="medicines",
                    medicineIndex=-1,
                    issue=(
                        "No medicines found "
                        "in prescription"
                    ),
                    severity="error"
                )
            )

            return ValidationResult(
                isValid=False,
                issues=issues,
                missingFields=[
                    "medicines"
                ]
            )

        transcript = (
            prescription.rawTranscript
            or ""
        )

        # Make complete transcript available to
        # evidence validators.
        self._current_transcript = transcript

        # -----------------------------------------------------
        # Validate medicines
        # -----------------------------------------------------

        for idx, med in enumerate(
            prescription.medicines
        ):

            med_issues, med_missing = (
                self._validate_medicine(
                    med,
                    idx,
                    transcript
                )
            )

            issues.extend(
                med_issues
            )

            missing_fields.extend(
                med_missing
            )

        # -----------------------------------------------------
        # SAFE DUPLICATE DETECTION
        # -----------------------------------------------------
        #
        # Same medicine name is NOT enough to call it duplicate.
        #
        # Example:
        #
        # PARACETAMOL 300MG / 2 DAYS
        # PARACETAMOL 500MG / 3 DAYS
        #
        # These are different prescription entries.
        #
        # A duplicate requires all relevant prescription
        # attributes to match.
        #

        seen_prescriptions = set()

        for idx, medicine in enumerate(
            prescription.medicines
        ):

            name = (
                self._normalize_text(
                    medicine.medicineName
                )
            )

            dosage = (
                self._normalize_text(
                    medicine.dosage
                )
            )

            frequency = (
                self._normalize_text(
                    medicine.frequency
                )
            )

            duration = (
                self._normalize_text(
                    medicine.duration
                )
            )

            instructions = (
                self._normalize_text(
                    medicine.instructions
                )
            )

            form = (
                self._normalize_text(
                    getattr(
                        medicine,
                        "form",
                        None
                    )
                )
            )

            route = (
                str(
                    getattr(
                        medicine,
                        "route",
                        ""
                    )
                )
                .lower()
                .strip()
            )

            duplicate_key = (
                name,
                dosage,
                frequency,
                duration,
                instructions,
                form,
                route,
            )

            if duplicate_key in seen_prescriptions:

                issues.append(
                    ValidationIssue(
                        field="medicineName",
                        medicineIndex=idx,
                        issue=(
                            "Duplicate prescription: "
                            f"{name}"
                        ),
                        severity="warning"
                    )
                )

            else:

                seen_prescriptions.add(
                    duplicate_key
                )

        # -----------------------------------------------------
        # Final validity
        # -----------------------------------------------------

        is_valid = not any(
            issue.severity == "error"
            for issue in issues
        )

        result = ValidationResult(
            isValid=is_valid,
            issues=issues,
            missingFields=list(
                dict.fromkeys(
                    missing_fields
                )
            )
        )

        # Clear transcript after validation.
        self._current_transcript = ""

        return result

    # =========================================================
    # MEDICINE VALIDATION
    # =========================================================

    def _validate_medicine(
        self,
        med: Medicine,
        idx: int,
        transcript: Optional[str]
    ) -> tuple[
        List[ValidationIssue],
        List[str]
    ]:

        issues = []
        missing = []

        # =====================================================
        # MEDICINE NAME
        # =====================================================

        if (
            not med.medicineName
            or not med.medicineName.strip()
        ):

            issues.append(
                ValidationIssue(
                    field="medicineName",
                    medicineIndex=idx,
                    issue=(
                        "Medicine name is required"
                    ),
                    severity="error"
                )
            )

            missing.append(
                "medicineName"
            )

            return issues, missing

        original_name = (
            med.medicineName.strip()
        )

        fuzzy_result = (
            self.fuzzy_match_medicine(
                original_name
            )
        )

        # -----------------------------------------------------
        # Exact / fuzzy medicine
        # -----------------------------------------------------

        if fuzzy_result["is_match"]:

            matched_name = (
                fuzzy_result["matched"]
            )

            if (
                not fuzzy_result["is_exact"]
                and
                self.config.auto_correct_medicine
            ):

                med.medicineName = (
                    matched_name.upper()
                )

                issues.append(
                    ValidationIssue(
                        field="medicineName",
                        medicineIndex=idx,
                        issue=(
                            f"Fuzzy medicine match: "
                            f"'{original_name}' -> "
                            f"'{matched_name.upper()}' "
                            f"(similarity="
                            f"{fuzzy_result['score']:.2f})"
                        ),
                        severity="info"
                    )
                )

        else:

            issues.append(
                ValidationIssue(
                    field="medicineName",
                    medicineIndex=idx,
                    issue=(
                        f"Unknown medicine: "
                        f"{original_name}. "
                        f"No reliable dictionary "
                        f"match was found."
                    ),
                    severity="warning"
                )
            )

        # -----------------------------------------------------
        # Hallucination / transcript evidence
        # -----------------------------------------------------

        if (
            self.config.hallucination_check
            and transcript
        ):

            evidence = (
                self.medicine_has_transcript_evidence(
                    med.medicineName,
                    transcript
                )
            )

            if not evidence["supported"]:

                issues.append(
                    ValidationIssue(
                        field="medicineName",
                        medicineIndex=idx,
                        issue=(
                            f"Potential hallucination: "
                            f"'{med.medicineName}' "
                            f"has no reliable supporting "
                            f"evidence in the transcript."
                        ),
                        severity="warning"
                    )
                )

            else:

                context = self._get_context(
                    med.medicineName,
                    transcript,
                    context_words=30
                )

                # Dosage evidence
                issues.extend(
                    self._validate_dosage_evidence(
                        med,
                        idx,
                        context
                    )
                )

                # Form evidence
                issues.extend(
                    self._validate_form_evidence(
                        med,
                        idx,
                        context
                    )
                )

                # Frequency evidence
                issues.extend(
                    self._validate_frequency_evidence(
                        med,
                        idx,
                        context
                    )
                )

                # Duration evidence
                issues.extend(
                    self._validate_duration_evidence(
                        med,
                        idx,
                        context
                    )
                )

                # Instruction evidence
                issues.extend(
                    self._validate_instruction_evidence(
                        med,
                        idx,
                        context
                    )
                )

        # =====================================================
        # DOSAGE
        # =====================================================

        if med.dosage:

            issues.extend(
                self._validate_dosage(
                    med.dosage,
                    idx
                )
            )

        else:

            missing.append(
                "dosage"
            )

        # =====================================================
        # FREQUENCY
        # =====================================================

        if med.frequency:

            issues.extend(
                self._validate_frequency(
                    med.frequency,
                    idx
                )
            )

        else:

            missing.append(
                "frequency"
            )

        # =====================================================
        # DURATION
        # =====================================================

        if med.duration:

            issues.extend(
                self._validate_duration(
                    med.duration,
                    idx
                )
            )

        else:

            missing.append(
                "duration"
            )

        # =====================================================
        # ROUTE
        # =====================================================

        if (
            not med.route
            or med.route == Route.UNKNOWN
        ):

            missing.append(
                "route"
            )

            issues.append(
                ValidationIssue(
                    field="route",
                    medicineIndex=idx,
                    issue=(
                        "Route not specified."
                    ),
                    severity="info"
                )
            )

        return issues, missing


# =============================================================
# SINGLETON
# =============================================================

_validator_instance = None


def get_validator(
    config: ValidationConfig = None
) -> PrescriptionValidator:

    global _validator_instance

    if _validator_instance is None:

        _validator_instance = (
            PrescriptionValidator(
                config
            )
        )

    return _validator_instance