from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ============================================================
# PRESCRIPTION WORKFLOW STATUS
# ============================================================

class PrescriptionStatus(str, Enum):
    AI_DRAFT = "AI_DRAFT"
    DOCTOR_EDITED = "DOCTOR_EDITED"
    APPROVED = "APPROVED"


# ============================================================
# FREQUENCY
# ============================================================

class FrequencyPattern(str, Enum):
    ONCE_DAILY = "1-0-0"
    TWICE_DAILY = "1-0-1"
    THRICE_DAILY = "1-1-1"
    FOUR_TIMES_DAILY = "1-1-1-1"
    AS_NEEDED = "SOS"
    CUSTOM = "custom"


# ============================================================
# ROUTE
# ============================================================

class Route(str, Enum):
    ORAL = "oral"
    TOPICAL = "topical"
    INJECTION = "injection"
    INHALATION = "inhalation"
    SUBLINGUAL = "sublingual"
    RECTAL = "rectal"
    OPHTHALMIC = "ophthalmic"
    OTIC = "otic"
    NASAL = "nasal"
    TRANSDERMAL = "transdermal"
    UNKNOWN = "unknown"


# ============================================================
# MEDICINE
# ============================================================

class Medicine(BaseModel):
    medicineName: str = Field(
        ...,
        description="Name of the medicine"
    )

    dosage: Optional[str] = Field(
        None,
        description="Dosage/strength (e.g., '500 mg', '10 ml')"
    )

    frequency: Optional[str] = Field(
        None,
        description="Frequency (e.g., '1-0-1', 'SOS', 'BD')"
    )

    duration: Optional[str] = Field(
        None,
        description="Duration (e.g., '5 days', '2 weeks')"
    )

    route: Optional[Route] = Field(
        Route.UNKNOWN,
        description="Route of administration"
    )

    instructions: Optional[str] = Field(
        None,
        description="Special instructions (e.g., 'After food')"
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score"
    )

    @field_validator(
        "frequency",
        mode="before"
    )
    @classmethod
    def normalize_frequency(cls, v):

        if v is None:
            return v

        value = str(v).strip().lower()

        freq_map = {

            # --------------------------------------------
            # Once daily
            # --------------------------------------------

            "od": "1-0-0",
            "once daily": "1-0-0",
            "once a day": "1-0-0",
            "one time a day": "1-0-0",

            # Hindi / Hinglish
            "din mein ek baar": "1-0-0",
            "din me ek baar": "1-0-0",
            "roz ek baar": "1-0-0",

            # --------------------------------------------
            # Twice daily
            # --------------------------------------------

            "bd": "1-0-1",
            "twice daily": "1-0-1",
            "twice a day": "1-0-1",
            "two times a day": "1-0-1",

            # Hindi / Hinglish
            "din mein do baar": "1-0-1",
            "din me do baar": "1-0-1",
            "roz do baar": "1-0-1",
            "subah shaam": "1-0-1",

            # --------------------------------------------
            # Thrice daily
            # --------------------------------------------

            "tid": "1-1-1",
            "thrice daily": "1-1-1",
            "three times": "1-1-1",
            "three times a day": "1-1-1",

            # Hindi / Hinglish
            "din mein teen baar": "1-1-1",
            "din me teen baar": "1-1-1",
            "roz teen baar": "1-1-1",

            # --------------------------------------------
            # Four times daily
            # --------------------------------------------

            "qid": "1-1-1-1",
            "four times": "1-1-1-1",
            "four times a day": "1-1-1-1",

            # Hindi / Hinglish
            "din mein chaar baar": "1-1-1-1",
            "din me chaar baar": "1-1-1-1",
            "roz chaar baar": "1-1-1-1",

            # --------------------------------------------
            # As needed
            # --------------------------------------------

            "sos": "SOS",
            "as needed": "SOS",
            "prn": "SOS",

            # Hindi / Hinglish
            "zarurat padne par": "SOS",
            "jarurat padne par": "SOS",
            "jab zarurat ho": "SOS",
        }

        return freq_map.get(
            value,
            v
        )


# ============================================================
# PRESCRIPTION
# ============================================================

class Prescription(BaseModel):

    medicines: List[Medicine] = Field(
        default_factory=list,
        description="List of prescribed medicines"
    )

    notes: Optional[str] = Field(
        None,
        description="Additional prescription notes"
    )

    rawTranscript: Optional[str] = Field(
        None,
        description="Original speech-to-text transcript"
    )


# ============================================================
# VALIDATION
# ============================================================

class ValidationIssue(BaseModel):

    field: str

    medicineIndex: int

    issue: str

    severity: Literal[
        "error",
        "warning",
        "info"
    ]


class ValidationResult(BaseModel):

    isValid: bool

    issues: List[ValidationIssue] = Field(
        default_factory=list
    )

    missingFields: List[str] = Field(
        default_factory=list
    )


# ============================================================
# EXTRACTION RESULT
# ============================================================

class ExtractionResult(BaseModel):

    prescription: Prescription

    validation: ValidationResult

    processingTimeMs: float

    sttConfidence: Optional[float] = None

    # --------------------------------------------
    # New clinical workflow status
    # --------------------------------------------

    status: PrescriptionStatus = Field(
        default=PrescriptionStatus.AI_DRAFT,
        description="Current prescription review status"
    )


# ============================================================
# PRESCRIPTION RESPONSE
# ============================================================

class PrescriptionResponse(BaseModel):

    success: bool

    data: Optional[ExtractionResult] = None

    error: Optional[str] = None


# ============================================================
# AUDIO PROCESS REQUEST
# ============================================================

class AudioProcessRequest(BaseModel):

    audioBase64: Optional[str] = None

    audioUrl: Optional[str] = None

    # Multilingual default
    language: str = "auto"

    returnTranscript: bool = True


# ============================================================
# DOCTOR REVIEW REQUEST
# ============================================================

class DoctorReviewRequest(BaseModel):

    prescription: Prescription

    status: PrescriptionStatus = (
        PrescriptionStatus.DOCTOR_EDITED
    )

    doctorId: Optional[str] = Field(
        None,
        description="Identifier of the reviewing doctor"
    )

    editReason: Optional[str] = Field(
        None,
        description="Reason for doctor modification"
    )


# ============================================================
# DOCTOR APPROVAL REQUEST
# ============================================================

class DoctorApprovalRequest(BaseModel):

    doctorId: Optional[str] = Field(
        None,
        description="Identifier of the approving doctor"
    )

    approvalNote: Optional[str] = Field(
        None,
        description="Optional approval note"
    )


# ============================================================
# AUDIT EVENT
# ============================================================

class PrescriptionAuditEvent(BaseModel):

    action: Literal[
        "AI_GENERATED",
        "DOCTOR_EDITED",
        "DOCTOR_APPROVED"
    ]

    doctorId: Optional[str] = None

    timestamp: Optional[str] = None

    details: Optional[str] = None