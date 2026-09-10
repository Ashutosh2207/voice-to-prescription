import streamlit as st
import requests
import json
import base64
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from audio_recorder_streamlit import audio_recorder


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Voice-to-Prescription | Medhant Lite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

API_BASE = "http://localhost:8000"

# Automatic evaluation dataset:
#   evaluation_dataset/
#       case_001.mp3
#       case_001.json
#       case_002.mp3
#       case_002.json
EVALUATION_DATASET_DIR = (
    Path(__file__).parent / "evaluation_dataset"
)

AUDIO_TIMEOUT = 360
TEXT_TIMEOUT = 180
EVALUATION_TIMEOUT = 60
REVIEW_TIMEOUT = 30


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}

.sub-header {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 2rem;
}

.transcript-box {
    background: #f8fbff;
    border: 1px solid #d7e7f7;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0 1rem 0;
}

.metric-title {
    font-size: 0.85rem;
    color: #666;
    font-weight: 600;
}

.quality-card {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}

.section-note {
    color: #666;
    font-size: 0.9rem;
}

.doctor-review-box {
    border: 2px solid #d7e7f7;
    border-radius: 12px;
    padding: 18px;
    margin-top: 10px;
    margin-bottom: 15px;
}

.approval-box {
    border: 2px solid #d9f0d9;
    border-radius: 12px;
    padding: 18px;
    margin-top: 15px;
}

.status-draft {
    padding: 8px 14px;
    border-radius: 8px;
    background: #fff4cc;
    display: inline-block;
    font-weight: 600;
}

.status-edited {
    padding: 8px 14px;
    border-radius: 8px;
    background: #dceeff;
    display: inline-block;
    font-weight: 600;
}

.status-approved {
    padding: 8px 14px;
    border-radius: 8px;
    background: #dff5df;
    display: inline-block;
    font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# API HEALTH CHECK
# ============================================================

def check_api_health() -> bool:
    try:
        response = requests.get(
            f"{API_BASE}/health",
            timeout=5
        )

        return response.status_code == 200

    except Exception:
        return False


# ============================================================
# PROCESS UPLOADED AUDIO
# ============================================================

def process_audio_file(
    audio_file,
    language: str = "auto"
) -> Optional[Dict]:

    try:
        files = {
            "file": (
                audio_file.name,
                audio_file.getvalue(),
                audio_file.type
                or "application/octet-stream"
            )
        }

        response = requests.post(
            f"{API_BASE}/process-audio/transformed",
            files=files,
            data={
                "language": language
            },
            timeout=AUDIO_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:
        st.error(
            f"⏱️ Audio processing exceeded "
            f"{AUDIO_TIMEOUT} seconds."
        )
        return None

    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Could not connect to API at {API_BASE}"
        )
        return None

    except requests.exceptions.HTTPError as e:
        st.error(
            f"❌ API HTTP Error: {e}"
        )
        return None

    except Exception as e:
        st.error(
            f"❌ API Error: {e}"
        )
        return None


# ============================================================
# PROCESS BASE64 AUDIO
# ============================================================

def process_base64_audio(
    base64_audio: str,
    language: str = "auto"
) -> Optional[Dict]:

    try:
        payload = {
            "audioBase64": base64_audio,
            "language": language,
            "returnTranscript": True
        }

        response = requests.post(
            f"{API_BASE}/process-base64/transformed",
            json=payload,
            timeout=AUDIO_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:
        st.error(
            f"⏱️ Voice processing exceeded "
            f"{AUDIO_TIMEOUT} seconds."
        )
        return None

    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Could not connect to API at {API_BASE}"
        )
        return None

    except requests.exceptions.HTTPError as e:
        st.error(
            f"❌ API HTTP Error: {e}"
        )
        return None

    except Exception as e:
        st.error(
            f"❌ API Error: {e}"
        )
        return None


# ============================================================
# PROCESS TEXT
# ============================================================

def process_text(
    text: str
) -> Optional[Dict]:

    try:
        response = requests.post(
            f"{API_BASE}/process-text/transformed",
            json={
                "text": text,
                "language": "auto"
            },
            timeout=TEXT_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:
        st.error(
            f"⏱️ Text processing exceeded "
            f"{TEXT_TIMEOUT} seconds."
        )
        return None

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Could not connect to API server."
        )
        return None

    except requests.exceptions.HTTPError as e:
        st.error(
            f"❌ API HTTP Error: {e}"
        )
        return None

    except Exception as e:
        st.error(
            f"❌ API Error: {e}"
        )
        return None


# ============================================================
# EVALUATE PIPELINE
# ============================================================

def evaluate_result(
    reference_transcript: str,
    predicted_transcript: str,
    reference_prescription: Dict[str, Any],
    predicted_prescription: Dict[str, Any]
) -> Optional[Dict]:

    payload = {
        "referenceTranscript": reference_transcript,
        "predictedTranscript": predicted_transcript,
        "referencePrescription": reference_prescription,
        "predictedPrescription": predicted_prescription,
    }

    try:
        response = requests.post(
            f"{API_BASE}/evaluate",
            json=payload,
            timeout=EVALUATION_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:
        st.error(
            "⏱️ Evaluation request timed out."
        )
        return None

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Could not connect to evaluation API."
        )
        return None

    except requests.exceptions.HTTPError as e:
        st.error(
            f"❌ Evaluation API error: {e}"
        )
        return None

    except Exception as e:
        st.error(
            f"❌ Evaluation error: {e}"
        )
        return None


# ============================================================
# GET RAW TRANSCRIPT
# ============================================================

def get_raw_transcript(
    result: Dict
) -> str:

    if not isinstance(result, dict):
        return ""

    transcript = result.get("rawTranscript")

    if isinstance(transcript, str):
        return transcript.strip()

    data = result.get("data")

    if isinstance(data, dict):
        transcript = data.get("rawTranscript")

        if isinstance(transcript, str):
            return transcript.strip()

    return ""


# ============================================================
# GET PRESCRIPTION
# ============================================================

def get_prescription(
    result: Dict
) -> list:

    if not isinstance(result, dict):
        return []

    prescription = result.get("prescription")

    if isinstance(prescription, list):
        return prescription

    data = result.get("data")

    if isinstance(data, dict):
        prescription = data.get("prescription")

        if isinstance(prescription, list):
            return prescription

    return []


# ============================================================
# CREATE PRESCRIPTION ID
# ============================================================

def get_prescription_id() -> str:

    existing_id = st.session_state.get(
        "prescription_id"
    )

    if existing_id:
        return existing_id

    new_id = str(uuid.uuid4())

    st.session_state[
        "prescription_id"
    ] = new_id

    return new_id


# ============================================================
# RESET EVALUATION STATE
# ============================================================

def clear_evaluation_state():

    st.session_state.pop(
        "last_evaluation",
        None
    )

    st.session_state.pop(
        "automatic_evaluation",
        None
    )

    st.session_state.pop(
        "automatic_evaluation_case",
        None
    )


# ============================================================
# RENDER MEDICINE CARD
# ============================================================

def render_medicine_card(
    medicine: Dict,
    index: int
):

    st.markdown(
        f"### 💊 Medicine {index}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Name**")

        st.write(
            medicine.get(
                "name",
                "Not specified"
            )
        )

        st.markdown("**Form**")

        st.write(
            medicine.get(
                "form",
                "Not specified"
            )
        )

        st.markdown("**Dosage**")

        st.write(
            medicine.get(
                "dosage",
                "Not specified"
            )
        )

        st.markdown("**Unit Per Time**")

        st.write(
            medicine.get(
                "unitPerTime",
                "Not specified"
            )
        )

    with col2:

        st.markdown("**Number Of Time**")

        st.write(
            medicine.get(
                "numberOfTime",
                "Not specified"
            )
        )

        st.markdown("**Remarks**")

        st.write(
            medicine.get(
                "remarks",
                "Not specified"
            )
        )

        st.markdown("**Duration**")

        st.write(
            medicine.get(
                "timesPerDay",
                "Not specified"
            )
        )

    st.divider()


# ============================================================
# RENDER STT METRICS
# ============================================================

def render_stt_metrics(
    result: Dict
):

    st.markdown(
        "## 🎯 Speech & Processing Metrics"
    )

    col1, col2, col3 = st.columns(3)

    confidence = result.get(
        "sttConfidence"
    )

    with col1:

        st.markdown(
            '<div class="metric-title">'
            'STT Confidence'
            '</div>',
            unsafe_allow_html=True
        )

        if confidence is not None:

            try:

                st.metric(
                    "Confidence",
                    f"{float(confidence):.2%}"
                )

            except Exception:

                st.metric(
                    "Confidence",
                    str(confidence)
                )

        else:

            st.metric(
                "Confidence",
                "N/A"
            )

    processing_time = result.get(
        "processingTimeMs"
    )

    with col2:

        st.markdown(
            '<div class="metric-title">'
            'Processing Time'
            '</div>',
            unsafe_allow_html=True
        )

        if processing_time is not None:

            try:

                seconds = (
                    float(processing_time)
                    / 1000
                )

                st.metric(
                    "Time",
                    f"{seconds:.2f} sec"
                )

            except Exception:

                st.metric(
                    "Time",
                    str(processing_time)
                )

        else:

            st.metric(
                "Time",
                "N/A"
            )

    prescription = get_prescription(
        result
    )

    with col3:

        st.markdown(
            '<div class="metric-title">'
            'Extracted Medicines'
            '</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Medicines",
            str(len(prescription))
        )


# ============================================================
# RENDER VALIDATION
# ============================================================

def render_validation(
    result: Dict
):

    validation = result.get(
        "validation"
    )

    st.markdown(
        "## 🛡️ Validation & Hallucination Guard"
    )

    if not isinstance(
        validation,
        dict
    ):

        st.info(
            "Validation information not available."
        )

        return

    is_valid = validation.get(
        "isValid"
    )

    issues = validation.get(
        "issues",
        []
    )

    missing_fields = validation.get(
        "missingFields",
        []
    )

    if is_valid is True:

        st.success(
            "✅ Prescription validation passed."
        )

    else:

        st.warning(
            "⚠️ Validation reported issues."
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Validation Issues",
            str(len(issues))
        )

    with col2:

        st.metric(
            "Missing Fields",
            str(len(missing_fields))
        )

    if issues:

        with st.expander(
            "🔎 View Validation Details",
            expanded=True
        ):

            for issue in issues:

                severity = issue.get(
                    "severity",
                    "warning"
                )

                field = issue.get(
                    "field",
                    "unknown"
                )

                message = issue.get(
                    "issue",
                    "Unknown issue"
                )

                if severity == "error":

                    st.error(
                        f"**{field}** — {message}"
                    )

                elif severity == "warning":

                    st.warning(
                        f"**{field}** — {message}"
                    )

                else:

                    st.info(
                        f"**{field}** — {message}"
                    )


# ============================================================
# DOCTOR REVIEW / EDIT
# ============================================================

def render_doctor_review(
    result: Dict
):

    prescription = get_prescription(
        result
    )

    if not prescription:

        st.warning(
            "No prescription is available for doctor review."
        )

        return

    st.markdown(
        "## 👨‍⚕️ Doctor Review & Approval"
    )

    current_status = st.session_state.get(
        "prescription_status",
        "AI_DRAFT"
    )

    prescription_id = get_prescription_id()

    if current_status == "AI_DRAFT":

        st.markdown(
            '<div class="status-draft">'
            '🟡 AI DRAFT'
            '</div>',
            unsafe_allow_html=True
        )

    elif current_status == "DOCTOR_EDITED":

        st.markdown(
            '<div class="status-edited">'
            '🔵 DOCTOR EDITED'
            '</div>',
            unsafe_allow_html=True
        )

    elif current_status == "APPROVED":

        st.markdown(
            '<div class="status-approved">'
            '🟢 APPROVED'
            '</div>',
            unsafe_allow_html=True
        )

    st.caption(
        f"Prescription ID: {prescription_id}"
    )

    # --------------------------------------------------------
    # APPROVED STATE
    # --------------------------------------------------------

    if current_status == "APPROVED":

        st.success(
            "✅ This prescription has been approved "
            "by the doctor."
        )

        approval_info = st.session_state.get(
            "approval_info",
            {}
        )

        if approval_info:

            st.write(
                f"**Doctor ID:** "
                f"{approval_info.get('doctorId', 'N/A')}"
            )

            st.write(
                f"**Approved At:** "
                f"{approval_info.get('timestamp', 'N/A')}"
            )

            if approval_info.get(
                "approvalNote"
            ):

                st.write(
                    f"**Approval Note:** "
                    f"{approval_info.get('approvalNote')}"
                )

        st.markdown(
            "### 🔒 Final Approved Prescription"
        )

        for index, medicine in enumerate(
            prescription,
            start=1
        ):

            render_medicine_card(
                medicine,
                index
            )

        return

    # --------------------------------------------------------
    # DOCTOR DETAILS
    # --------------------------------------------------------

    doctor_id = st.text_input(
        "Doctor ID / Identifier",
        value=st.session_state.get(
            "doctor_id",
            ""
        ),
        placeholder="Enter reviewing doctor's identifier",
        key="doctor_id_input"
    )

    edit_reason = st.text_input(
        "Edit Reason (optional)",
        value=st.session_state.get(
            "edit_reason",
            ""
        ),
        placeholder="Example: Corrected dosage after reviewing prescription",
        key="edit_reason_input"
    )

    st.markdown(
        '<div class="doctor-review-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### ✏️ Review Prescription"
    )

    st.info(
        "Doctor can edit the AI-generated fields before approval."
    )

    edited_prescription = []

    for index, medicine in enumerate(
        prescription
    ):

        st.markdown(
            f"#### 💊 Medicine {index + 1}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            name = st.text_input(
                "Medicine Name",
                value=str(
                    medicine.get(
                        "name",
                        ""
                    )
                ),
                key=f"doctor_name_{index}"
            )

        with col2:

            form = st.text_input(
                "Form",
                value=str(
                    medicine.get(
                        "form",
                        ""
                    )
                ),
                key=f"doctor_form_{index}"
            )

        with col3:

            dosage = st.text_input(
                "Dosage",
                value=str(
                    medicine.get(
                        "dosage",
                        ""
                    )
                ),
                key=f"doctor_dosage_{index}"
            )

        col4, col5, col6 = st.columns(3)

        with col4:

            frequency = st.text_input(
                "Frequency",
                value=str(
                    medicine.get(
                        "numberOfTime",
                        ""
                    )
                ),
                key=f"doctor_frequency_{index}"
            )

        with col5:

            duration = st.text_input(
                "Duration",
                value=str(
                    medicine.get(
                        "timesPerDay",
                        ""
                    )
                ),
                key=f"doctor_duration_{index}"
            )

        with col6:

            unit_per_time = st.text_input(
                "Unit Per Time",
                value=str(
                    medicine.get(
                        "unitPerTime",
                        ""
                    )
                ),
                key=f"doctor_unit_{index}"
            )

        remarks = st.text_input(
            "Instructions / Remarks",
            value=str(
                medicine.get(
                    "remarks",
                    ""
                )
            ),
            key=f"doctor_remarks_{index}"
        )

        edited_medicine = dict(
            medicine
        )

        edited_medicine[
            "name"
        ] = name.strip()

        edited_medicine[
            "form"
        ] = form.strip()

        edited_medicine[
            "dosage"
        ] = dosage.strip()

        edited_medicine[
            "numberOfTime"
        ] = frequency.strip()

        edited_medicine[
            "timesPerDay"
        ] = duration.strip()

        edited_medicine[
            "unitPerTime"
        ] = unit_per_time.strip()

        edited_medicine[
            "remarks"
        ] = remarks.strip()

        edited_prescription.append(
            edited_medicine
        )

        st.divider()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SAVE DOCTOR REVIEW
    # --------------------------------------------------------

    if current_status != "APPROVED":

        if st.button(
            "💾 Save Doctor Changes",
            type="primary",
            use_container_width=True
        ):

            if not doctor_id.strip():

                st.warning(
                    "Please enter Doctor ID / Identifier."
                )

                return

            payload = {
                "prescription": {
                    "medicines": []
                },
                "status": "DOCTOR_EDITED",
                "doctorId": doctor_id.strip(),
                "editReason": (
                    edit_reason.strip()
                    or None
                )
            }

            api_medicines = []

            for medicine in edited_prescription:

                route_value = medicine.get(
                    "route",
                    "unknown"
                )

                api_medicines.append({
                    "medicineName": medicine.get(
                        "name",
                        ""
                    ),
                    "dosage": (
                        medicine.get(
                            "dosage"
                        )
                        or None
                    ),
                    "frequency": (
                        medicine.get(
                            "numberOfTime"
                        )
                        or None
                    ),
                    "duration": (
                        medicine.get(
                            "timesPerDay"
                        )
                        or None
                    ),
                    "route": route_value,
                    "instructions": (
                        medicine.get(
                            "remarks"
                        )
                        or None
                    ),
                    "confidence": medicine.get(
                        "confidence"
                    )
                })

            payload[
                "prescription"
            ] = {
                "medicines": api_medicines,
                "notes": None,
                "rawTranscript": get_raw_transcript(
                    result
                )
            }

            try:

                response = requests.post(
                    f"{API_BASE}/prescription/review/"
                    f"{prescription_id}",
                    json=payload,
                    timeout=REVIEW_TIMEOUT
                )

                response.raise_for_status()

                review_response = response.json()

                if not review_response.get(
                    "success",
                    False
                ):

                    st.error(
                        review_response.get(
                            "error",
                            "Doctor review failed."
                        )
                    )

                    return

                st.session_state[
                    "doctor_edited_prescription"
                ] = edited_prescription

                st.session_state[
                    "prescription_status"
                ] = "DOCTOR_EDITED"

                st.session_state[
                    "doctor_review_response"
                ] = review_response

                st.success(
                    "✅ Doctor changes saved successfully."
                )

                st.rerun()

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ Doctor review request timed out."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to API."
                )

            except requests.exceptions.HTTPError as e:

                st.error(
                    f"❌ Doctor review API error: {e}"
                )

            except Exception as e:

                st.error(
                    f"❌ Doctor review error: {e}"
                )

    # --------------------------------------------------------
    # APPROVAL SECTION
    # --------------------------------------------------------

    if st.session_state.get(
        "prescription_status"
    ) == "DOCTOR_EDITED":

        st.markdown(
            '<div class="approval-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### ✅ Final Doctor Approval"
        )

        st.caption(
            "The doctor-reviewed prescription will be marked "
            "as final only after explicit approval."
        )

        approval_note = st.text_area(
            "Approval Note (optional)",
            placeholder=(
                "Optional note before final approval."
            ),
            key="approval_note"
        )

        if st.button(
            "✅ APPROVE PRESCRIPTION",
            type="primary",
            use_container_width=True
        ):

            if not doctor_id.strip():

                st.warning(
                    "Doctor ID is required for approval."
                )

                return

            payload = {
                "doctorId": doctor_id.strip(),
                "approvalNote": (
                    approval_note.strip()
                    or None
                )
            }

            try:

                response = requests.post(
                    f"{API_BASE}/prescription/approve/"
                    f"{prescription_id}",
                    json=payload,
                    timeout=REVIEW_TIMEOUT
                )

                response.raise_for_status()

                approval_response = response.json()

                if not approval_response.get(
                    "success",
                    False
                ):

                    st.error(
                        approval_response.get(
                            "error",
                            "Approval failed."
                        )
                    )

                    return

                st.session_state[
                    "prescription_status"
                ] = "APPROVED"

                st.session_state[
                    "approval_info"
                ] = {
                    "doctorId": approval_response.get(
                        "doctorId"
                    ),
                    "approvalNote": approval_response.get(
                        "approvalNote"
                    ),
                    "timestamp": approval_response.get(
                        "timestamp"
                    )
                }

                st.success(
                    "✅ Prescription approved successfully."
                )

                st.rerun()

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ Approval request timed out."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to approval API."
                )

            except requests.exceptions.HTTPError as e:

                st.error(
                    f"❌ Approval API error: {e}"
                )

            except Exception as e:

                st.error(
                    f"❌ Approval error: {e}"
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# AUTOMATIC DATASET EVALUATION
# ============================================================

def get_evaluation_cases():

    cases = []

    if not EVALUATION_DATASET_DIR.exists():
        return cases

    audio_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".flac"
    }

    for audio_path in sorted(
        EVALUATION_DATASET_DIR.iterdir()
    ):

        if audio_path.suffix.lower() not in audio_extensions:
            continue

        ground_truth_path = audio_path.with_suffix(
            ".json"
        )

        if not ground_truth_path.exists():
            continue

        cases.append({
            "name": audio_path.stem,
            "audio_path": audio_path,
            "ground_truth_path": ground_truth_path
        })

    return cases


def load_ground_truth(
    ground_truth_path: Path
) -> Optional[Dict[str, Any]]:

    try:

        with open(
            ground_truth_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if not isinstance(data, dict):

            st.error(
                "❌ Ground-truth JSON must be a JSON object."
            )

            return None

        reference_transcript = data.get(
            "referenceTranscript"
        )

        reference_prescription = data.get(
            "referencePrescription"
        )

        if not isinstance(
            reference_transcript,
            str
        ) or not reference_transcript.strip():

            st.error(
                "❌ `referenceTranscript` is missing "
                "or empty."
            )

            return None

        if not isinstance(
            reference_prescription,
            dict
        ):

            st.error(
                "❌ `referencePrescription` is missing "
                "or invalid."
            )

            return None

        return data

    except json.JSONDecodeError as e:

        st.error(
            f"❌ Invalid ground-truth JSON: {e}"
        )

        return None

    except Exception as e:

        st.error(
            f"❌ Could not load ground truth: {e}"
        )

        return None


def process_evaluation_audio(
    audio_path: Path
) -> Optional[Dict]:

    try:

        with open(
            audio_path,
            "rb"
        ) as f:

            audio_bytes = f.read()

        extension = audio_path.suffix.lower()

        if extension == ".mp3":
            mime_type = "audio/mpeg"

        elif extension == ".wav":
            mime_type = "audio/wav"

        elif extension == ".m4a":
            mime_type = "audio/mp4"

        elif extension == ".ogg":
            mime_type = "audio/ogg"

        elif extension == ".flac":
            mime_type = "audio/flac"

        else:
            mime_type = "application/octet-stream"

        files = {
            "file": (
                audio_path.name,
                audio_bytes,
                mime_type
            )
        }

        response = requests.post(
            f"{API_BASE}/process-audio/transformed",
            files=files,
            data={
                "language": "auto"
            },
            timeout=AUDIO_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:

        st.error(
            f"⏱️ Evaluation audio processing exceeded "
            f"{AUDIO_TIMEOUT} seconds."
        )

        return None

    except requests.exceptions.ConnectionError:

        st.error(
            f"❌ Could not connect to API at {API_BASE}"
        )

        return None

    except requests.exceptions.HTTPError as e:

        st.error(
            f"❌ Evaluation API HTTP Error: {e}"
        )

        return None

    except Exception as e:

        st.error(
            f"❌ Evaluation audio error: {e}"
        )

        return None


def run_automatic_evaluation(
    audio_path: Path,
    ground_truth: Dict[str, Any]
) -> Optional[Dict[str, Any]]:

    # --------------------------------------------------------
    # STEP 1 — RUN ACTUAL PIPELINE
    # --------------------------------------------------------

    prediction = process_evaluation_audio(
        audio_path
    )

    if not prediction:
        return None

    if not prediction.get(
        "success",
        False
    ):

        st.error(
            prediction.get(
                "error",
                "Prediction failed."
            )
        )

        return None

    # --------------------------------------------------------
    # STEP 2 — GET PREDICTION
    # --------------------------------------------------------

    predicted_transcript = get_raw_transcript(
        prediction
    )

    predicted_prescription = {
        "prescription": get_prescription(
            prediction
        )
    }

    # --------------------------------------------------------
    # STEP 3 — AUTOMATIC EVALUATION
    # --------------------------------------------------------

    evaluation_response = evaluate_result(

        reference_transcript=(
            ground_truth[
                "referenceTranscript"
            ]
        ),

        predicted_transcript=(
            predicted_transcript
        ),

        reference_prescription=(
            ground_truth[
                "referencePrescription"
            ]
        ),

        predicted_prescription=(
            predicted_prescription
        )
    )

    if not evaluation_response:
        return None

    if not evaluation_response.get(
        "success",
        False
    ):

        st.error(
            evaluation_response.get(
                "error",
                "Automatic evaluation failed."
            )
        )

        return None

    return {
        "prediction": prediction,
        "evaluation": evaluation_response.get(
            "evaluation",
            {}
        )
    }


def render_evaluation(
    result: Optional[Dict] = None
):

    st.markdown(
        "## 📊 Automatic Performance Evaluation"
    )

    st.markdown(
        '<div class="section-note">'
        'Doctor-verified ground truth is automatically '
        'loaded from the evaluation dataset. '
        'No manual transcript or JSON entry is required.'
        '</div>',
        unsafe_allow_html=True
    )

    cases = get_evaluation_cases()

    if not cases:

        st.warning(
            "⚠️ No evaluation cases found."
        )

        st.code(
            "evaluation_dataset/"
            "\n├── case_001.mp3"
            "\n├── case_001.json"
        )

        return

    # --------------------------------------------------------
    # CASE SELECTOR
    # --------------------------------------------------------

    case_names = [
        case["name"]
        for case in cases
    ]

    selected_name = st.selectbox(
        "🎧 Select Evaluation Case",
        case_names,
        key="evaluation_case_selector"
    )

    selected_case = next(
        case
        for case in cases
        if case["name"] == selected_name
    )

    audio_path = selected_case[
        "audio_path"
    ]

    ground_truth_path = selected_case[
        "ground_truth_path"
    ]

    # --------------------------------------------------------
    # FILE INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🎧 Evaluation Audio"
        )

        st.write(
            audio_path.name
        )

        try:

            with open(
                audio_path,
                "rb"
            ) as f:

                audio_bytes = f.read()

            extension = audio_path.suffix.lower()

            if extension == ".mp3":
                audio_format = "audio/mpeg"

            elif extension == ".wav":
                audio_format = "audio/wav"

            elif extension == ".m4a":
                audio_format = "audio/mp4"

            elif extension == ".ogg":
                audio_format = "audio/ogg"

            else:
                audio_format = "audio"

            st.audio(
                audio_bytes,
                format=audio_format
            )

        except Exception:

            st.warning(
                "Audio preview unavailable."
            )

    with col2:

        st.markdown(
            "### 📌 Ground Truth"
        )

        st.write(
            ground_truth_path.name
        )

        st.success(
            "✅ Automatically detected"
        )

    # --------------------------------------------------------
    # LOAD GROUND TRUTH
    # --------------------------------------------------------

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    if not ground_truth:
        return

    # --------------------------------------------------------
    # GROUND TRUTH PREVIEW
    # --------------------------------------------------------

    with st.expander(
        "🔍 View Ground Truth",
        expanded=False
    ):

        st.markdown(
            "#### Reference Transcript"
        )

        st.text_area(
            "Reference Transcript Preview",
            value=ground_truth[
                "referenceTranscript"
            ],
            height=140,
            disabled=True,
            key=f"ground_truth_transcript_preview_{selected_name}"
        )

        st.markdown(
            "#### Reference Prescription JSON"
        )

        st.json(
            ground_truth[
                "referencePrescription"
            ]
        )

    # --------------------------------------------------------
    # RUN AUTOMATIC EVALUATION
    # --------------------------------------------------------

    if st.button(
        "🚀 Run Automatic Evaluation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Running "
            "Whisper → Extraction → Validation → Evaluation..."
        ):

            result_bundle = (
                run_automatic_evaluation(
                    audio_path=audio_path,
                    ground_truth=ground_truth
                )
            )

        if result_bundle:

            st.session_state[
                "automatic_evaluation"
            ] = result_bundle

            st.session_state[
                "automatic_evaluation_case"
            ] = selected_name

            st.success(
                "✅ Automatic evaluation completed."
            )

    # --------------------------------------------------------
    # STORED RESULTS
    # --------------------------------------------------------

    result_bundle = st.session_state.get(
        "automatic_evaluation"
    )

    stored_case = st.session_state.get(
        "automatic_evaluation_case"
    )

    if not result_bundle:
        return

    if stored_case != selected_name:
        return

    prediction = result_bundle.get(
        "prediction",
        {}
    )

    evaluation = result_bundle.get(
        "evaluation",
        {}
    )

    # --------------------------------------------------------
    # SYSTEM PREDICTION
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 System Prediction"
    )

    predicted_transcript = get_raw_transcript(
        prediction
    )

    predicted_prescription = {
        "prescription": get_prescription(
            prediction
        )
    }

    with st.expander(
        "📝 Predicted Raw Transcript",
        expanded=False
    ):

        st.text_area(
            "Predicted Transcript",
            value=predicted_transcript,
            height=140,
            disabled=True,
            key=f"automatic_prediction_transcript_{selected_name}"
        )

    with st.expander(
        "📋 Predicted Prescription JSON",
        expanded=False
    ):

        st.json(
            predicted_prescription
        )

        st.code(
            json.dumps(
                predicted_prescription,
                indent=2
            ),
            language="json"
        )

    # --------------------------------------------------------
    # CORE METRICS
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Core Metrics"
    )

    wer = evaluation.get(
        "wer",
        {}
    )

    medicine_metrics = evaluation.get(
        "medicineMetrics",
        {}
    )

    hallucination = evaluation.get(
        "hallucination",
        {}
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "WER",
            f"{wer.get('werPercent', 0):.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{medicine_metrics.get('precisionPercent', 0):.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{medicine_metrics.get('recallPercent', 0):.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{medicine_metrics.get('f1Percent', 0):.2f}%"
        )

    with col5:

        st.metric(
            "Hallucination",
            f"{hallucination.get('hallucinationRatePercent', 0):.2f}%"
        )

    # --------------------------------------------------------
    # WER DETAILS
    # --------------------------------------------------------

    with st.expander(
        "🗣️ WER Details",
        expanded=False
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Reference Words",
                str(
                    wer.get(
                        "referenceWords",
                        0
                    )
                )
            )

        with c2:

            st.metric(
                "Hypothesis Words",
                str(
                    wer.get(
                        "hypothesisWords",
                        0
                    )
                )
            )

        with c3:

            st.metric(
                "Edit Distance",
                str(
                    wer.get(
                        "editDistance",
                        0
                    )
                )
            )

    # --------------------------------------------------------
    # MEDICINE METRICS
    # --------------------------------------------------------

    with st.expander(
        "💊 Medicine Extraction Metrics",
        expanded=False
    ):

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:

            st.metric(
                "Precision",
                f"{medicine_metrics.get('precisionPercent', 0):.2f}%"
            )

        with c2:

            st.metric(
                "Recall",
                f"{medicine_metrics.get('recallPercent', 0):.2f}%"
            )

        with c3:

            st.metric(
                "F1",
                f"{medicine_metrics.get('f1Percent', 0):.2f}%"
            )

        with c4:

            st.metric(
                "True Positive",
                str(
                    medicine_metrics.get(
                        "truePositive",
                        0
                    )
                )
            )

        with c5:

            st.metric(
                "False Positive",
                str(
                    medicine_metrics.get(
                        "falsePositive",
                        0
                    )
                )
            )

        with c6:

            st.metric(
                "False Negative",
                str(
                    medicine_metrics.get(
                        "falseNegative",
                        0
                    )
                )
            )

    # --------------------------------------------------------
    # FIELD METRICS
    # --------------------------------------------------------

    field_metrics = evaluation.get(
        "fieldMetrics",
        {}
    )

    if field_metrics:

        st.markdown(
            "### 🧪 Field-Level Accuracy"
        )

        rows = []

        for field, metrics in field_metrics.items():

            rows.append({

                "Field":
                    field,

                "Precision":
                    f"{metrics.get('precisionPercent', 0):.2f}%",

                "Recall":
                    f"{metrics.get('recallPercent', 0):.2f}%",

                "F1":
                    f"{metrics.get('f1Percent', 0):.2f}%",

                "TP":
                    metrics.get(
                        "truePositive",
                        0
                    ),

                "FP":
                    metrics.get(
                        "falsePositive",
                        0
                    ),

                "FN":
                    metrics.get(
                        "falseNegative",
                        0
                    )
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # HALLUCINATION
    # --------------------------------------------------------

    st.markdown(
        "### 🚨 Hallucination Analysis"
    )

    hallucinated = hallucination.get(
        "hallucinatedMedicines",
        []
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Hallucination Rate",
            f"{hallucination.get('hallucinationRatePercent', 0):.2f}%"
        )

    with c2:

        st.metric(
            "Hallucinated Medicines",
            str(
                hallucination.get(
                    "hallucinatedCount",
                    0
                )
            )
        )

    with c3:

        st.metric(
            "Predicted Medicines",
            str(
                hallucination.get(
                    "totalPredictedMedicines",
                    0
                )
            )
        )

    if hallucinated:

        st.error(
            "Unsupported predicted medicines:"
        )

        for medicine in hallucinated:

            st.write(
                f"❌ {str(medicine).upper()}"
            )

    else:

        st.success(
            "✅ No hallucinated medicines detected "
            "against the supplied ground truth."
        )

    with st.expander(
        "🧾 Raw Evaluation JSON",
        expanded=False
    ):

        st.json(
            evaluation
        )


# ============================================================
# RENDER RESULTS
# ============================================================

def render_results(
    result: Dict
):

    if not result:

        st.error(
            "No response received from API."
        )

        return

    if not result.get(
        "success",
        False
    ):

        st.error(
            result.get(
                "error",
                "API processing failed."
            )
        )

        return

    # ========================================================
    # RAW TRANSCRIPT
    # ========================================================

    raw_transcript = get_raw_transcript(
        result
    )

    with st.expander(
        "🎙️ Raw Transcript — Click to View",
        expanded=False
    ):

        if raw_transcript:

            st.markdown(
                "### 📝 Speech-to-Text"
            )

            st.caption(
                "This is the text generated by Whisper."
            )

            st.markdown(
                '<div class="transcript-box">',
                unsafe_allow_html=True
            )

            st.text_area(
                "What the system heard",
                value=raw_transcript,
                height=180,
                disabled=True,
                label_visibility="collapsed"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "No raw transcript was returned by the API."
            )

    # ========================================================
    # STT METRICS
    # ========================================================

    render_stt_metrics(
        result
    )

    st.divider()

    # ========================================================
    # VALIDATION
    # ========================================================

    render_validation(
        result
    )

    st.divider()

    # ========================================================
    # AI PRESCRIPTION
    # ========================================================

    prescription = get_prescription(
        result
    )

    st.success(
        f"✅ {len(prescription)} medicine(s) extracted"
    )

    if prescription:

        st.markdown(
            "## 🤖 AI Generated Prescription"
        )

        for index, medicine in enumerate(
            prescription,
            start=1
        ):

            render_medicine_card(
                medicine,
                index
            )

    else:

        st.warning(
            "No medicines extracted."
        )

    # ========================================================
    # JSON OUTPUT
    # ========================================================

    final_output = {
        "prescription":
            prescription
    }

    st.markdown(
        "## 📋 AI Generated JSON"
    )

    st.json(
        final_output
    )

    json_string = json.dumps(
        final_output,
        indent=2
    )

    st.download_button(
        label="📥 Download AI JSON",
        data=json_string,
        file_name="prescription.json",
        mime="application/json"
    )

    st.divider()

    # ========================================================
    # DOCTOR REVIEW
    # ========================================================

    render_doctor_review(
        result
    )

    st.divider()

    # ========================================================
    # PERFORMANCE EVALUATION
    # ========================================================

    render_evaluation(
        result
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-header">'
        '🩺 Voice-to-Prescription'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-header">'
        'Medhant Lite — Voice → STT → Extraction → '
        'Validation → Doctor Review → Approval'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # API HEALTH
    # ========================================================

    api_healthy = check_api_health()

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.markdown(
            "## ⚙️ Settings"
        )

        if api_healthy:

            st.success(
                "✅ API Connected"
            )

        else:

            st.error(
                "❌ API Disconnected"
            )

        st.divider()

        st.markdown(
            "### 🎤 Input Mode"
        )

        input_mode = st.radio(
            "Choose input type",
            [
                "🎤 Upload Audio",
                "🎙️ Record Audio",
                "⌨️ Text Input"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        st.markdown(
            "### ℹ️ Pipeline"
        )

        st.markdown(
            """
**Audio**

Audio → Faster-Whisper → Transcript

**Extraction**

Transcript → Ministral → Structured Prescription

**Validation**

Medical Dictionary → Fuzzy Matching → Evidence Check

**Clinical Workflow**

AI Draft → Doctor Edit → Doctor Approval

**Evaluation**

Evaluation Audio → Prediction
→ Doctor-Verified Ground Truth
→ WER → Precision → Recall → F1 → Hallucination
"""
        )

    # ========================================================
    # API NOT RUNNING
    # ========================================================

    if not api_healthy:

        st.error(
            "⚠️ API server is not running."
        )

        st.code(
            "python -m uvicorn app.api:app "
            "--host 127.0.0.1 --port 8000"
        )

        st.stop()

    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.markdown(
        "## 📥 Input Prescription"
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    if input_mode == "🎤 Upload Audio":

        audio_file = st.file_uploader(
            "Upload audio file",
            type=[
                "wav",
                "mp3",
                "m4a",
                "ogg",
                "flac"
            ]
        )

        if audio_file:

            st.audio(
                audio_file,
                format=audio_file.type
            )

            if st.button(
                "🔄 Process Audio",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Processing Audio → Whisper → "
                    "Extraction → Validation..."
                ):

                    result = process_audio_file(
                        audio_file,
                        language="auto"
                    )

                if result:

                    st.session_state[
                        "last_result"
                    ] = result

                    st.session_state[
                        "prescription_id"
                    ] = str(
                        uuid.uuid4()
                    )

                    st.session_state[
                        "prescription_status"
                    ] = "AI_DRAFT"

                    st.session_state.pop(
                        "doctor_edited_prescription",
                        None
                    )

                    st.session_state.pop(
                        "approval_info",
                        None
                    )

                    clear_evaluation_state()

                    st.rerun()

    # ========================================================
    # RECORD
    # ========================================================

    elif input_mode == "🎙️ Record Audio":

        st.markdown(
            "### 🎙️ Record Prescription"
        )

        st.caption(
            "Speak clearly and stop recording when finished."
        )

        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            key="voice_recorder"
        )

        if audio_bytes:

            st.audio(
                audio_bytes,
                format="audio/wav"
            )

            st.success(
                f"✅ Recording ready "
                f"({len(audio_bytes)} bytes)"
            )

            if st.button(
                "🔄 Process Recording",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Processing Recording → Whisper → "
                    "Extraction → Validation..."
                ):

                    encoded_audio = (
                        base64
                        .b64encode(
                            audio_bytes
                        )
                        .decode(
                            "utf-8"
                        )
                    )

                    result = process_base64_audio(
                        encoded_audio,
                        language="auto"
                    )

                if result:

                    st.session_state[
                        "last_result"
                    ] = result

                    st.session_state[
                        "prescription_id"
                    ] = str(
                        uuid.uuid4()
                    )

                    st.session_state[
                        "prescription_status"
                    ] = "AI_DRAFT"

                    st.session_state.pop(
                        "doctor_edited_prescription",
                        None
                    )

                    st.session_state.pop(
                        "approval_info",
                        None
                    )

                    clear_evaluation_state()

                    st.rerun()

    # ========================================================
    # TEXT INPUT
    # ========================================================

    else:

        text_input = st.text_area(
            "Type or paste prescription text",
            placeholder=(
                "Example: Aspirin 100 mg tablet, "
                "morning and evening after food "
                "for 2 days."
            ),
            height=150
        )

        if st.button(
            "🔄 Process Text",
            type="primary",
            use_container_width=True
        ):

            if not text_input.strip():

                st.warning(
                    "Please enter some text."
                )

            else:

                with st.spinner(
                    "Extracting prescription..."
                ):

                    result = process_text(
                        text_input
                    )

                if result:

                    if not result.get(
                        "rawTranscript"
                    ):

                        result[
                            "rawTranscript"
                        ] = text_input.strip()

                    st.session_state[
                        "last_result"
                    ] = result

                    st.session_state[
                        "prescription_id"
                    ] = str(
                        uuid.uuid4()
                    )

                    st.session_state[
                        "prescription_status"
                    ] = "AI_DRAFT"

                    st.session_state.pop(
                        "doctor_edited_prescription",
                        None
                    )

                    st.session_state.pop(
                        "approval_info",
                        None
                    )

                    clear_evaluation_state()

                    st.rerun()

    # ========================================================
    # RESULTS
    # ========================================================

    if "last_result" in st.session_state:

        st.divider()

        st.markdown(
            "## 📤 Results"
        )

        render_results(
            st.session_state[
                "last_result"
            ]
        )

        st.divider()

        # ====================================================
        # ACTION BUTTONS
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔄 Process Another",
                use_container_width=True
            ):

                st.session_state.pop(
                    "last_result",
                    None
                )

                st.session_state.pop(
                    "prescription_id",
                    None
                )

                st.session_state.pop(
                    "prescription_status",
                    None
                )

                st.session_state.pop(
                    "doctor_edited_prescription",
                    None
                )

                st.session_state.pop(
                    "approval_info",
                    None
                )

                clear_evaluation_state()

                st.rerun()

        with col2:

            if st.button(
                "🧹 Clear Evaluation",
                use_container_width=True
            ):

                clear_evaluation_state()

                st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
