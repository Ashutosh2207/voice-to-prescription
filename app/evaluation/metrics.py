import re
from typing import List, Dict, Any, Optional


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text).lower().strip()

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


def normalize_medicine_name(
    name: Optional[str]
) -> str:

    if not name:
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(name).lower()
    )


# =========================================================
# WORD TOKENIZATION
# =========================================================

def tokenize(
    text: Optional[str]
) -> List[str]:

    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


# =========================================================
# EDIT DISTANCE
# =========================================================

def edit_distance(
    reference: List[str],
    hypothesis: List[str]
) -> int:

    n = len(reference)
    m = len(hypothesis)

    if n == 0:
        return m

    if m == 0:
        return n

    previous = list(range(m + 1))

    for i in range(1, n + 1):

        current = [i]

        for j in range(1, m + 1):

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                +
                (
                    0
                    if reference[i - 1]
                    == hypothesis[j - 1]
                    else 1
                )
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[m]


# =========================================================
# WER
# =========================================================

def calculate_wer(
    reference: str,
    hypothesis: str
) -> Dict[str, Any]:

    reference_words = tokenize(
        reference
    )

    hypothesis_words = tokenize(
        hypothesis
    )

    reference_count = len(
        reference_words
    )

    if reference_count == 0:

        wer = (
            0.0
            if not hypothesis_words
            else 1.0
        )

        return {
            "wer": round(
                wer,
                4
            ),
            "werPercent": round(
                wer * 100,
                2
            ),
            "referenceWords": 0,
            "hypothesisWords": len(
                hypothesis_words
            ),
            "editDistance": len(
                hypothesis_words
            ),
        }

    distance = edit_distance(
        reference_words,
        hypothesis_words
    )

    wer = (
        distance /
        reference_count
    )

    return {
        "wer": round(
            wer,
            4
        ),
        "werPercent": round(
            wer * 100,
            2
        ),
        "referenceWords": reference_count,
        "hypothesisWords": len(
            hypothesis_words
        ),
        "editDistance": distance,
    }


# =========================================================
# EXTRACT MEDICINE OBJECTS
# =========================================================

def extract_medicines(
    prescription_data: Any
) -> List[Dict[str, Any]]:

    if not prescription_data:
        return []

    medicines = None

    if isinstance(
        prescription_data,
        dict
    ):

        # Internal schema
        medicines = (
            prescription_data.get(
                "medicines"
            )
        )

        # Medhant Lite schema
        if medicines is None:

            medicines = (
                prescription_data.get(
                    "prescription"
                )
            )

    if not isinstance(
        medicines,
        list
    ):
        return []

    result = []

    for medicine in medicines:

        if not isinstance(
            medicine,
            dict
        ):
            continue

        result.append(
            medicine
        )

    return result


# =========================================================
# GET FIELD VALUE
# =========================================================

def get_field(
    medicine: Dict[str, Any],
    field: str
) -> str:

    field_map = {

        "medicineName": [
            "medicineName",
            "name"
        ],

        "dosage": [
            "dosage"
        ],

        "form": [
            "form"
        ],

        "frequency": [
            "frequency",
            "numberOfTime"
        ],

        "duration": [
            "duration",
            "timesPerDay"
        ],

        "instructions": [
            "instructions",
            "remarks"
        ],

        "route": [
            "route"
        ],

        "unitPerTime": [
            "unitPerTime"
        ],
    }

    possible_fields = (
        field_map.get(
            field,
            [field]
        )
    )

    for key in possible_fields:

        value = medicine.get(
            key
        )

        if value is not None:

            value = str(
                value
            ).strip()

            if value:
                return value

    return ""


# =========================================================
# NORMALIZE FIELD
# =========================================================

def normalize_field(
    value: str,
    field: str
) -> str:

    if not value:
        return ""

    value = normalize_text(
        value
    )

    # -----------------------------------------------------
    # Medicine name
    # -----------------------------------------------------

    if field == "medicineName":

        return normalize_medicine_name(
            value
        )

    # -----------------------------------------------------
    # Dosage
    # -----------------------------------------------------

    if field == "dosage":

        value = value.replace(
            " ",
            ""
        )

        return value

    # -----------------------------------------------------
    # Frequency
    # -----------------------------------------------------

    if field == "frequency":

        frequency_map = {

            "once daily": "1-0-0",
            "once a day": "1-0-0",
            "od": "1-0-0",

            "twice daily": "1-0-1",
            "twice a day": "1-0-1",
            "bd": "1-0-1",

            "three times daily": "1-1-1",
            "three times a day": "1-1-1",
            "thrice daily": "1-1-1",
            "tid": "1-1-1",

            "four times daily": "1-1-1-1",
            "four times a day": "1-1-1-1",
            "qid": "1-1-1-1",

            "as needed": "sos",
            "prn": "sos",
            "sos": "sos",
        }

        return frequency_map.get(
            value,
            value
        )

    # -----------------------------------------------------
    # Instructions
    # -----------------------------------------------------

    if field == "instructions":

        replacements = {
            "after meals": "after food",
            "after eating": "after food",
            "before meals": "before food",
            "before eating": "before food",
        }

        return replacements.get(
            value,
            value
        )

    return value


# =========================================================
# MEDICINE MATCHING
# =========================================================

def medicine_key(
    medicine: Dict[str, Any]
) -> str:

    return normalize_medicine_name(
        get_field(
            medicine,
            "medicineName"
        )
    )


# =========================================================
# FIELD METRICS
# =========================================================

def calculate_field_metrics(
    reference_medicines: List[Dict[str, Any]],
    predicted_medicines: List[Dict[str, Any]],
    field: str
) -> Dict[str, Any]:

    reference_map = {}

    for medicine in reference_medicines:

        key = medicine_key(
            medicine
        )

        if key:
            reference_map[key] = (
                medicine
            )

    predicted_map = {}

    for medicine in predicted_medicines:

        key = medicine_key(
            medicine
        )

        if key:
            predicted_map[key] = (
                medicine
            )

    true_positive = 0
    false_positive = 0
    false_negative = 0

    details = []

    # -----------------------------------------------------
    # Compare predicted medicines
    # -----------------------------------------------------

    for key, predicted in predicted_map.items():

        predicted_value = normalize_field(
            get_field(
                predicted,
                field
            ),
            field
        )

        if key not in reference_map:

            if predicted_value:

                false_positive += 1

                details.append({
                    "medicine": key,
                    "status": "false_positive",
                    "reference": "",
                    "predicted": predicted_value
                })

            continue

        reference = reference_map[key]

        reference_value = normalize_field(
            get_field(
                reference,
                field
            ),
            field
        )

        if (
            reference_value
            and
            predicted_value
            and
            reference_value
            == predicted_value
        ):

            true_positive += 1

            details.append({
                "medicine": key,
                "status": "correct",
                "reference": reference_value,
                "predicted": predicted_value
            })

        elif (
            predicted_value
        ):

            false_positive += 1
            false_negative += 1

            details.append({
                "medicine": key,
                "status": "incorrect",
                "reference": reference_value,
                "predicted": predicted_value
            })

        elif (
            reference_value
        ):

            false_negative += 1

            details.append({
                "medicine": key,
                "status": "missing",
                "reference": reference_value,
                "predicted": ""
            })

    # -----------------------------------------------------
    # Reference medicines completely missing
    # -----------------------------------------------------

    for key, reference in reference_map.items():

        if key in predicted_map:
            continue

        reference_value = normalize_field(
            get_field(
                reference,
                field
            ),
            field
        )

        if reference_value:

            false_negative += 1

            details.append({
                "medicine": key,
                "status": "missing_medicine",
                "reference": reference_value,
                "predicted": ""
            })

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    precision_denominator = (
        true_positive
        + false_positive
    )

    recall_denominator = (
        true_positive
        + false_negative
    )

    if precision_denominator > 0:

        precision = (
            true_positive
            /
            precision_denominator
        )

    else:

        precision = 0.0

    if recall_denominator > 0:

        recall = (
            true_positive
            /
            recall_denominator
        )

    else:

        recall = 0.0

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    return {

        "precision": round(
            precision,
            4
        ),

        "precisionPercent": round(
            precision * 100,
            2
        ),

        "recall": round(
            recall,
            4
        ),

        "recallPercent": round(
            recall * 100,
            2
        ),

        "f1": round(
            f1,
            4
        ),

        "f1Percent": round(
            f1 * 100,
            2
        ),

        "truePositive": true_positive,

        "falsePositive": false_positive,

        "falseNegative": false_negative,

        "details": details
    }


# =========================================================
# MEDICINE NAME METRICS
# =========================================================

def calculate_medicine_metrics(
    reference_medicines: List[Dict[str, Any]],
    predicted_medicines: List[Dict[str, Any]]
) -> Dict[str, Any]:

    reference_names = {
        medicine_key(m)
        for m in reference_medicines
        if medicine_key(m)
    }

    predicted_names = {
        medicine_key(m)
        for m in predicted_medicines
        if medicine_key(m)
    }

    true_positive = len(
        reference_names
        &
        predicted_names
    )

    false_positive = len(
        predicted_names
        -
        reference_names
    )

    false_negative = len(
        reference_names
        -
        predicted_names
    )

    if (
        true_positive
        + false_positive
    ) > 0:

        precision = (
            true_positive
            /
            (
                true_positive
                + false_positive
            )
        )

    else:

        precision = 0.0

    if (
        true_positive
        + false_negative
    ) > 0:

        recall = (
            true_positive
            /
            (
                true_positive
                + false_negative
            )
        )

    else:

        recall = 0.0

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    return {

        "precision": round(
            precision,
            4
        ),

        "precisionPercent": round(
            precision * 100,
            2
        ),

        "recall": round(
            recall,
            4
        ),

        "recallPercent": round(
            recall * 100,
            2
        ),

        "f1": round(
            f1,
            4
        ),

        "f1Percent": round(
            f1 * 100,
            2
        ),

        "truePositive": true_positive,

        "falsePositive": false_positive,

        "falseNegative": false_negative,
    }


# =========================================================
# HALLUCINATION RATE
# =========================================================

def calculate_hallucination_rate(
    reference_medicines: List[Dict[str, Any]],
    predicted_medicines: List[Dict[str, Any]]
) -> Dict[str, Any]:

    reference_names = {
        medicine_key(m)
        for m in reference_medicines
        if medicine_key(m)
    }

    predicted_names = {
        medicine_key(m)
        for m in predicted_medicines
        if medicine_key(m)
    }

    hallucinated = sorted(
        predicted_names
        -
        reference_names
    )

    total_predicted = len(
        predicted_names
    )

    if total_predicted > 0:

        rate = (
            len(hallucinated)
            /
            total_predicted
        )

    else:

        rate = 0.0

    return {

        "hallucinationRate": round(
            rate,
            4
        ),

        "hallucinationRatePercent": round(
            rate * 100,
            2
        ),

        "hallucinatedMedicines":
            hallucinated,

        "hallucinatedCount":
            len(hallucinated),

        "totalPredictedMedicines":
            total_predicted,
    }


# =========================================================
# COMPLETE EVALUATION
# =========================================================

def evaluate_pipeline(
    reference_transcript: str,
    predicted_transcript: str,
    reference_prescription: Any,
    predicted_prescription: Any
) -> Dict[str, Any]:

    # -----------------------------------------------------
    # WER
    # -----------------------------------------------------

    wer_result = calculate_wer(
        reference_transcript,
        predicted_transcript
    )

    # -----------------------------------------------------
    # Extract medicine arrays
    # -----------------------------------------------------

    reference_medicines = extract_medicines(
        reference_prescription
    )

    predicted_medicines = extract_medicines(
        predicted_prescription
    )

    # -----------------------------------------------------
    # Medicine name metrics
    # -----------------------------------------------------

    medicine_metrics = (
        calculate_medicine_metrics(
            reference_medicines,
            predicted_medicines
        )
    )

    # -----------------------------------------------------
    # Field metrics
    # -----------------------------------------------------

    fields = [
        "dosage",
        "form",
        "frequency",
        "duration",
        "instructions",
        "route",
        "unitPerTime",
    ]

    field_metrics = {}

    for field in fields:

        field_metrics[field] = (
            calculate_field_metrics(
                reference_medicines,
                predicted_medicines,
                field
            )
        )

    # -----------------------------------------------------
    # Hallucination
    # -----------------------------------------------------

    hallucination = (
        calculate_hallucination_rate(
            reference_medicines,
            predicted_medicines
        )
    )

    # -----------------------------------------------------
    # Return
    # -----------------------------------------------------

    return {

        "wer": wer_result,

        "medicineMetrics":
            medicine_metrics,

        "fieldMetrics":
            field_metrics,

        "hallucination":
            hallucination,

        "referenceMedicineCount":
            len(reference_medicines),

        "predictedMedicineCount":
            len(predicted_medicines),

        "referenceMedicines": [
            medicine_key(m)
            for m in reference_medicines
            if medicine_key(m)
        ],

        "predictedMedicines": [
            medicine_key(m)
            for m in predicted_medicines
            if medicine_key(m)
        ],
    }