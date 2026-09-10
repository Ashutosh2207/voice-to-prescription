import json
import re
import httpx

from typing import List, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.models.schemas import Route
from app.pipeline.nlp_extractor import ExtractedMedicine


# ============================================================
# OLLAMA CONFIG
# ============================================================

@dataclass
class OllamaConfig:

    base_url: str = "http://localhost:11434"

    model: str = "ministral-3:3b"

    temperature: float = 0.1

    timeout: float = 300.0

    max_retries: int = 2


# ============================================================
# OLLAMA PRESCRIPTION EXTRACTOR
# ============================================================

class OllamaPrescriptionExtractor:
    """
    Local LLM prescription extractor.

    Pipeline:

        Raw Transcript
              ↓
        Ministral 3:3b
              ↓
        Strict JSON extraction
              ↓
        Medicine-name cleanup
              ↓
        ASR correction / transcript evidence
              ↓
        Field normalization
              ↓
        Duplicate protection
              ↓
        ExtractedMedicine objects
    """

    # ========================================================
    # LOCAL MEDICAL VOCABULARY
    # ========================================================

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

    # ========================================================
    # NARROW ASR CORRECTIONS
    # ========================================================
    #
    # These are NOT generic "invent a medicine" rules.
    # They are narrow speech-recognition corrections observed
    # in prescription dictation.
    #
    # IMPORTANT:
    # A correction is accepted only when the surrounding
    # transcript contains prescription context.
    #

    ASR_MEDICINE_ALIASES = {

        # Current observed Whisper error:
        # "Scribe as 300 mg tablet..."
        #
        # Only use with prescription context:
        # dosage + form + prescription instruction.
        "scribe": "aspirin",

        # Common minor ASR spelling errors.
        "amoxiling": "amoxicillin",
        "amoxicilin": "amoxicillin",
        "amoxicillin": "amoxicillin",

        "paracetmol": "paracetamol",
        "paracetamol": "paracetamol",

        "ibuprofren": "ibuprofen",
        "ibuprofin": "ibuprofen",

        "azithromicin": "azithromycin",
        "azithromycin": "azithromycin",

        "cetirizene": "cetirizine",
        "cetirizine": "cetirizine",
    }

    # ========================================================
    # MEDICINE-NAME JUNK TOKENS
    # ========================================================

    MEDICINE_NAME_JUNK = {
        "mg",
        "g",
        "mcg",
        "ml",
        "l",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "pill",
        "pills",
        "syrup",
        "injection",
        "cream",
        "ointment",
        "drops",
        "drop",
        "dose",
        "dosage",
        "medicine",
        "medication",
        "take",
        "give",
        "given",
        "after",
        "before",
        "food",
        "meal",
        "meals",
        "morning",
        "afternoon",
        "evening",
        "night",
    }

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        config: OllamaConfig = None
    ):

        self.config = (
            config
            or OllamaConfig()
        )

        self.client = httpx.Client(
            timeout=self.config.timeout
        )

        self.system_prompt = (
            self._get_system_prompt()
        )

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def _get_system_prompt(self) -> str:

        return """
You are a STRICT medical prescription extraction engine.

Your ONLY task is to extract prescription information explicitly
supported by the supplied transcript.

The transcript may contain:
- English
- Hindi
- Hinglish
- Roman Hindi
- Speech-recognition spelling errors

============================================================
CRITICAL SAFETY RULES
============================================================

1. NEVER invent a medicine.

2. NEVER add a medicine because it is medically common.

3. NEVER infer a medicine from dosage alone.

4. NEVER infer a medicine from frequency alone.

5. NEVER assign a dosage from another medicine.

6. NEVER assign duration from another medicine.

7. NEVER change a medicine into another medicine.

8. NEVER replace a transcript-supported form.

9. NEVER replace CAPSULE with TABLET.

10. NEVER replace TABLET with CAPSULE.

11. NEVER put dosage inside medicineName.

12. NEVER put form inside medicineName.

13. NEVER put instructions inside medicineName.

14. medicineName must contain ONLY the medicine name.

15. Example:
       WRONG:
       "PARACETAMOL 500MG"

       CORRECT:
       medicineName = "PARACETAMOL"
       dosage = "500MG"

16. Example:
       WRONG:
       "AMOXICILLIN 500MG CAPSULE"

       CORRECT:
       medicineName = "AMOXICILLIN"
       dosage = "500MG"
       form = "CAPSULE"

17. If a field is not explicitly available, return null.

18. Extract ALL medicines explicitly mentioned.

19. Do not duplicate a medicine because it appears elsewhere
    in the transcript.

20. Preserve independent prescriptions even when the same medicine
    is legitimately prescribed more than once with different
    instructions. Do not merge merely because the name is equal.

21. Do not output commentary.

22. Return JSON ONLY.

============================================================
OUTPUT FORMAT
============================================================

{
  "medicines": [
    {
      "medicineName": "ASPIRIN",
      "dosage": "100MG",
      "frequency": "1-0-1-0",
      "duration": "2 DAYS",
      "route": "oral",
      "instructions": "AFTER FOOD",
      "form": "TABLET",
      "unitPerTime": "1",
      "confidence": 0.95
    }
  ],
  "notes": null
}

============================================================
MEDICINE NAME
============================================================

medicineName:
- ONLY medicine name
- no dosage
- no form
- no frequency
- no duration
- no instructions
- no explanatory text

Valid:

"PARACETAMOL"

"AMOXICILLIN"

"ASPIRIN"

Invalid:

"PARACETAMOL 500MG"

"AMOXICILLIN CAPSULE"

"ASPIRIN 300 MG TABLET"

"PARACETAMOL AFTER FOOD"

============================================================
FORM
============================================================

tablet / tablets / pill / pills
→ TABLET

capsule / capsules
→ CAPSULE

syrup
→ SYRUP

injection
→ INJECTION

cream
→ CREAM

ointment
→ OINTMENT

drops
→ DROPS

If explicit form exists, preserve it exactly.

If absent:
UNKNOWN

============================================================
DOSAGE
============================================================

500 mg
500mg
500 milligram
→ 500MG

300 mg
300mg
→ 300MG

5 ml
5ml
→ 5ML

10 ml
10ml
→ 10ML

Supported:
mg
g
mcg
ml
l

Do not invent dosage.

============================================================
FREQUENCY
============================================================

once daily
once a day
one time a day
→ 1-0-0

twice daily
twice a day
two times a day
→ 1-0-1

three times daily
three times a day
three times per day
→ 1-1-1

four times daily
four times a day
→ 1-1-1-1

morning
in the morning
→ 1-0-0-0

morning and evening
morning + evening
→ 1-0-1-0

morning + afternoon
→ 1-1-0-0

morning afternoon evening
→ 1-1-1-0

morning afternoon evening night
→ 1-1-1-1

night
at night
→ 0-0-0-1

as needed
when required
→ SOS

============================================================
HINDI / HINGLISH FREQUENCY
============================================================

din mein ek baar
din me ek baar
roz ek baar
→ 1-0-0

din mein do baar
din me do baar
roz do baar
→ 1-0-1

din mein teen baar
din me teen baar
roz teen baar
→ 1-1-1

din mein chaar baar
din me chaar baar
roz chaar baar
→ 1-1-1-1

subah
→ 1-0-0-0

subah shaam
subah aur shaam
→ 1-0-1-0

subah dopahar
→ 1-1-0-0

subah dopahar shaam
→ 1-1-1-0

subah dopahar shaam raat
→ 1-1-1-1

raat
raat ko
→ 0-0-0-1

zarurat padne par
jarurat padne par
jab zarurat ho
→ SOS

============================================================
FOOD INSTRUCTIONS
============================================================

after food
after meals
after eating
with food
→ AFTER FOOD

before food
before meals
before eating
before breakfast
on an empty stomach
→ BEFORE FOOD

khane ke baad
khana khane ke baad
meal ke baad
meals ke baad
→ AFTER FOOD

khane se pehle
khana khane se pehle
meal se pehle
→ BEFORE FOOD

khali pet
subah khali pet
→ BEFORE FOOD

If absent:
null

============================================================
DURATION
============================================================

2 days → 2 DAYS
3 days → 3 DAYS
5 days → 5 DAYS
7 days → 7 DAYS

1 week → 1 WEEK
2 weeks → 2 WEEKS

1 month → 1 MONTH
2 months → 2 MONTHS

Hindi:

2 din → 2 DAYS
do din → 2 DAYS
5 din → 5 DAYS
paanch din → 5 DAYS
7 din → 7 DAYS

1 hafta → 1 WEEK
ek hafta → 1 WEEK
2 hafte → 2 WEEKS
do hafte → 2 WEEKS

Only return a duration when an explicit quantity exists.

============================================================
UNIT PER TIME
============================================================

one tablet → 1
two tablets → 2
one capsule → 1
two capsules → 2

ek goli → 1
do goli → 2
ek tablet → 1
do tablet → 2
ek capsule → 1
do capsule → 2

If not specified:
1

============================================================
ROUTE
============================================================

Allowed:

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

Normal tablet/capsule:
oral

munh se
muh se
oral
by mouth
→ oral

skin par
skin pe
lagana hai
→ topical

Do not infer a special route unless supported.

============================================================
ASR SPELLING
============================================================

Minor speech-recognition spelling errors can be normalized
when the medicine identity is strongly supported.

Examples:

amoxiling → AMOXICILLIN
amoxicilin → AMOXICILLIN
paracetmol → PARACETAMOL
ibuprofren → IBUPROFEN

Do NOT invent a medicine solely from dosage.

============================================================
FINAL RULE
============================================================

Return ONLY valid JSON.

Do not explain your answer.
"""

    # ========================================================
    # TEXT NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: Optional[str]
    ) -> str:

        if not text:
            return ""

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9]+",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ========================================================
    # SIMILARITY
    # ========================================================

    @staticmethod
    def _similarity(
        left: str,
        right: str
    ) -> float:

        left = (
            OllamaPrescriptionExtractor
            ._normalize_text(left)
        )

        right = (
            OllamaPrescriptionExtractor
            ._normalize_text(right)
        )

        if not left or not right:
            return 0.0

        return SequenceMatcher(
            None,
            left,
            right
        ).ratio()

    # ========================================================
    # FIND LOCAL TRANSCRIPT CONTEXT
    # ========================================================

    def _get_context(
        self,
        medicine_name: str,
        transcript: str,
        radius: int = 60
    ) -> str:

        source = (
            self._normalize_text(
                transcript
            )
        )

        candidate = (
            self._normalize_text(
                medicine_name
            )
        )

        if not source:
            return ""

        words = source.split()

        if not candidate:
            return source[:500]

        candidate_words = candidate.split()

        best_index = None
        best_score = 0.0

        # Exact token / phrase search first.
        window_size = len(candidate_words)

        for i in range(
            max(1, len(words) - window_size + 1)
        ):

            window = " ".join(
                words[
                    i:i + window_size
                ]
            )

            score = self._similarity(
                candidate,
                window
            )

            if score > best_score:
                best_score = score
                best_index = i

        if best_index is None:
            return source[:500]

        start = max(
            0,
            best_index - radius
        )

        end = min(
            len(words),
            best_index + window_size + radius
        )

        return " ".join(
            words[start:end]
        )

    # ========================================================
    # PRESCRIPTION CONTEXT
    # ========================================================

    def _has_prescription_context(
        self,
        context: str
    ) -> bool:

        if not context:
            return False

        context = (
            self._normalize_text(
                context
            )
        )

        prescription_tokens = {
            "tablet",
            "tablets",
            "capsule",
            "capsules",
            "syrup",
            "injection",
            "cream",
            "ointment",
            "drops",
            "mg",
            "ml",
            "milligram",
            "milliliter",
            "morning",
            "afternoon",
            "evening",
            "night",
            "food",
            "days",
            "day",
            "week",
            "weeks",
        }

        words = set(
            context.split()
        )

        return len(
            words.intersection(
                prescription_tokens
            )
        ) >= 2

    # ========================================================
    # ASR MEDICINE CORRECTION
    # ========================================================

    def _correct_asr_medicine_name(
        self,
        name: str,
        transcript: str
    ) -> str:

        normalized_name = (
            self._normalize_text(name)
        )

        if not normalized_name:
            return ""
        
        # -----------------------------------------------------
        # Exact known medicine
        # -----------------------------------------------------

        if normalized_name in self.KNOWN_MEDICINES:

            return normalized_name

        # -----------------------------------------------------
        # Direct ASR alias
        # -----------------------------------------------------

        if normalized_name in self.ASR_MEDICINE_ALIASES:

            corrected = (
                self.ASR_MEDICINE_ALIASES[
                    normalized_name
                ]
            )

            context = (
                self._get_context(
                    name,
                    transcript
                )
            )

            # The alias must have prescription context.
            if (
                corrected in self.KNOWN_MEDICINES
                and self._has_prescription_context(
                    context
                )
            ):

                print(
                    "ASR medicine correction: "
                    f"{name} -> {corrected}"
                )

                return corrected

        # -----------------------------------------------------
        # Fuzzy comparison against known medicines
        # -----------------------------------------------------

        best_match = None
        best_score = 0.0

        for medicine in self.KNOWN_MEDICINES:

            score = self._similarity(
                normalized_name,
                medicine
            )

            if score > best_score:

                best_score = score
                best_match = medicine

        if (
            best_match
            and best_score >= 0.84
        ):

            context = (
                self._get_context(
                    name,
                    transcript
                )
            )

            if self._has_prescription_context(
                context
            ):

                print(
                    "Fuzzy ASR medicine correction: "
                    f"{name} -> {best_match} "
                    f"(score={best_score:.3f})"
                )

                return best_match

        return normalized_name

    # ========================================================
    # TRANSCRIPT EVIDENCE
    # ========================================================

    def _has_transcript_evidence(
        self,
        medicine_name: str,
        transcript: str
    ) -> bool:

        if not medicine_name or not transcript:
            return False

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
            return False

        # Exact phrase
        if medicine in source:
            return True

        source_words = source.split()

        # -----------------------------------------------------
        # Fuzzy word matching
        # -----------------------------------------------------

        best_score = 0.0

        if " " not in medicine:

            for word in source_words:

                score = self._similarity(
                    medicine,
                    word
                )

                if score > best_score:
                    best_score = score

        else:

            medicine_words = medicine.split()
            size = len(
                medicine_words
            )

            for index in range(
                len(source_words) - size + 1
            ):

                window = " ".join(
                    source_words[
                        index:index + size
                    ]
                )

                score = self._similarity(
                    medicine,
                    window
                )

                if score > best_score:
                    best_score = score

        if medicine in self.KNOWN_MEDICINES:

            return best_score >= 0.76

        return best_score >= 0.84

    # ========================================================
    # CLEAN MEDICINE NAME
    # ========================================================

    @classmethod
    def _clean_medicine_name(
        cls,
        name: Optional[str]
    ) -> str:

        if not name:
            return ""

        name = str(
            name
        ).strip()

        name = name.strip(
            "\"'`"
        )

        # -----------------------------------------------------
        # Remove forbidden commentary
        # -----------------------------------------------------

        forbidden = [
            "corrected from transcript",
            "corrected for clarity",
            "corrected from",
            "for clarity",
            "likely",
            "probably",
            "inferred",
            "assumed",
            "suggested",
        ]

        lower_name = name.lower()

        for phrase in forbidden:

            if phrase in lower_name:
                return ""

        # -----------------------------------------------------
        # Remove accidental punctuation
        # -----------------------------------------------------

        name = re.sub(
            r"[.,;:]+$",
            "",
            name
        ).strip()

        # -----------------------------------------------------
        # Remove dosage/form contamination
        # -----------------------------------------------------

        # Example:
        # "AMOXICILLIN 500MG CAPSULE"
        # becomes:
        # "AMOXICILLIN"
        #

        tokens = name.split()

        cleaned_tokens = []

        for token in tokens:

            token_clean = (
                token.lower()
                .strip(
                    ".,;:()[]{}"
                )
            )

            # Dosage token
            if re.fullmatch(
                r"\d+(?:\.\d+)?(?:mg|g|mcg|ml|l)",
                token_clean
            ):
                continue

            # Number alone
            if re.fullmatch(
                r"\d+(?:\.\d+)?",
                token_clean
            ):
                continue

            # Prescription form / junk
            if token_clean in cls.MEDICINE_NAME_JUNK:
                continue

            cleaned_tokens.append(
                token
            )

        name = " ".join(
            cleaned_tokens
        ).strip()

        return name

    # ========================================================
    # NORMALIZE FIELD
    # ========================================================

    @staticmethod
    def _normalize_string(
        value
    ) -> Optional[str]:

        if value is None:
            return None

        value = str(
            value
        ).strip()

        if not value:
            return None

        if value.lower() in {
            "null",
            "none",
            "unknown",
            "n/a",
            "na",
        }:

            return None

        return value

    # ========================================================
    # NORMALIZE DURATION
    # ========================================================

    @staticmethod
    def _normalize_duration(
        value
    ) -> Optional[str]:

        if value is None:
            return None

        text = str(
            value
        ).strip()

        if not text:
            return None

        normalized = re.sub(
            r"\s+",
            " ",
            text
        ).strip().lower()

        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*"
            r"(day|days|week|weeks|month|months)",
            normalized
        )

        if not match:
            return text.upper()

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

    # ========================================================
    # NORMALIZE FORM
    # ========================================================

    @staticmethod
    def _normalize_form(
        value
    ) -> str:

        if value is None:
            return "UNKNOWN"

        text = str(
            value
        ).strip().lower()

        form_map = {
            "tablet": "TABLET",
            "tablets": "TABLET",
            "pill": "TABLET",
            "pills": "TABLET",

            "capsule": "CAPSULE",
            "capsules": "CAPSULE",

            "syrup": "SYRUP",

            "injection": "INJECTION",

            "cream": "CREAM",

            "ointment": "OINTMENT",

            "drop": "DROPS",
            "drops": "DROPS",
        }

        return form_map.get(
            text,
            "UNKNOWN"
        )

    # ========================================================
    # NORMALIZE FREQUENCY
    # ========================================================

    @staticmethod
    def _normalize_frequency(
        value
    ) -> Optional[str]:

        if value is None:
            return None

        text = str(
            value
        ).strip().lower()

        if not text:
            return None

        mapping = {

            "od": "1-0-0",
            "once daily": "1-0-0",
            "once a day": "1-0-0",

            "bd": "1-0-1",
            "twice daily": "1-0-1",
            "twice a day": "1-0-1",

            "tid": "1-1-1",
            "three times": "1-1-1",
            "three times daily": "1-1-1",
            "three times a day": "1-1-1",

            "qid": "1-1-1-1",
            "four times": "1-1-1-1",
            "four times daily": "1-1-1-1",
            "four times a day": "1-1-1-1",

            "sos": "SOS",
            "prn": "SOS",
            "as needed": "SOS",
            "when required": "SOS",
        }

        return mapping.get(
            text,
            str(value).strip()
        )

    # ========================================================
    # NORMALIZE UNIT PER TIME
    # ========================================================

    @staticmethod
    def _normalize_unit_per_time(
        value
    ) -> str:

        if value is None:
            return "1"

        text = str(
            value
        ).strip().lower()

        if not text:
            return "1"

        number_match = re.search(
            r"\d+(?:\.\d+)?",
            text
        )

        if number_match:
            return number_match.group(
                0
            )

        words = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
        }

        for word, number in words.items():

            if word in text:
                return number

        return "1"

    # ========================================================
    # EXTRACT ASYNC
    # ========================================================

    async def extract_async(
        self,
        text: str
    ) -> List[ExtractedMedicine]:

        if not text or not text.strip():
            return []

        last_error = None

        for attempt in range(
            self.config.max_retries + 1
        ):

            try:

                async with httpx.AsyncClient(
                    timeout=self.config.timeout
                ) as client:

                    response = await client.post(

                        f"{self.config.base_url}/api/chat",

                        json={
                            "model": self.config.model,

                            "messages": [

                                {
                                    "role": "system",
                                    "content": (
                                        self.system_prompt
                                    )
                                },

                                {
                                    "role": "user",
                                    "content": (
                                        "Extract ALL prescription "
                                        "medicines explicitly supported "
                                        "by this transcript.\n\n"

                                        "IMPORTANT:\n"
                                        "- medicineName = ONLY medicine name\n"
                                        "- dosage = ONLY dosage\n"
                                        "- form = ONLY TABLET/CAPSULE/etc.\n"
                                        "- duration = ONLY duration\n"
                                        "- do NOT create medicines\n"
                                        "- preserve CAPSULE vs TABLET\n\n"

                                        "TRANSCRIPT:\n"
                                        f"{text}"
                                    )
                                }
                            ],

                            "temperature":
                                self.config.temperature,

                            "format":
                                "json",

                            "stream":
                                False
                        }
                    )

                    response.raise_for_status()

                    result = response.json()

                    content = (
                        result
                        .get(
                            "message",
                            {}
                        )
                        .get(
                            "content",
                            "{}"
                        )
                    )

                    # ------------------------------------------------
                    # JSON PARSE
                    # ------------------------------------------------

                    try:

                        parsed = json.loads(
                            content
                        )

                    except json.JSONDecodeError:

                        print(
                            "Ollama returned invalid JSON:"
                        )

                        print(content)

                        return []

                    if not isinstance(
                        parsed,
                        dict
                    ):

                        return []

                    raw_medicines = parsed.get(
                        "medicines",
                        []
                    )

                    if not isinstance(
                        raw_medicines,
                        list
                    ):

                        return []

                    medicines = []

                    # Used to prevent accidental identical
                    # duplicate objects from LLM output.
                    seen_prescriptions = set()

                    # ------------------------------------------------
                    # PROCESS EACH MEDICINE
                    # ------------------------------------------------

                    for med_data in raw_medicines:

                        if not isinstance(
                            med_data,
                            dict
                        ):
                            continue

                        raw_name = med_data.get(
                            "medicineName"
                        )

                        name = (
                            self._clean_medicine_name(
                                raw_name
                            )
                        )

                        if not name:

                            print(
                                "Rejected medicine with empty/"
                                "invalid medicineName."
                            )

                            continue

                        # ------------------------------------------------
                        # Correct ASR medicine spelling
                        # ------------------------------------------------

                        corrected_name = (
                            self._correct_asr_medicine_name(
                                name,
                                text
                            )
                        )

                        if corrected_name:
                            name = corrected_name

                        # ------------------------------------------------
                        # Reject known junk
                        # ------------------------------------------------

                        normalized_name = (
                            self._normalize_text(
                                name
                            )
                        )

                        if (
                            normalized_name in
                            self.MEDICINE_NAME_JUNK
                        ):

                            print(
                                "Rejected invalid medicineName: "
                                f"{name}"
                            )

                            continue

                        # ------------------------------------------------
                        # Transcript evidence
                        # ------------------------------------------------

                        evidence_ok = (
                            self._has_transcript_evidence(
                                name,
                                text
                            )
                        )

                        # ------------------------------------------------
                        # Special ASR alias evidence
                        # ------------------------------------------------
                        #
                        # Example:
                        # Whisper:
                        # "Scribe as 300 mg tablet..."
                        #
                        # Corrected:
                        # ASPIRIN
                        #
                        # The transcript itself won't contain the word
                        # "aspirin", so allow the correction only if the
                        # original LLM name was a configured ASR alias.
                        #

                        original_normalized_name = (
                            self._normalize_text(
                                self._clean_medicine_name(
                                    raw_name
                                )
                            )
                        )

                        is_asr_alias = (
                            original_normalized_name
                            in self.ASR_MEDICINE_ALIASES
                        )

                        if (
                            not evidence_ok
                            and not is_asr_alias
                        ):

                            print(
                                "Hallucination guard rejected "
                                f"medicine: {name}"
                            )

                            continue

                        # ------------------------------------------------
                        # Route
                        # ------------------------------------------------

                        route_value = (
                            med_data.get(
                                "route"
                            )
                        )

                        route_value = (
                            self._normalize_string(
                                route_value
                            )
                            or "unknown"
                        )

                        route_value = (
                            route_value.lower()
                        )

                        try:

                            route = Route(
                                route_value
                            )

                        except ValueError:

                            route = Route.UNKNOWN

                        # ------------------------------------------------
                        # Confidence
                        # ------------------------------------------------

                        confidence = (
                            med_data.get(
                                "confidence",
                                0.8
                            )
                        )

                        try:

                            confidence = float(
                                confidence
                            )

                        except (
                            TypeError,
                            ValueError
                        ):

                            confidence = 0.8

                        confidence = max(
                            0.0,
                            min(
                                1.0,
                                confidence
                            )
                        )

                        # ------------------------------------------------
                        # Raw fields
                        # ------------------------------------------------

                        dosage_value = (
                            med_data.get(
                                "dosage"
                            )
                        )

                        duration_value = (
                            med_data.get(
                                "duration"
                            )
                        )

                        instructions_value = (
                            med_data.get(
                                "instructions"
                            )
                        )

                        frequency_value = (
                            med_data.get(
                                "frequency"
                            )
                        )

                        form_value = (
                            med_data.get(
                                "form"
                            )
                        )

                        unit_value = (
                            med_data.get(
                                "unitPerTime"
                            )
                        )

                        # ------------------------------------------------
                        # Normalized fields
                        # ------------------------------------------------

                        dosage = (
                            self._normalize_string(
                                dosage_value
                            )
                        )

                        if dosage:
                            dosage = dosage.upper()

                        duration = (
                            self._normalize_duration(
                                duration_value
                            )
                        )

                        instructions = (
                            self._normalize_string(
                                instructions_value
                            )
                        )

                        if instructions:
                            instructions = (
                                instructions.upper()
                            )

                        frequency = (
                            self._normalize_frequency(
                                frequency_value
                            )
                        )

                        form = (
                            self._normalize_form(
                                form_value
                            )
                        )

                        unit_per_time = (
                            self._normalize_unit_per_time(
                                unit_value
                            )
                        )

                        # ------------------------------------------------
                        # SAFETY: do not allow dosage to leak into name
                        # ------------------------------------------------

                        if re.search(
                            r"\d+\s*(mg|g|mcg|ml|l)\b",
                            name.lower()
                        ):

                            name = re.sub(
                                r"\d+\s*(mg|g|mcg|ml|l)\b",
                                "",
                                name,
                                flags=re.IGNORECASE
                            ).strip()

                        # ------------------------------------------------
                        # SAFETY: normalize final name
                        # ------------------------------------------------

                        name = (
                            str(name)
                            .upper()
                            .strip()
                        )

                        if not name:
                            continue

                        # ------------------------------------------------
                        # Duplicate protection
                        # ------------------------------------------------
                        #
                        # Same medicine + same dosage + same frequency +
                        # same duration is treated as an accidental
                        # duplicate.
                        #
                        # Same medicine with different prescription
                        # details is preserved.
                        #

                        duplicate_key = (
                            name,
                            dosage,
                            frequency,
                            duration,
                            form,
                            instructions,
                            unit_per_time,
                        )

                        if duplicate_key in (
                            seen_prescriptions
                        ):

                            print(
                                "Duplicate prescription ignored: "
                                f"{name}"
                            )

                            continue

                        seen_prescriptions.add(
                            duplicate_key
                        )

                        # ------------------------------------------------
                        # Create ExtractedMedicine
                        # ------------------------------------------------

                        med = ExtractedMedicine(

                            name=name,

                            dosage=dosage,

                            frequency=frequency,

                            duration=duration,

                            route=route,

                            instructions=instructions,

                            confidence=confidence
                        )

                        # ------------------------------------------------
                        # Transformer compatibility
                        # ------------------------------------------------

                        try:
                            med.form = form
                        except Exception:
                            pass

                        try:
                            med.unitPerTime = (
                                unit_per_time
                            )
                        except Exception:
                            pass

                        medicines.append(
                            med
                        )

                    # ------------------------------------------------
                    # Return
                    # ------------------------------------------------

                    return medicines

            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.ConnectError
            ) as e:

                last_error = e

                print(
                    f"Ollama attempt "
                    f"{attempt + 1}/"
                    f"{self.config.max_retries + 1} "
                    f"failed: {type(e).__name__}"
                )

                if (
                    attempt
                    < self.config.max_retries
                ):

                    import asyncio

                    await asyncio.sleep(
                        2
                    )

            except Exception as e:

                print(
                    "Ollama extraction error:"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                return []

        print(
            "Ollama extraction failed after retries:"
        )

        if last_error:

            print(
                f"{type(last_error).__name__}: "
                f"{last_error}"
            )

        return []

    # ========================================================
    # SYNCHRONOUS EXTRACTION
    # ========================================================

    def extract(
        self,
        text: str
    ) -> List[ExtractedMedicine]:

        import asyncio

        try:

            loop = asyncio.get_event_loop()

        except RuntimeError:

            loop = asyncio.new_event_loop()

            asyncio.set_event_loop(
                loop
            )

        return loop.run_until_complete(
            self.extract_async(
                text
            )
        )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    async def check_model_available(
        self
    ) -> bool:

        try:

            response = await self._async_get(
                f"{self.config.base_url}/api/tags"
            )

            response.raise_for_status()

            models = (
                response
                .json()
                .get(
                    "models",
                    []
                )
            )

            return any(
                self.config.model
                in model.get(
                    "name",
                    ""
                )
                for model in models
            )

        except Exception as e:

            print(
                f"Ollama model check failed: {e}"
            )

            return False

    # ========================================================
    # ASYNC GET HELPER
    # ========================================================

    async def _async_get(
        self,
        url: str
    ):

        async with httpx.AsyncClient(
            timeout=self.config.timeout
        ) as client:

            return await client.get(
                url
            )

    # ========================================================
    # PULL MODEL
    # ========================================================

    async def pull_model(
        self
    ) -> bool:

        try:

            async with httpx.AsyncClient(
                timeout=300.0
            ) as client:

                response = await client.post(

                    f"{self.config.base_url}/api/pull",

                    json={
                        "name": self.config.model
                    }
                )

                response.raise_for_status()

                return True

        except Exception as e:

            print(
                f"Failed to pull model: {e}"
            )

            return False

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:

            self.client.close()

        except Exception:
            pass


# ============================================================
# SINGLETON
# ============================================================

_ollama_extractor_instance = None


def get_ollama_extractor(
    config: OllamaConfig = None
) -> OllamaPrescriptionExtractor:

    global _ollama_extractor_instance

    if _ollama_extractor_instance is None:

        if config is None:

            config = OllamaConfig(
                base_url="http://localhost:11434",
                model="ministral-3:3b",
                temperature=0.1,
                timeout=300.0,
                max_retries=2
            )

        _ollama_extractor_instance = (
            OllamaPrescriptionExtractor(
                config
            )
        )

    return _ollama_extractor_instance