import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any

import httpx


# ============================================================
# CONFIG
# ============================================================

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

DEFAULT_DB_PATH = "medical_terminology.db"


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class MedicineMatch:
    input_text: str
    normalized_name: Optional[str]
    rxcui: Optional[str]
    confidence: float
    source: str
    matched: bool


# ============================================================
# MEDICAL TERMINOLOGY SERVICE
# ============================================================

class MedicalTerminologyService:

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        timeout: float = 10.0
    ):

        self.db_path = db_path
        self.timeout = timeout

        self._create_database()

    # ========================================================
    # DATABASE
    # ========================================================

    def _create_database(self):

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS medicine_terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT UNIQUE NOT NULL,
                normalized_name TEXT NOT NULL,
                rxcui TEXT,
                source TEXT NOT NULL
            )
            """
        )

        connection.commit()
        connection.close()

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def normalize_text(
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
    def similarity(
        left: str,
        right: str
    ) -> float:

        left = (
            MedicalTerminologyService
            .normalize_text(left)
        )

        right = (
            MedicalTerminologyService
            .normalize_text(right)
        )

        if not left or not right:
            return 0.0

        return SequenceMatcher(
            None,
            left,
            right
        ).ratio()

    # ========================================================
    # SAVE LOCAL TERM
    # ========================================================

    def save_term(
        self,
        term: str,
        normalized_name: str,
        rxcui: Optional[str],
        source: str
    ):

        normalized_term = (
            self.normalize_text(term)
        )

        normalized_name = (
            self.normalize_text(normalized_name)
        )

        if not normalized_term:
            return

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO medicine_terms
            (
                term,
                normalized_name,
                rxcui,
                source
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                normalized_term,
                normalized_name,
                rxcui,
                source
            )
        )

        connection.commit()
        connection.close()

    # ========================================================
    # LOCAL EXACT LOOKUP
    # ========================================================

    def local_exact_lookup(
        self,
        term: str
    ) -> Optional[MedicineMatch]:

        normalized_term = (
            self.normalize_text(term)
        )

        if not normalized_term:
            return None

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                term,
                normalized_name,
                rxcui,
                source
            FROM medicine_terms
            WHERE term = ?
            LIMIT 1
            """,
            (
                normalized_term,
            )
        )

        row = cursor.fetchone()

        connection.close()

        if not row:
            return None

        return MedicineMatch(
            input_text=term,
            normalized_name=row[1].upper(),
            rxcui=row[2],
            confidence=1.0,
            source=row[3],
            matched=True
        )

    # ========================================================
    # LOCAL FUZZY LOOKUP
    # ========================================================

    def local_fuzzy_lookup(
        self,
        term: str,
        threshold: float = 0.88
    ) -> Optional[MedicineMatch]:

        normalized_term = (
            self.normalize_text(term)
        )

        if not normalized_term:
            return None

        connection = sqlite3.connect(
            self.db_path
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                term,
                normalized_name,
                rxcui,
                source
            FROM medicine_terms
            """
        )

        rows = cursor.fetchall()

        connection.close()

        best_row = None
        best_score = 0.0

        for row in rows:

            candidate = row[0]

            score = self.similarity(
                normalized_term,
                candidate
            )

            if score > best_score:

                best_score = score
                best_row = row

        if (
            best_row is None
            or best_score < threshold
        ):
            return None

        return MedicineMatch(
            input_text=term,
            normalized_name=(
                best_row[1].upper()
            ),
            rxcui=best_row[2],
            confidence=best_score,
            source=(
                f"{best_row[3]}_fuzzy"
            ),
            matched=True
        )

    # ========================================================
    # RXNORM APPROXIMATE LOOKUP
    # ========================================================

    async def rxnorm_lookup(
        self,
        term: str
    ) -> Optional[MedicineMatch]:

        normalized_term = (
            self.normalize_text(term)
        )

        if not normalized_term:
            return None

        try:

            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.get(
                    f"{RXNORM_BASE_URL}/"
                    f"approximateTerm.json",
                    params={
                        "term": normalized_term,
                        "maxEntries": 5,
                        "option": 1
                    }
                )

                response.raise_for_status()

                data = response.json()

        except Exception as e:

            print(
                f"RxNorm lookup failed: {e}"
            )

            return None

        approximate_group = (
            data.get(
                "approximateGroup",
                {}
            )
        )

        candidates = (
            approximate_group.get(
                "candidate",
                []
            )
        )

        if not candidates:
            return None

        best_candidate = None

        for candidate in candidates:

            name = candidate.get(
                "name"
            )

            rxcui = candidate.get(
                "rxcui"
            )

            if not name:
                continue

            if not rxcui:
                continue

            if best_candidate is None:

                best_candidate = candidate

        if best_candidate is None:
            return None

        candidate_name = (
            best_candidate.get(
                "name"
            )
        )

        rxcui = (
            best_candidate.get(
                "rxcui"
            )
        )

        api_score = float(
            best_candidate.get(
                "score",
                0.0
            )
        )

        # RxNorm's score is lexical-ranking information,
        # not a probability from 0 to 1.
        #
        # We therefore use our own secondary lexical
        # similarity before accepting a correction.

        lexical_similarity = self.similarity(
            normalized_term,
            candidate_name
        )

        confidence = lexical_similarity

        # Conservative threshold for automatic normalization.
        if lexical_similarity < 0.88:

            return MedicineMatch(
                input_text=term,
                normalized_name=None,
                rxcui=None,
                confidence=confidence,
                source="rxnorm_low_confidence",
                matched=False
            )

        return MedicineMatch(
            input_text=term,
            normalized_name=candidate_name.upper(),
            rxcui=str(rxcui),
            confidence=confidence,
            source="rxnorm",
            matched=True
        )

    # ========================================================
    # UNIFIED LOOKUP
    # ========================================================

    async def resolve_medicine(
        self,
        term: str
    ) -> MedicineMatch:

        # ----------------------------------------------------
        # 1. Exact local lookup
        # ----------------------------------------------------

        exact = self.local_exact_lookup(
            term
        )

        if exact:

            return exact

        # ----------------------------------------------------
        # 2. Local fuzzy lookup
        # ----------------------------------------------------

        fuzzy = self.local_fuzzy_lookup(
            term,
            threshold=0.88
        )

        if fuzzy:

            return fuzzy

        # ----------------------------------------------------
        # 3. RxNorm approximate lookup
        # ----------------------------------------------------

        rxnorm_match = await self.rxnorm_lookup(
            term
        )

        if rxnorm_match and rxnorm_match.matched:

            # Cache successful lookup locally.

            self.save_term(
                term=term,
                normalized_name=(
                    rxnorm_match.normalized_name
                ),
                rxcui=rxnorm_match.rxcui,
                source="rxnorm"
            )

            return rxnorm_match

        # ----------------------------------------------------
        # 4. No reliable match
        # ----------------------------------------------------

        return MedicineMatch(
            input_text=term,
            normalized_name=None,
            rxcui=None,
            confidence=0.0,
            source="unresolved",
            matched=False
        )

    # ========================================================
    # EXTRACT POSSIBLE MEDICINE TOKENS
    # ========================================================

    @staticmethod
    def find_candidate_words(
        transcript: str
    ) -> List[str]:

        if not transcript:
            return []

        words = re.findall(
            r"\b[A-Za-z][A-Za-z-]{3,}\b",
            transcript
        )

        return words

    # ========================================================
    # RESOLVE CANDIDATES FROM TRANSCRIPT
    # ========================================================

    async def resolve_transcript(
        self,
        transcript: str
    ) -> List[MedicineMatch]:

        candidates = (
            self.find_candidate_words(
                transcript
            )
        )

        results = []

        seen = set()

        for candidate in candidates:

            normalized = (
                self.normalize_text(
                    candidate
                )
            )

            if (
                not normalized
                or normalized in seen
            ):
                continue

            seen.add(normalized)

            match = await self.resolve_medicine(
                candidate
            )

            if match.matched:

                results.append(
                    match
                )

        return results


# ============================================================
# SINGLETON
# ============================================================

_terminology_service: Optional[
    MedicalTerminologyService
] = None


def get_medical_terminology_service():

    global _terminology_service

    if _terminology_service is None:

        _terminology_service = (
            MedicalTerminologyService()
        )

    return _terminology_service