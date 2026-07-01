"""
Schema Definition for Patient Medication Profile Extraction.

This module defines the strict Pydantic schemas used to extract and structure
clinical medication profiles from unstructured medical notes. The design incorporates:
1. Strict type checking and literal enums to prevent hallucinated parameters.
2. Semantic Anchoring: Forcing the model to extract text clues before categorizing 
   parameters. This keeps the model locked to text evidence.
3. Post-validation checks to maintain structural alignment.
"""
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, model_validator
from typing import Literal

class Dosage(BaseModel):
    """
    Structured representation of a drug dosage.
    
    Splits quantity from the measurement unit to facilitate downstream logic and audits.
    """
    amount: str = Field(
        ..., 
        description="Numeric value representing the quantity as a string (e.g. '50', '100', '0.05', '12.5', '7.5', '2')."
                    "If no numerical value is specified in the text, use 'not specified'."
    )
    unit: Literal[
        'mg', 'ug', 'mcg', 'mL', 'g', 'mEq', 'tablet', 'tablets', 
        'capsule', 'capsules', 'drops', 'puffs', 'sprays', 'units', 
        'percent', 'not specified'
    ] = Field(
        ..., 
        description="Standard clinical unit of measurement."
    )

    @model_validator(mode='after')
    def enforce_not_specified_alignment(self) -> 'Dosage':
        """
        Pydantic validator to enforce alignment.
        If the amount is 'not specified', the unit must also be forced to 'not specified'
        to prevent cases where the model extracts an arbitrary unit without a quantity.
        """
        if self.amount == 'not specified' and self.unit != 'not specified':
            self.unit = 'not specified'
        return self

class MedicationEntry(BaseModel):
    """
    A unified entry representing a single medication mention or allergen.
    
    Uses Semantic Anchoring by requesting text evidence (`administration_text_clue` and
    `classification_text_clue`) before demanding standard classification or routes.
    """
    drug_name: str = Field(
        ..., 
        description="Generic or Brand name of the medication as written in the note."
    )
    dosage: Dosage = Field(
        ..., 
        description="Strictly structured dosage information containing quantity and unit."
    )
    frequency: Literal[
        'QD', 'BID', 'TID', 'QID', 'PRN', 'QHS', 'QOD', 'QAM', 'QPM', 
        'q4h', 'q6h', 'q8h', 'q12h', 
        'Q4H', 'Q6H', 'Q8H', 'Q12H', 
        'monthly', 'other', 'not specified'
    ] = Field(
        ..., 
        description="Standard clinical frequency abbreviation."
    )
    
    # SEMANTIC ANCHORING FOR ROUTES
    administration_text_clue: str = Field(
        ..., 
        description="The exact words in the text indicating how this drug is taken or packaged, "
                    "e.g., 'instilled in eyes', 'subcutaneous injection', 'nasal spray', 'p.o.', 'swallowed'."
    )
    route: Literal[
        'Oral', 'Sublingual', 'Rectal', 'Intravenous', 'Subcutaneous', 'Nasal', 
        'Inhalation', 'Topical', 'Intramuscular', 'Transdermal', 'Ophthalmic', 
        'Otic', 'Vaginal', 'Enteral', 'Intradermal', 'other', 'not specified'
    ] = Field(
        ..., 
        description="Standard anatomical route of administration."
    )
    
    # SEMANTIC ANCHORING FOR STATUS
    classification_text_clue: str = Field(
        ...,
        description="The exact words in the text indicating the status or context of this drug mention, "
                    "e.g., 'presently on', 'history of allergy to', 'administered in the ER', 'stopped taking'."
    )
    clinical_classification: Literal[
        'active', 
        'recently discontinued', 
        'excluded - allergy', 
        'excluded - acute procedural', 
        'excluded - historical past', 
        'excluded - vaccine', 
        'excluded - pending/conditional',
        'excluded - other'
    ] = Field(
        ..., 
        description="The definitive clinical status or category assigned to this specific drug mention."
    )


class PatientProfileStrict(BaseModel):
    """
    The root patient profile returned by the parser.
    """
    reasoning: str = Field(
        ..., 
        description="Detailed step-by-step clinical analysis of the transcription, explaining the categorization of each mentioned drug."
    )
    all_mentioned_medications: list[MedicationEntry] = Field(
        default_factory=list, 
        description="A single comprehensive list of every medication entity and allergen found in the note."
    )
