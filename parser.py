import requests
import time
from medication_strict import PatientProfileStrict

# Pass 1: Extract Checklist
PASS1_PROMPT = """
You are an expert clinical data extraction helper. Read the clinical transcription and extract a simple comma-separated list of all pharmacological substances.

CRITICAL NEGATIVE CONSTRAINTS:
- Do NOT extract medical diagnoses or symptoms (e.g., 'bipolar', 'diabetes', 'sprain', 'pain').
- Do NOT extract surgical procedures (e.g., 'lithotripsy', 'thrombectomy').
- Do NOT extract medical devices or labs (e.g., 'catheter', 'stent', 'guaiac', 'pacemaker').
- Do NOT extract lifestyle habits (e.g., 'smoking', 'alcohol').
Extract ONLY chemical substances, medications, supplements, vaccines, and allergens.

RULES:
1. Output ONLY a comma-separated list of substance names.
2. Include active medications, recently discontinued medications, vitamins, over-the-counter supplements, vaccines, and any allergens (medications, foods, or environmental).
3. Be strictly accurate: If a term is a condition or device, omit it entirely.
4. Do not include duplicates.
5. If there are no medications or allergies extracted in the text, just output "none".

EXAMPLE OUTPUT:
aspirin, lisinopril, atorvastatin, peanut, pneumovax
"""

# Pass 2: Checklist-Guided Structured Parsing
PASS2_SYSTEM_PROMPT = """
You are an expert clinical data extraction engine. Extract medication details and allergy/allergen mentions from the clinical text for the specific list of target substances provided and classify them using the Pydantic schema.

### TARGET CHECKLIST RULE:
You will be provided with a target checklist of substances extracted from the text.
For each substance in the checklist, you MUST generate exactly one entry in the 'all_mentioned_medications' array. 

### CLASSIFICATION RULES:
1. **active**:
   - Currently active chronic home/maintenance medications, active ongoing prescriptions, and regular daily supplements/vitamins.
   - Definitive Final Plan: If a medication is explicitly listed in the final "PLAN" or "DISPOSITION" to be continued, restarted, or definitively newly prescribed, it is "active".
2. **recently discontinued**:
   - Medications explicitly stopped, finished, or discontinued during this current episode of care, AND not restarted in the final plan.
3. **excluded - allergy**:
   - Substances listed under allergies, intolerances, or causing adverse reactions (drugs, foods, environmental triggers).
4. **excluded - acute procedural**:
   - One-off doses administered in the ER, hospital, or during a procedure (e.g., procedural sedatives, IV fluids, anesthetics).
5. **excluded - historical past**:
   - Medications the patient took in the past but is no longer taking, or historical substance abuse (e.g., past cocaine use).
6. **excluded - vaccine**:
   - Preventive immunizations/vaccines given during a visit.
7. **excluded - pending/conditional**:
   - Medications that *might* be started in the future (e.g., "may start if labs are okay", "will consider starting").

### STANDARDIZATION & FORMATTING RULES:
1. **Dosage**: Split dosage into:
   - 'amount': A numeric string (e.g., '50', '12.5'). If no numerical value is specified, use 'not specified'.
   - 'unit': Choose the matching standard literal from the schema (e.g., 'mg', 'mcg', 'tablet'). If unknown, use 'not specified'.
2. **Frequency & Route**: Must match the schema enums. Do not guess or default to 'Oral'. If the route or frequency is not explicitly mentioned or clearly indicated by a text clue (e.g., 'tablets' or 'p.o.' indicating 'Oral', 'instilled in eyes' or 'drops' indicating 'Ophthalmic', 'inhaler' indicating 'Inhalation', 'nasal spray' or 'spray' or 'nasal' indicating 'Nasal', 'injection' indicating 'Subcutaneous' or 'Intravenous', 'b.i.d.' or 'BID' indicating 'BID'), set it strictly to 'not specified'.
3. **Handling Missing Fields**:
   - For allergies, vaccines, historical drugs, or errors, set 'dosage.amount', 'dosage.unit', 'frequency', and 'route' to 'not specified'.
   - Set 'administration_text_clue' to 'not specified'.
   - Set 'classification_text_clue' to the exact words anchoring the classification.
4. **Recency and Conflict Resolution Policy**:
   - The final treatment/discharge plan ALWAYS overrides prior admission or home parameters. If mentioned multiple times, extract parameters (dosage, frequency, route) solely from the most recent clinical decision. Do not mix parameters from different sections.

### SEMANTIC ANCHORING (Chain of Thought):
For each entry, extract:
- 'administration_text_clue': Exact words indicating how this drug is taken (e.g., 'take two tablets'). Use 'not specified' if none.
- 'classification_text_clue': Exact words indicating the status (e.g., 'presently on', 'allergic to', 'will stop taking').
This anchors your decision directly to raw text evidence before categorization.

### CLINICAL REASONING:
Use the 'reasoning' field to perform a step-by-step clinical analysis, explaining your findings based strictly on the text.
"""

def get_checklist(transcription: str) -> list[str]:
    """
    Executes Pass 1.
    Queries the LLM to extract a simple comma-separated checklist of all substances.
    """
    messages = [
        {"role": "system", "content": PASS1_PROMPT},
        {"role": "user", "content": transcription}
    ]
    payload = {
        "model": "qwen2.5:14b",
        "messages": messages,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192
        },
        "think": False,
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/chat", json=payload)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "").strip()
    if not content:
        return []
    
    content = content.replace("`", "").replace("[", "").replace("]", "").replace("\n", ",")
    items = [x.strip() for x in content.split(",") if x.strip()]
    
    # Filter out empty entries and common garbage phrases
    garbage = {"none", "None", "no medications", "no allergies", "n/a", "no known allergies", "none known"}
    items = [x for x in items if x.lower() not in garbage]
    
    return list(dict.fromkeys(items))

def parse_single_transcription(text_to_parse: str) -> dict:
    """
    Parses a single clinical transcription using the two-pass checklist-guided pipeline.
    Returns a dictionary of parsed active/discontinued/excluded medications and reasoning.
    """
    checklist = get_checklist(text_to_parse)
    if not checklist:
        return {
            "status": "Success",
            "reasoning": "No medications or allergies extracted in Pass 1 checklist.",
            "medications": [],
            "excluded_medications": []
        }
        
    user_content = (
        f"Clinical Transcription:\n\"\"\"\n{text_to_parse}\n\"\"\"\n\n"
        f"Checklist of target substances to extract:\n{checklist}\n\n"
        f"Please generate a MedicationEntry in 'all_mentioned_medications' for every substance on the checklist."
    )
    
    messages = [
        {"role": "system", "content": PASS2_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    retries = 3
    for attempt in range(retries):
        content = ""
        try:
            payload = {
                "model": "qwen2.5:14b",
                "messages": messages,
                "format": PatientProfileStrict.model_json_schema(),
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 8192,
                    "num_predict": 4096
                },
                "think": False,
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/chat", json=payload)
            response.raise_for_status()
            
            res_json = response.json()
            message = res_json.get("message", {})
            content = str(message.get("content", ""))
            
            patient_profile = PatientProfileStrict.model_validate_json(content)
            reasoning = patient_profile.reasoning
            
            active_meds = []
            disc_meds = []
            excl_meds = []
            
            for entry in patient_profile.all_mentioned_medications:
                med_dict = {
                    "drug_name": entry.drug_name,
                    "dosage": {
                        "amount": entry.dosage.amount,
                        "unit": entry.dosage.unit
                    },
                    "frequency": entry.frequency,
                    "route": entry.route,
                    "administration_text_clue": entry.administration_text_clue,
                    "classification_text_clue": entry.classification_text_clue,
                    "clinical_classification": entry.clinical_classification
                }
                classification = entry.clinical_classification
                if classification == 'active':
                    med_dict['status'] = 'active'
                    active_meds.append(med_dict)
                elif classification == 'recently discontinued':
                    med_dict['status'] = 'discontinued'
                    disc_meds.append(med_dict)
                elif classification.startswith('excluded'):
                    reason = classification.replace('excluded - ', '')
                    excl_meds.append({
                        "drug_name": entry.drug_name,
                        "exclusion_reason": reason,
                        "classification_text_clue": entry.classification_text_clue
                    })
            
            return {
                "status": "Success",
                "reasoning": reasoning,
                "medications": active_meds + disc_meds,
                "excluded_medications": excl_meds
            }
        except Exception as e:
            if attempt == retries - 1:
                return {
                    "status": "Failed",
                    "error": str(e),
                    "reasoning": "Schema validation failed after all retries.",
                    "medications": [],
                    "excluded_medications": []
                }
            else:
                if content:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user", 
                        "content": f"Validation failed with the following error:\n{e}\n\nPlease output the corrected JSON matching the schema."
                    })
                time.sleep(1)
                
    return {
        "status": "Failed",
        "error": "Failed after maximum retries.",
        "reasoning": "",
        "medications": [],
        "excluded_medications": []
    }
