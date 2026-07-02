# Clinical Medication Parser

A local-LLM pipeline for extracting structured medication profiles from unstructured clinical transcription notes.

This project uses a two-pass extraction design with Ollama, Qwen 2.5 14B, and Pydantic validation to identify active medications, recently discontinued medications, allergies, vaccines, acute/procedural drugs, historical mentions, dosage, frequency, route, and source-text evidence.

Built from scratch as a personal project without relying on high-level LLM orchestration frameworks.

> **Status:** Portfolio / research project. This is not a medical device, not clinical decision support, and should not be used for patient care without review and formal validation.

---

## Why This Project Exists

Clinical notes often mention medications in messy, overlapping contexts:

- home medication lists
- allergies and intolerances
- drugs given once in the ER or during a procedure
- medications stopped during the encounter
- vaccines
- historical medications
- conditional future plans
- incomplete dosage, route, or frequency details

A simple single-pass LLM extraction prompt often finds the obvious active drugs but misses non-active mentions or guesses missing details. This project was built to test whether a smaller local model can perform safer, evidence-anchored clinical medication extraction when combined with strict schemas and a two-pass workflow.

---

## Demo

```bash
python medication_parser.py --ids 4313
```

Example console output:

```text
==================================================
Parsing Patient ID 4313 (Specialty:  Consult - History and Phy.)
==================================================

🧠 Clinical Reasoning:
The clinical transcription provides detailed information about the patient's medical history, current medications, and treatment plan. The medication list includes several drugs that are part of the patient's ongoing management for epilepsy and other neurological conditions. Additionally, there is a mention of past medications in the history section.

📋 Active & Discontinued Medications:
   • Zonegran | Dose: 50 mg | Frequency: BID | Route: Oral | Status: active
   • Lamictal | Dose: 200 mg | Frequency: BID | Route: Oral | Status: active
   • baclofen | Dose: 10 mg | Frequency: TID | Route: Oral | Status: active
   • Xopenex | Dose: not specified | Frequency: not specified | Route: not specified | Status: active
   • Atrovent | Dose: not specified | Frequency: not specified | Route: not specified | Status: active

🚫 Excluded Mentions & Allergies:
   • Vigabatrin | Reason: historical past (Clue: 'history of use')
   • phenobarbital | Reason: historical past (Clue: 'history of use')
   • valproic acid | Reason: historical past (Clue: 'history of use')
   • Topamax | Reason: historical past (Clue: 'history of use')
```
---

## How It Works

```text
Raw clinical transcription
        |
        v
Pass 1: Substance checklist
- Extracts a comma-separated list of medications, supplements, vaccines, and allergens.
- Applies negative constraints to avoid diagnoses, devices, procedures, labs, and symptoms.
        |
        v
Pass 2: Checklist-guided structured parsing
- Re-reads the original note plus the target checklist.
- Attempts to create one structured MedicationEntry per checklist item.
- Classifies each mention as active, discontinued, allergy, historical, vaccine, procedural, conditional, or other.
        |
        v
Pydantic validation
- Enforces allowed dosage units, route enums, frequency enums, and classification enums.
- Aligns missing dosage values so incomplete dose fields remain internally consistent.
        |
        v
Normalized output
- Returns active/discontinued medications separately from excluded mentions and allergies.
```

The key idea is to separate **recall** from **structured extraction**. Pass 1 focuses only on finding possible substances. Pass 2 then performs the harder clinical classification and field extraction against a strict schema.

---

## Key Design Decisions

### Checklist-Guided Extraction

Rather than extracting names, dosages, routes, and statuses all in a single pass, the pipeline separates medication/allergen recall from structured formatting. Pass 1 asks for a simple comma-separated substance list. Pass 2 feeds the raw transcription and that checklist back together, instructing the model to generate a structured entry for each checklist item.

This design reduces silent omission of non-active substances such as allergies, discontinued drugs, vaccines, and historical medications. It does not make omission impossible: if Pass 1 misses a substance, the downstream structured pass will not recover it.

### Schema-Constrained Structured Output

Pydantic schemas are passed to Ollama through the `format` parameter, which strongly constrains the model toward schema-valid JSON. This greatly reduces malformed output.

The code still includes retry and failure handling because schema-constrained generation should not be treated as a perfect guarantee. If validation fails after the retry loop, the parser returns a structured failure response instead of silently accepting invalid output.

### Structured Reasoning Field

Some reasoning models emit separate `<think>...</think>` traces, but unconstrained natural language can conflict with structured JSON generation. Instead, the schema includes a dedicated `reasoning` string field so the model can summarize its clinical interpretation inside the validated JSON object.

### Semantic Anchoring

The schema requires the model to extract raw text evidence before selecting standardized fields:

- `administration_text_clue`: text supporting route or administration details
- `classification_text_clue`: text supporting active, discontinued, allergy, historical, vaccine, procedural, or conditional status

This makes the output easier to inspect and helps reduce unsupported route/status guesses. It does not eliminate all extraction errors, especially when the note is ambiguous.

### Unified Medication Mention Model

Early versions used separate arrays for active medications, discontinued medications, and exclusions. That made it easier for the same drug to appear in multiple categories. The current schema first represents every medication-like mention in a unified `MedicationEntry` list with one classification field, then splits entries into active/discontinued medications and excluded mentions for the final output.

### Resumption Ledger

Batch runs can take a long time on local hardware. The batch runner writes results incrementally to a JSON ledger after every record. If the script is interrupted, it can reload the existing ledger and skip records that were already processed.

### Cloud Adaptability

The extraction logic is separated from the CLI, making it feasible to swap the Ollama backend for a cloud structured-output API. A cloud migration would mainly require replacing the model-call wrapper and adapting the structured-output/schema call format.

---

## Engineering Journey

The final architecture came from several rounds of testing, failure analysis, and redesign. This section is included because the project was not simply a one-shot LLM prompt. Most of the value came from identifying failure modes and changing the system architecture around them.

| Phase | Approach | Model Tested | Main Issue Found | Design Change |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Single-pass extraction through an Instructor wrapper | Gemma 4:12B | Long notes hit context/truncation issues, and wrapper behavior made low-level model settings harder to control | Dropped orchestration wrappers and moved to direct Ollama REST calls |
| 2 | Single-pass extraction with separate Pydantic arrays | Gemma 4:12B | Some drugs appeared in both active and excluded lists; allergy recall and route handling were inconsistent | Replaced separate arrays with one unified medication-entry schema and an explicit classification field |
| 3 | Single-pass unified schema with semantic anchoring | Gemma 4:12B | Route hallucinations improved, but non-active recall was still inconsistent | Added a checklist pre-pass focused only on substance recall |
| 4 | One LLM call per checklist item | Gemma 4:e4b | Recall improved, but runtime became too slow for batch evaluation | Switched to Qwen 2.5:14B and consolidated per-drug calls into one checklist-guided structured pass |
| 5 | Current checklist-guided two-pass architecture | Qwen 2.5:14B | Best balance so far between recall, structured output, and local runtime | Used for the current manual review and batch-run workflow |

The current version is the best-performing architecture from this project so far and provides a foundation for stronger evaluation, post-processing, and ontology integration.

---

## Evaluation

### Manual 30-Case Review

I manually reviewed a 30-case benchmark from MTSamples clinical transcription notes and compared the parser output against the source note.

Estimated performance from that review:

| Evaluation Level | Approximate Result | Notes |
|---|---:|---|
| Case-level extraction quality | ~87% | Equivalent to about 26/30 on the reviewed set |
| Medication-name recall | ~90-95% | The model usually finds clearly listed medications |
| Full structured JSON quality | ~80-85% | Most remaining errors involve frequency, route, status, or over/under-exclusion |

Common failure modes observed:

- over-inferred frequency, especially when text was ambiguous
- active medications occasionally classified as discontinued or excluded
- unclear route inference when route was not explicitly stated
- non-drug items occasionally treated as medication-like exclusions
- conditional treatment plans not always classified consistently

This benchmark is a small manual review, not a production validation study. A stronger evaluation would require a larger dataset.

### Batch Evaluation Mode

The repository also includes a 100-case batch runner that samples across multiple specialties and records runtime, success/failure status, extracted medications, excluded mentions, and model reasoning into a JSON ledger.

Important distinction: the batch runner's `success_count` measures whether the pipeline completed successfully and returned schema-valid output. It is not the same thing as clinical extraction accuracy.

Batch results are saved incrementally to:

```text
pipeline_qwen_100_results.json
```

---

## Project Structure

```text
clinical-medication-parser/
├── parser.py                 # Core two-pass extraction pipeline: prompts, Ollama calls, validation
├── medication_strict.py      # Pydantic schemas and validators
├── medication_parser.py      # CLI, batch runner, formatted output, and resumption ledger
├── mtsamples.csv             # Optional local dataset file for testing
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM runtime | Ollama |
| Model tested | Qwen 2.5 14B via `qwen2.5:14b` |
| Schema validation | Pydantic v2 |
| Data processing | pandas |
| CLI | argparse |
| Runtime style | local REST call to Ollama |
| Hardware used for benchmark | NVIDIA Tesla T4, 16 GB VRAM |

---

## Setup

### Prerequisites

- Python 3.10+
- Ollama installed locally
- Qwen 2.5 14B pulled through Ollama

```bash
ollama pull qwen2.5:14b
```

### Installation

```bash
git clone https://github.com/Capver/clinical-medication-parser.git
cd clinical-medication-parser

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## Usage

### Parse specific patient IDs from `mtsamples.csv`

```bash
python medication_parser.py --ids 4106,4452
```

### Parse custom clinical text

```bash
python medication_parser.py --text "Patient is allergic to PCN. Currently takes Lipitor 20 mg daily."
```

### Run the interactive menu

```bash
python medication_parser.py
```

Note: for long pasted text in terminal interactive mode, the current script reads from standard input. Depending on your shell, you may need to finish the pasted input with EOF, such as `Ctrl+D` on macOS/Linux or `Ctrl+Z` then Enter on Windows. For short examples, `--text` is simpler.

### Run batch mode

```bash
python medication_parser.py --batch
```

This runs the stratified 100-case batch workflow and writes incremental results to `pipeline_qwen_100_results.json`.

---

## Output Format

The final returned dictionary has this general structure:

```json
{
  "status": "Success",
  "reasoning": "Clinical rationale based on the source note.",
  "medications": [
    {
      "drug_name": "Lipitor",
      "dosage": {
        "amount": "20",
        "unit": "mg"
      },
      "frequency": "QD",
      "route": "Oral",
      "administration_text_clue": "20 mg daily",
      "classification_text_clue": "Currently takes Lipitor 20 mg daily",
      "clinical_classification": "active",
      "status": "active"
    }
  ],
  "excluded_medications": [
    {
      "drug_name": "PCN",
      "exclusion_reason": "allergy",
      "classification_text_clue": "allergic to PCN"
    }
  ]
}
```

---

## Clinical Classification Labels

The parser classifies each medication-like mention into one of the following categories:

| Label | Meaning |
|---|---|
| `active` | Current home/maintenance medication, active prescription, or regular supplement |
| `recently discontinued` | Explicitly stopped, finished, or discontinued during the current episode |
| `excluded - allergy` | Medication, food, or environmental allergen listed as an allergy/intolerance/adverse reaction |
| `excluded - acute procedural` | One-time medication given in the ER, hospital, or during a procedure |
| `excluded - historical past` | Past medication or past substance use not currently active |
| `excluded - vaccine` | Preventive immunization/vaccine mention |
| `excluded - pending/conditional` | Medication that may be started later but is not definitively active |
| `excluded - other` | Medication-like mention that should not be treated as an active medication |

---

## Data Source

Clinical transcriptions are sourced from the open-source [MTSamples](https://www.mtsamples.com/) dataset, a publicly available corpus of anonymized medical transcription samples. Used here for research and portfolio demonstration purposes.

---

## Limitations

- **Not clinically validated:** Results require human review.
- **Small benchmark:** Current accuracy estimates come from a limited manual review.
- **No drug ontology yet:** Brand/generic normalization is not implemented.
- **English only:** Multilingual notes are not supported.

---

## Room for improvement

- Add RxNorm or another drug ontology for brand/generic normalization, deduplication, and medication-code mapping.
- Add a separate allergen vocabulary/ontology for non-drug allergy mentions.
- Add confidence flags when the source text is ambiguous.
- The pipeline is currently tuned for Qwen 2.5:14B. Testing against flagship cloud models would likely provide stronger accuracy baselines.
- Fine-tuning: Build a larger gold-standard JSON dataset and fine-tune smaller local models to improve classification consistency, dosage/frequency normalization, and recall of allergy/historical mentions.

---

## Disclaimer

This repository is for education, experimentation, and portfolio demonstration only. It is not intended for diagnosis, treatment, medication reconciliation, clinical decision support, or autonomous use in patient care.
