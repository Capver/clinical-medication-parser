import pandas as pd
import json
import time
import os
import argparse
import sys
from parser import parse_single_transcription

def print_parsed_results(result: dict) -> None:
    """
    Prints the parsed results to standard output.
    """
    if result.get("status") == "Failed":
        print(f"\n❌ Parsing Failed: {result.get('error')}")
        return
        
    print(f"\n🧠 Clinical Reasoning:\n{result.get('reasoning')}\n")
    
    meds = result.get("medications", [])
    excl = result.get("excluded_medications", [])
    
    print("📋 Active & Discontinued Medications:")
    if not meds:
        print("   (None)")
    for m in meds:
        dose = m.get("dosage", {})
        amt = dose.get("amount", "not specified")
        unit = dose.get("unit", "not specified")
        if amt == "not specified":
            dose_str = "not specified"
        elif unit == "not specified":
            dose_str = f"{amt}"
        else:
            dose_str = f"{amt} {unit}"
        print(f"   • {m.get('drug_name')} | Dose: {dose_str} | Frequency: {m.get('frequency')} | Route: {m.get('route')} | Status: {m.get('status')}")
        
    print("\n🚫 Excluded Mentions & Allergies:")
    if not excl:
        print("   (None)")
    for e in excl:
        print(f"   • {e.get('drug_name')} | Reason: {e.get('exclusion_reason')} (Clue: '{e.get('classification_text_clue')}')")

def run_specific_ids(ids_str: str) -> None:
    """
    Loads specific patient IDs from mtsamples.csv and parses them.
    """
    try:
        df = pd.read_csv('mtsamples.csv')
    except Exception as e:
        print(f"Error loading mtsamples.csv: {e}")
        return
        
    ids = [int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()]
    if not ids:
        print("No valid integer IDs provided.")
        return
        
    for pid in ids:
        patient_row = df[df['ID'] == pid]
        if patient_row.empty:
            print(f"\n⚠️ Patient ID {pid} not found in mtsamples.csv.")
            continue
            
        transcription = str(patient_row.iloc[0].get('transcription', ''))
        specialty = str(patient_row.iloc[0].get('medical_specialty', ''))
        
        print(f"\n==================================================")
        print(f"Parsing Patient ID {pid} (Specialty: {specialty})")
        print(f"==================================================")
        
        result = parse_single_transcription(transcription)
        print_parsed_results(result)

def run_interactive() -> None:
    """
    Runs interactive console loop.
    """
    print("\n==================================================")
    print("Clinical Medication Parser - Interactive Mode")
    print("==================================================")
    while True:
        print("\nChoose an input type:")
        print("1) Parse specific Patient IDs from mtsamples.csv")
        print("2) Paste custom clinical transcription text")
        print("3) Exit")
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            ids_str = input("Enter comma-separated Patient IDs (e.g. 4106, 4452): ").strip()
            run_specific_ids(ids_str)
        elif choice == '2':
            print("\nPaste/type your transcription. Press Enter to process:")
            text = sys.stdin.read().strip()
            if not text:
                print("No text provided.")
                continue
            print("\nProcessing transcription...")
            result = parse_single_transcription(text)
            print_parsed_results(result)
        elif choice == '3':
            print("Exiting interactive mode.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def run_batch_evaluation() -> None:
    """
    Runs a 100-case inference run with resumption ledger.
    """
    try:
        df = pd.read_csv('mtsamples.csv')
    except Exception as e:
        print(f"Error loading mtsamples.csv: {e}")
        return
        
    specialties_distribution = {
        "Consult - History and Phy.": 30,
        "General Medicine": 30,
        "Discharge Summary": 20,
        "Cardiovascular / Pulmonary": 10,
        "Endocrinology": 10
    }
    
    sampled_dfs = []
    for specialty, count in specialties_distribution.items():
        specialty_df = df[df['medical_specialty'].str.strip() == specialty]
        specialty_df = specialty_df[specialty_df['transcription'].notna()]
        
        if len(specialty_df) < count:
            sampled_dfs.append(specialty_df)
        else:
            sampled_dfs.append(specialty_df.sample(n=count, random_state=41))
            
    test_df = pd.concat(sampled_dfs).reset_index(drop=True)
    total_records = len(test_df)
    
    output_filename = "pipeline_qwen_100_results.json"
    results = []
    success_count = 0
    
    previous_elapsed_time = 0
    # Load incremental ledger/results if they exist (interruption recovery)
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r") as f:
                saved_data = json.load(f)
                results = saved_data.get("results", [])
                success_count = saved_data.get("success_count", 0)
                previous_elapsed_time = saved_data.get("total_duration_minutes", 0) * 60
                print(f"==========================================================")
                print(f"Resuming pipeline from ledger: {len(results)}/{total_records} completed.")
                print(f"==========================================================")
        except Exception as e:
            print(f"Could not load ledger: {e}. Starting fresh.")
            results = []
            success_count = 0
            previous_elapsed_time = 0
            
    total_start_time = time.time()
    
    # Process each patient transcription record
    for i, row in test_df.iterrows():
        patient_id = int(row['ID'])
        specialty = str(row['medical_specialty'])
        
        # Interruption safety: check if this ID is already in the ledger
        if any(r['id'] == patient_id for r in results):
            print(f"[{i+1}/{total_records}] Patient ID {patient_id} already processed. Skipping.")
            continue
            
        transcription_val = row.get('transcription', '')
        text_to_parse = str(transcription_val)
        
        print(f"[{i+1}/{total_records}] Processing ID {patient_id} ({specialty}) - {len(text_to_parse)} chars...")
        
        start_time = time.time()
        result = parse_single_transcription(text_to_parse)
        elapsed = time.time() - start_time
        
        status = result.get("status", "Success")
        error_msg = result.get("error", "")
        extracted_meds = result.get("medications", [])
        excl_meds = result.get("excluded_medications", [])
        reasoning = result.get("reasoning", "")
        
        if status == "Success":
            success_count += 1
            print(f"  -> Success in {elapsed:.2f}s. Extracted {len(extracted_meds)} active/discontinued meds, {len(excl_meds)} excluded.")
        else:
            print(f"  -> Failed in {elapsed:.2f}s. Error: {error_msg}")
            
        results.append({
            "id": patient_id,
            "specialty": specialty,
            "transcription_length": len(text_to_parse),
            "status": status,
            "error": error_msg,
            "duration_seconds": elapsed,
            "reasoning": reasoning,
            "medications": extracted_meds,
            "excluded_medications": excl_meds
        })
        print("-" * 50)
        
        # Save progress incrementally to act as the ledger
        with open(output_filename, "w") as f:
            current_elapsed = previous_elapsed_time + (time.time() - total_start_time)
            json.dump({
                "total_records": total_records,
                "success_count": success_count,
                "total_duration_minutes": current_elapsed / 60,
                "avg_latency_seconds": current_elapsed / len(results) if results else 0,
                "results": results
            }, f, indent=2)
            
    total_elapsed = previous_elapsed_time + (time.time() - total_start_time)
    avg_latency = total_elapsed / total_records if total_records > 0 else 0
    
    print("\n" + "=" * 50)
    print("CHECKLIST-GUIDED QWEN 100-CASE TEST COMPLETED")
    print(f"Total time elapsed: {total_elapsed/60:.2f} minutes")
    print(f"Average latency per record: {avg_latency:.2f} seconds")
    print(f"Overall Success Rate: {success_count}/{total_records} ({success_count/total_records*100:.1f}%)")
    print("=" * 50)
    print(f"Detailed results saved to {output_filename}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clinical Medication Parser using Qwen 2.5:14B"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--batch", 
        action="store_true", 
        help="Run the 100-case evaluation batch benchmark from mtsamples.csv."
    )
    group.add_argument(
        "--ids", 
        type=str, 
        help="Comma-separated patient IDs from mtsamples.csv to parse (e.g. '4106,4452')."
    )
    group.add_argument(
        "--text", 
        type=str, 
        help="Direct clinical transcription text to parse."
    )
    group.add_argument(
        "--interactive", 
        action="store_true", 
        help="Start interactive CLI mode to paste transcriptions or enter IDs."
    )
    
    args = parser.parse_args()
    
    if args.batch:
        run_batch_evaluation()
    elif args.ids:
        run_specific_ids(args.ids)
    elif args.text:
        result = parse_single_transcription(args.text)
        print_parsed_results(result)
    elif args.interactive:
        run_interactive()
    else:
        # Fallback interactive menu if run without arguments
        print("Welcome to the Medication Parser CLI!")
        print("1) Run 100-case batch evaluation benchmark")
        print("2) Parse specific patient IDs from mtsamples.csv")
        print("3) Paste custom transcription text")
        print("4) Exit")
        choice = input("Select an option (1-4): ").strip()
        if choice == '1':
            run_batch_evaluation()
        elif choice == '2':
            ids_str = input("Enter comma-separated Patient IDs (e.g. 4106, 4452): ").strip()
            run_specific_ids(ids_str)
        elif choice == '3':
            print("\nPaste/type your transcription. Press Enter to process:")
            text = sys.stdin.read().strip()
            if text:
                result = parse_single_transcription(text)
                print_parsed_results(result)
            else:
                print("No text provided.")
        else:
            print("Exiting.")

if __name__ == "__main__":
    main()