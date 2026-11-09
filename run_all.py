"""run_all.py -- Master Runner for Instamart Customer Intelligence & Support Copilot.
Executes Part A, Part B, Part C, and Part D end-to-end.
"""

import sys
import subprocess
import time

def run_step(step_name, command):
    print(f"\n========================================================")
    print(f"RUNNING: {step_name}")
    print(f"COMMAND: {command}")
    print(f"========================================================")
    t0 = time.time()
    res = subprocess.run([sys.executable] + command.split()[1:], capture_output=True, text=True)
    duration = time.time() - t0
    
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
        
    if res.returncode != 0:
        print(f"FAILED: {step_name} with return code {res.returncode}")
        sys.exit(res.returncode)
    else:
        print(f"SUCCESS: {step_name} ({duration:.2f}s)")

def main():
    start_total = time.time()
    print("STARTING COMPLETE MASTER EXECUTION (PARTS A - D)")
    
    # Part A
    run_step("Part A: Audit Data", "python src/part_a/audit_data.py")
    run_step("Part A: Clean Data", "python src/part_a/clean_data.py")
    run_step("Part A: Build SQLite DB", "python src/part_a/build_database.py")
    run_step("Part A: Validate Data Checkpoints", "python src/part_a/validate_data.py")
    run_step("Part A: SQL Analysis", "python src/part_a/sql_analysis.py")
    run_step("Part A: Exploratory Data Analysis", "python src/part_a/exploratory_analysis.py")
    
    # Part B
    run_step("Part B: Segments & Lapse Prediction", "python src/part_b/run_part_b.py")
    
    # Part C
    run_step("Part C: Support Ticket Triage", "python src/part_c/run_part_c.py")
    
    # Part D
    run_step("Part D: Policy RAG & Refund Agent", "python src/part_d/run_part_d.py")
    
    total_time = time.time() - start_total
    print("\n========================================================")
    print(f"ALL PARTS (A, B, C, D) EXECUTED SUCCESSFULLY IN {total_time:.2f}s!")
    print("========================================================")

if __name__ == "__main__":
    main()
