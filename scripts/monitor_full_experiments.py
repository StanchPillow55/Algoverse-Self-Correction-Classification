#!/usr/bin/env python3
"""
Monitor the full dataset experiments (1364 questions) for both:
1. Teacher-learner pipeline
2. GPT-4 self-correction baseline
"""
import os
import time
import json
from pathlib import Path

def check_file_progress(filepath, description):
    """Check progress of an output file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = sum(1 for _ in f)
        size = os.path.getsize(filepath)
        return f"{description}: {lines} lines, {size} bytes"
    else:
        return f"{description}: File not created yet"

def check_process_status(pid):
    """Check if process is still running"""
    try:
        os.kill(int(pid), 0)  # Send signal 0 to check if process exists
        return "Running"
    except (OSError, ValueError):
        return "Not running"

def main():
    teacher_learner_file = "outputs/teacher_learner_full1364.json"
    baseline_file = "outputs/baseline_full1364.jsonl" 
    
    # Expected PIDs (you may need to update these)
    teacher_pid = "52708"
    baseline_pid = "52982"
    
    print("🚀 Full Dataset Experiment Monitor")
    print("=================================")
    print(f"Expected dataset size: 1364 questions")
    print(f"Teacher-learner PID: {teacher_pid}")
    print(f"GPT-4 baseline PID: {baseline_pid}")
    print()
    
    while True:
        print(f"📊 Status at {time.strftime('%H:%M:%S')}:")
        print(f"Teacher-learner process: {check_process_status(teacher_pid)}")
        print(f"Baseline process: {check_process_status(baseline_pid)}")
        print()
        
        print(check_file_progress(teacher_learner_file, "Teacher-learner"))
        print(check_file_progress(baseline_file, "GPT-4 baseline"))
        print()
        
        # Check if baseline is complete by looking for summary line
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, 'r') as f:
                    lines = f.readlines()
                    if lines and '"__summary__"' in lines[-1]:
                        print("🎉 GPT-4 baseline COMPLETED!")
                        try:
                            summary = json.loads(lines[-1])
                            print(f"Final accuracy: {summary['__summary__']['exact_acc']:.3f}")
                        except:
                            pass
                        print()
            except:
                pass
        
        # Check if teacher-learner is complete
        if os.path.exists(teacher_learner_file):
            try:
                with open(teacher_learner_file, 'r') as f:
                    data = json.load(f)
                    if 'results' in data and len(data['results']) >= 1364:
                        print("🎉 Teacher-learner COMPLETED!")
                        correct = sum(1 for r in data['results'] if r.get('correct', False))
                        accuracy = correct / len(data['results'])
                        print(f"Final accuracy: {accuracy:.3f} ({correct}/{len(data['results'])})")
                        print()
            except:
                pass
        
        print("-" * 50)
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
