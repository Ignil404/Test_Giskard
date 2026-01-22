import subprocess
import sys
import json
import os
from datetime import datetime


def run_pipeline():
    scripts = [
        ('source/generate_questions.py', 'Generating questions'),
        ('source/view_questions.py', 'Viewing questions'),
        ('source/evaluate_rag.py', 'Evaluating system'),
        ('source/present_results.py', 'Presenting results')
    ]

    for script, desc in scripts:
        print(f'\n{desc}...')
        result = subprocess.run([sys.executable, script], cwd='.', capture_output=True, text=True)
        
        if result.returncode != 0:
            if '429' in result.stderr or 'RESOURCE_EXHAUSTED' in result.stderr:
                print(f"\n API квота превышена: {script}")
                print(f"Сообщение об ошибке:\n{result.stderr}\n")
                sys.exit(1)
            else:
                print(f"stdout:\n{result.stdout}")
                print(f"stderr:\n{result.stderr}")
                sys.exit(1)
        else:
            print(result.stdout)

def main():
    try:
        run_pipeline()
    except Exception as e:
        print(f"\nОшибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
