import subprocess
import sys
import json
import os
from datetime import datetime
from source.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def run_pipeline():
    scripts = [
        ('source/generate_questions.py', 'Generating questions'),
        ('source/view_questions.py', 'Viewing questions'),
        ('source/evaluate_rag.py', 'Evaluating system'),
        # ('source/present_results.py', 'Presenting results')
    ]

    for script, desc in scripts:
        logger.info("Running script", script=script, description=desc)
        result = subprocess.run([sys.executable, script], cwd='.', stdout=sys.stdout, stderr=sys.stderr, text=True)
        
        if result.returncode != 0:
            logger.error("Script failed", script=script)
            sys.exit(1)
        else:
            logger.info("Script completed successfully", script=script)

def main():
    try:
        logger.info("Starting pipeline")
        run_pipeline()
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.exception("Pipeline error", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
