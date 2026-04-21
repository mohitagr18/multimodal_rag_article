#!/usr/bin/env python3
"""
All-in-one script to run the full pipeline or test queries.

Usage:
    python run_all.py                    # Run all phases + test queries
    python run_all.py --test-only       # Run only test queries
    python run_all.py --phase 2         # Run specific phase
    python run_all.py --pdf myfile.pdf  # Use custom PDF
"""
import argparse
import subprocess
import sys
from pathlib import Path

def run_phase1(pdf_path):
    """Run Phase 1: Parse PDF"""
    print("\n" + "="*50)
    print("PHASE 1: Parsing PDF")
    print("="*50)
    result = subprocess.run(
        [sys.executable, "phase1_parse.py", pdf_path, "--engine", "real"],
        capture_output=False
    )
    return result.returncode == 0

def run_phase2():
    """Run Phase 2: Enrich"""
    print("\n" + "="*50)
    print("PHASE 2: Enriching chunks with VLM captions")
    print("="*50)
    result = subprocess.run(
        [sys.executable, "phase2_enrich.py"],
        capture_output=False
    )
    return result.returncode == 0

def run_phase3():
    """Run Phase 3: Ingest to Qdrant"""
    print("\n" + "="*50)
    print("PHASE 3: Ingesting to Qdrant")
    print("="*50)
    result = subprocess.run(
        [sys.executable, "phase3_ingest.py"],
        capture_output=False
    )
    return result.returncode == 0

def run_test_queries():
    """Run test queries"""
    print("\n" + "="*50)
    print("RUNNING TEST QUERIES")
    print("="*50)
    result = subprocess.run(
        [sys.executable, "run_test_queries.py"],
        capture_output=False
    )
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description="Run full pipeline or test queries for Multimodal RAG"
    )
    parser.add_argument(
        "--phase", 
        type=int, 
        choices=[1, 2, 3, 4],
        help="Run specific phase (1=Parse, 2=Enrich, 3=Ingest, 4=Test)"
    )
    parser.add_argument(
        "--test-only", 
        action="store_true", 
        help="Only run test queries (skip phases 1-3)"
    )
    parser.add_argument(
        "--pdf", 
        type=str, 
        default="test.pdf", 
        help="PDF file to process (default: test.pdf)"
    )
    args = parser.parse_args()
    
    # Check PDF exists
    if not Path(args.pdf).exists() and not args.test_only:
        print(f"Error: PDF file '{args.pdf}' not found!")
        return
    
    if args.test_only:
        print("Running TEST QUERIES only...")
        success = run_test_queries()
        if success:
            print("\n" + "="*50)
            print("TEST QUERIES COMPLETE!")
            print("="*50)
            print("Results saved to TEST_RESULTS.md")
        return
    
    # Determine which phases to run
    if args.phase:
        phases = [args.phase]
    else:
        phases = [1, 2, 3, 4]
    
    print("="*50)
    print("MULTIMODAL RAG PIPELINE")
    print("="*50)
    print(f"PDF: {args.pdf}")
    print(f"Phases: {phases}")
    
    for phase in phases:
        if phase == 1:
            success = run_phase1(args.pdf)
            if not success:
                print("Phase 1 (Parse) FAILED!")
                return
        elif phase == 2:
            success = run_phase2()
            if not success:
                print("Phase 2 (Enrich) FAILED!")
                return
        elif phase == 3:
            success = run_phase3()
            if not success:
                print("Phase 3 (Ingest) FAILED!")
                return
        elif phase == 4:
            success = run_test_queries()
            if not success:
                print("Test queries FAILED!")
                return
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    print("\nTo query interactively, run:")
    print("  python phase4_retrieve.py")

if __name__ == "__main__":
    main()
