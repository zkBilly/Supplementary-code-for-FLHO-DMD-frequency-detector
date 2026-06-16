#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLHO Single-Note Detector - Test Set Batch Evaluation Script

Performs batch detection on the synthesized_chords_testset and
synthesized_chords_testset_centered5octaves datasets, computing per-file
and overall precision, recall, and F1 scores.

Detector usage rules:
  - synthesized_chords_testset: uses all available FLHO detectors (A0-C8)
  - synthesized_chords_testset_centered5octaves: uses only detectors within C2-B6

Usage:
    python flho_testset_evaluator.py                    # Process both datasets
    python flho_testset_evaluator.py --dataset testset  # Process testset only
    python flho_testset_evaluator.py --dataset centered # Process centered5octaves only
    python flho_testset_evaluator.py --workers 16       # Specify parallel process count
"""

import os
import sys
import json
import re
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed as thread_as_completed

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flho_pitch_detector_utils import PitchDetector

# ==================== Global Configuration ====================
# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET_PATHS = {
    'testset': os.path.join(PROJECT_ROOT, 'synthesized_chords_testset'),
    'centered': os.path.join(PROJECT_ROOT, 'synthesized_chords_testset_centered5octaves'),
}

# Detector range configuration
DETECTOR_RANGES = {
    'testset': None,            # None means use all detectors
    'centered': ('C2', 'B6'),   # (min_note, max_note)
}

# Configuration file
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'flho_single_detector_config.json')
AUDIO_DIR = os.path.join(PROJECT_ROOT, 'piano_single_note', 'single_notes')

# Output directory
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'flho_testset_evaluation_results')

# Default number of parallel processes
DEFAULT_MAX_WORKERS = 32


# ==================== Note Parsing Utilities ====================

# Note name -> semitone index mapping (C0 = 0)
NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
}

# Flat -> sharp equivalent mapping
FLAT_TO_SHARP = {
    'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E',
    'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
}

# Filename note regex: matches a single note like C4, D#3, Eb5
NOTE_PATTERN = re.compile(r'([A-Ga-g])(#|b)?(\d+)')


def note_to_semitone(note_name):
    """
    Convert a note name to a semitone index (C0 = 0).

    Args:
        note_name: Note name, e.g. 'C4', 'D#3', 'Eb5'

    Returns:
        int: Semitone index, C0=0, C#0=1, ..., returns -1 on parse failure
    """
    match = NOTE_PATTERN.match(note_name)
    if not match:
        return -1

    letter = match.group(1).upper()
    accidental = match.group(2) or ''
    octave = int(match.group(3))

    # Handle flat accidentals
    if accidental == 'b':
        flat_key = letter + 'b'
        if flat_key in FLAT_TO_SHARP:
            sharp_equiv = FLAT_TO_SHARP[flat_key]
            letter = sharp_equiv[0]
            accidental = '#' if '#' in sharp_equiv else ''
            # For pure letters (e.g. Cb->B), decrement octave
            if '#' not in sharp_equiv:
                octave -= 1

    # Build lookup key
    key = letter + '#' if accidental == '#' else letter
    semitone = NOTE_TO_SEMITONE.get(key)
    if semitone is None:
        return -1

    return octave * 12 + semitone


def parse_ground_truth_notes(filename):
    """
    Parse ground truth note list from audio filename.

    Filename format example: 'A#0C#1E1.wav' -> ['A#0', 'C#1', 'E1']

    Args:
        filename: Audio filename (without path)

    Returns:
        list[str]: List of note names
    """
    basename = os.path.splitext(filename)[0]
    notes = []
    for match in NOTE_PATTERN.finditer(basename):
        letter = match.group(1).upper()
        accidental = match.group(2) or ''
        octave = match.group(3)
        note = f"{letter}{accidental}{octave}"
        notes.append(note)
    return notes


def is_note_in_range(note_name, min_note, max_note):
    """
    Check whether a note is within the specified range.

    Args:
        note_name: Note name
        min_note: Range lower bound
        max_note: Range upper bound

    Returns:
        bool
    """
    note_val = note_to_semitone(note_name)
    min_val = note_to_semitone(min_note)
    max_val = note_to_semitone(max_note)
    if note_val < 0 or min_val < 0 or max_val < 0:
        return False
    return min_val <= note_val <= max_val


# ==================== Detector with Note Range Filtering ====================

class FilteredPitchDetector(PitchDetector):
    """
    FLHO multi-note detector with note range filtering support.

    Inherits from PitchDetector; can specify a detector range at initialization
    and will only use detectors within that range.
    """

    def __init__(self, config_path=CONFIG_PATH, audio_dir=AUDIO_DIR,
                 max_workers=None, note_range=None):
        """
        Args:
            config_path: Path to configuration file
            audio_dir: Audio file directory
            max_workers: Maximum number of parallel processes
            note_range: (min_note, max_note) tuple, None means use all detectors
        """
        super().__init__(config_path, audio_dir, max_workers)
        if note_range is not None:
            min_note, max_note = note_range
            self.detector_names = [
                name for name in self.detector_names
                if is_note_in_range(name, min_note, max_note)
            ]
            self._note_range = note_range
        else:
            self._note_range = None

    @property
    def note_range_description(self):
        """Return a human-readable description of the detector range."""
        if self._note_range is None:
            return "ALL (A0-C8)"
        return f"{self._note_range[0]}-{self._note_range[1]}"


# ==================== Evaluation Metric Computation ====================

def compute_metrics(detected_notes, ground_truth_notes):
    """
    Compute detection evaluation metrics for a single file.

    Args:
        detected_notes: List of detected note names
        ground_truth_notes: List of ground truth note names

    Returns:
        dict: Contains precision, recall, f1_score, tp, fp, fn
    """
    detected_set = set(detected_notes)
    truth_set = set(ground_truth_notes)

    tp = len(detected_set & truth_set)
    fp = len(detected_set - truth_set)
    fn = len(truth_set - detected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'detected_notes': sorted(detected_notes),
        'ground_truth_notes': sorted(ground_truth_notes),
    }


def aggregate_metrics(file_metrics_list):
    """
    Aggregate evaluation metrics across all files, computing dataset-wide averages.

    Args:
        file_metrics_list: List of per-file metric dictionaries

    Returns:
        dict: Aggregated metrics
    """
    if not file_metrics_list:
        return {'avg_precision': 0.0, 'avg_recall': 0.0, 'avg_f1_score': 0.0,
                'total_files': 0}

    total_p = sum(m['precision'] for m in file_metrics_list)
    total_r = sum(m['recall'] for m in file_metrics_list)
    total_f1 = sum(m['f1_score'] for m in file_metrics_list)
    n = len(file_metrics_list)

    return {
        'avg_precision': total_p / n,
        'avg_recall': total_r / n,
        'avg_f1_score': total_f1 / n,
        'total_files': n,
    }


# ==================== Dataset Evaluation Engine ====================

class DatasetEvaluator:
    """Dataset evaluation engine: iterates files, detects, computes metrics, outputs reports."""

    def __init__(self, dataset_name, dataset_path, detector_range,
                 max_workers=DEFAULT_MAX_WORKERS, output_root=OUTPUT_ROOT,
                 parallel_files=1):
        """
        Args:
            dataset_name: Dataset identifier name
            dataset_path: Dataset directory path
            detector_range: Detector range (min_note, max_note) or None
            max_workers: Number of parallel processes (per-file internal detector parallelism)
            output_root: Output root directory
            parallel_files: Number of files to process concurrently (>1 uses thread pool;
                            each file's internal process count is auto-adjusted to
                            max_workers // parallel_files)
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.detector_range = detector_range
        self.max_workers = max_workers
        self.parallel_files = max(1, parallel_files)
        self.output_root = output_root

        self.logger = logging.getLogger(f'evaluator.{dataset_name}')
        self.file_results = []      # Per-file evaluation results
        self.all_metrics = []       # Per-file metrics
        self.start_time = None
        self.end_time = None

        # Thread-safe locks
        self._results_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._completed_count = 0
        self._total_files = 0

    def get_audio_files(self):
        """Get a sorted list of all audio filenames in the dataset."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        files = sorted([
            f for f in os.listdir(self.dataset_path)
            if f.lower().endswith('.wav')
        ])
        return files

    def _process_single_file(self, filename, detector):
        """Process a single audio file (thread-safe)."""
        audio_path = os.path.join(self.dataset_path, filename)

        # Parse ground truth
        ground_truth = parse_ground_truth_notes(filename)

        # Detect
        file_start = time.time()
        try:
            result = detector.detect_notes(audio_path)
            detected_names = result['detection_summary']['detected_note_names']
        except Exception as e:
            self.logger.error(f"Detection failed [{filename}]: {e}")
            result = None
            detected_names = []

        file_elapsed = time.time() - file_start

        # Compute metrics
        metrics = compute_metrics(detected_names, ground_truth)

        # Show single-file result
        self.logger.info(
            f"[OK] {filename}  "
            f"GT={ground_truth}  "
            f"Detected={detected_names}  "
            f"P={metrics['precision']:.3f} "
            f"R={metrics['recall']:.3f} "
            f"F1={metrics['f1_score']:.3f} "
            f"TP={metrics['true_positives']} "
            f"FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} "
            f"[{file_elapsed:.1f}s]"
        )

        # Thread-safe result storage
        file_entry = {
            'filename': filename,
            'ground_truth': ground_truth,
            'detected_notes': detected_names,
            'metrics': metrics,
            'detection_summary': result['detection_summary'] if result else None,
            'elapsed_seconds': file_elapsed,
        }

        with self._results_lock:
            self.file_results.append(file_entry)
            self.all_metrics.append(metrics)

        # Thread-safe progress update
        with self._progress_lock:
            self._completed_count += 1
            completed = self._completed_count
            total = self._total_files

            if completed % max(1, total // 20) == 0 or completed == total:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / completed
                remaining = (total - completed) * avg_time
                self.logger.info(
                    f"--- Progress [{completed}/{total}] ({completed*100//total}%): "
                    f"elapsed {elapsed:.0f}s, est. remaining {remaining:.0f}s ---"
                )

    def run(self):
        """Execute the full dataset evaluation pipeline (supports multi-file concurrency)."""
        self.logger.info("=" * 80)
        self.logger.info(f"Starting evaluation for dataset: {self.dataset_name}")
        self.logger.info(f"Dataset path: {self.dataset_path}")
        self.logger.info(f"Detector range: {self.detector_range or 'ALL (A0-C8)'}")
        self.logger.info(f"Total parallel processes: {self.max_workers}")
        self.logger.info(f"Concurrent files: {self.parallel_files}")
        self.logger.info("=" * 80)

        # Get file list
        audio_files = self.get_audio_files()
        self._total_files = len(audio_files)
        if self._total_files == 0:
            self.logger.warning("No .wav files found in dataset!")
            return

        self.logger.info(f"Files to process: {self._total_files}")

        # When processing concurrently, scale down per-file process count proportionally
        # to avoid total processes exceeding max_workers * 2
        per_file_workers = max(1, self.max_workers // self.parallel_files)
        if self.parallel_files > 1:
            self.logger.info(
                f"Per-file internal processes: {per_file_workers} "
                f"(total process cap: {per_file_workers * self.parallel_files})"
            )
        else:
            per_file_workers = self.max_workers

        # Create detector (all threads share the same instance; detect_notes creates independent process pools internally)
        detector = FilteredPitchDetector(
            config_path=CONFIG_PATH,
            audio_dir=AUDIO_DIR,
            max_workers=per_file_workers,
            note_range=self.detector_range,
        )
        self.logger.info(f"Available detectors: {len(detector.detector_names)}")

        # Start detection
        self.start_time = time.time()
        self.file_results = []
        self.all_metrics = []
        self._completed_count = 0

        if self.parallel_files <= 1:
            # ---- Sequential mode (original behavior) ----
            for filename in audio_files:
                self._process_single_file(filename, detector)
        else:
            # ---- Concurrent mode ----
            self.logger.info(
                f"Using ThreadPoolExecutor(max_workers={self.parallel_files}) for concurrent file processing"
            )
            with ThreadPoolExecutor(max_workers=self.parallel_files) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, filename, detector): filename
                    for filename in audio_files
                }
                for future in thread_as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"File processing exception [{filename}]: {e}")

        self.end_time = time.time()
        total_elapsed = self.end_time - self.start_time

        # Sort results by filename (concurrent mode does not guarantee order)
        self.file_results.sort(key=lambda x: x['filename'])

        # Compute overall metrics
        overall = aggregate_metrics(self.all_metrics)

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Dataset '{self.dataset_name}' evaluation complete!")
        self.logger.info(f"Total time: {total_elapsed:.2f}s "
                         f"({total_elapsed/60:.1f}min)")
        self.logger.info(f"Average per file: {total_elapsed/self._total_files:.2f}s")
        self.logger.info("-" * 40)
        self.logger.info(f"Avg Precision: {overall['avg_precision']:.4f}")
        self.logger.info(f"Avg Recall:    {overall['avg_recall']:.4f}")
        self.logger.info(f"Avg F1-Score:  {overall['avg_f1_score']:.4f}")
        self.logger.info("=" * 80)

        return {
            'dataset_name': self.dataset_name,
            'detector_range': f"{self.detector_range[0]}-{self.detector_range[1]}"
                              if self.detector_range else 'ALL',
            'total_files': self._total_files,
            'total_elapsed_seconds': total_elapsed,
            'overall_metrics': overall,
            'file_results': self.file_results,
        }

    def save_results(self, summary):
        """Save evaluation results as a JSON file."""
        dataset_output_dir = os.path.join(
            self.output_root, f'evaluation_{self.dataset_name}')
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Save detailed results (complete per-file information)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detail_path = os.path.join(
            dataset_output_dir, f'detailed_results_{timestamp}.json')

        serializable = {
            'dataset_name': summary['dataset_name'],
            'detector_range': summary['detector_range'],
            'total_files': summary['total_files'],
            'total_elapsed_seconds': summary['total_elapsed_seconds'],
            'overall_metrics': summary['overall_metrics'],
            'per_file_results': [
                {
                    'filename': r['filename'],
                    'ground_truth': r['ground_truth'],
                    'detected_notes': r['detected_notes'],
                    'metrics': r['metrics'],
                    'elapsed_seconds': r['elapsed_seconds'],
                }
                for r in summary['file_results']
            ],
        }

        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\nDetailed results saved to: {detail_path}")

        # Save concise summary
        summary_path = os.path.join(
            dataset_output_dir, f'summary_{timestamp}.json')
        summary_data = {
            'dataset_name': summary['dataset_name'],
            'detector_range': summary['detector_range'],
            'total_files': summary['total_files'],
            'total_elapsed_seconds': summary['total_elapsed_seconds'],
            'overall_metrics': summary['overall_metrics'],
            'worst_files': self._get_extreme_files(n=5, best=False),
            'best_files': self._get_extreme_files(n=5, best=True),
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Summary saved to: {summary_path}")
        return detail_path, summary_path

    def _get_extreme_files(self, n=5, best=True):
        """Get the n best/worst performing files."""
        sorted_results = sorted(
            self.file_results,
            key=lambda x: x['metrics']['f1_score'],
            reverse=best,
        )
        top_n = sorted_results[:n]
        return [
            {
                'filename': r['filename'],
                'ground_truth': r['ground_truth'],
                'detected_notes': r['detected_notes'],
                'f1_score': r['metrics']['f1_score'],
                'precision': r['metrics']['precision'],
                'recall': r['metrics']['recall'],
            }
            for r in top_n
        ]


# ==================== Main Program ====================

def setup_logging(output_root=OUTPUT_ROOT):
    """Configure the logging system."""
    log_dir = os.path.join(output_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'evaluation_{timestamp}.log')

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers (avoid duplication)
    root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_file


def print_final_report(all_summaries, log_file):
    """Print the final comprehensive report."""
    logger = logging.getLogger('report')
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("  FLHO Single-Note Detector - Test Set Evaluation Final Report")
    logger.info("=" * 80)

    for summary in all_summaries:
        overall = summary['overall_metrics']
        logger.info(f"  Dataset: {summary['dataset_name']}")
        logger.info(f"    Detector range: {summary['detector_range']}")
        logger.info(f"    Total files: {summary['total_files']}")
        logger.info(f"    Total time: {summary['total_elapsed_seconds']:.1f}s"
                    f" ({summary['total_elapsed_seconds']/60:.1f}min)")
        logger.info(f"    {'-' * 68}")
        logger.info(f"    Avg Precision: {overall['avg_precision']:.4f}")
        logger.info(f"    Avg Recall:    {overall['avg_recall']:.4f}")
        logger.info(f"    Avg F1-Score:  {overall['avg_f1_score']:.4f}")
        logger.info("")

    logger.info("=" * 80)
    logger.info(f"\nFull log saved to: {log_file}")


def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description='FLHO Single-Note Detector - Test Set Batch Evaluation',
    )
    parser.add_argument(
        '--dataset', '-d',
        choices=['testset', 'centered', 'all'],
        default='all',
        help='Select dataset(s) to evaluate (default: all)',
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f'Number of parallel processes (default: {DEFAULT_MAX_WORKERS})',
    )
    parser.add_argument(
        '--output', '-o',
        default=OUTPUT_ROOT,
        help=f'Output directory (default: {OUTPUT_ROOT})',
    )
    parser.add_argument(
        '--parallel-files', '-p',
        type=int,
        default=1,
        help='Number of files to process concurrently (>1 uses thread pool; '
             'per-file internal process count auto-adjusted to max_workers // parallel_files; '
             'recommended value is 1/4 to 1/2 of CPU cores, e.g. -p 4 for 32-core CPU)',
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    log_file = setup_logging(args.output)
    logger = logging.getLogger('main')
    logger.info("FLHO test set evaluation script started")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Parallel processes: {args.workers}")

    # Determine which datasets to evaluate
    datasets_to_evaluate = []
    if args.dataset in ('testset', 'all'):
        datasets_to_evaluate.append('testset')
    if args.dataset in ('centered', 'all'):
        datasets_to_evaluate.append('centered')

    logger.info(f"Will evaluate {len(datasets_to_evaluate)} dataset(s): {datasets_to_evaluate}")

    all_summaries = []

    for ds_name in datasets_to_evaluate:
        ds_path = DATASET_PATHS[ds_name]
        detector_range = DETECTOR_RANGES[ds_name]

        if not os.path.exists(ds_path):
            logger.error(f"Dataset directory not found, skipping: {ds_path}")
            continue

        evaluator = DatasetEvaluator(
            dataset_name=ds_name,
            dataset_path=ds_path,
            detector_range=detector_range,
            max_workers=args.workers,
            output_root=args.output,
            parallel_files=args.parallel_files,
        )

        summary = evaluator.run()
        if summary:
            evaluator.save_results(summary)
            all_summaries.append(summary)

    # Print final report
    if all_summaries:
        print_final_report(all_summaries, log_file)
    else:
        logger.warning("No datasets were successfully evaluated.")


if __name__ == '__main__':
    main()
