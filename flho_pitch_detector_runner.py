#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLHO Pitch Detector - Interactive Runner Script

Detects pitches present in audio files (supports single notes and chords).
Supports both interactive selection and command-line invocation modes.

Usage examples:
    # Interactive mode (default)
    python flho_pitch_detector_runner.py

    # Command-line mode - detect a single file
    python flho_pitch_detector_runner.py run audio/C4.wav

    # Command-line mode - detect a chord file
    python flho_pitch_detector_runner.py run audio/C4D4G4A4.wav

    # Command-line mode - specify number of parallel processes
    python flho_pitch_detector_runner.py run audio/C4D4G4A4.wav -w 16

    # Command-line mode - batch process multiple files
    python flho_pitch_detector_runner.py run audio/C4.wav audio/A4.wav audio/C4D4G4A4.wav

    # Command-line mode - save results to a specified directory
    python flho_pitch_detector_runner.py run audio/C4D4G4A4.wav -o results

    # Command-line mode - verbose output
    python flho_pitch_detector_runner.py run audio/C4.wav -v
"""

import os
import sys
import time
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flho_pitch_detector_utils import PitchDetector

# ==================== Global Configuration ====================
# Directory containing audio files to detect (scanned in interactive mode)
AUDIO_DIR = 'audio'
# =============================================================


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def detect_audio_file(detector, audio_file, verbose=False):
    """
    Detect notes in a single audio file.

    Args:
        detector: PitchDetector instance
        audio_file: Path to the audio file
        verbose: Whether to show detailed information

    Returns:
        dict: Detection results
    """
    print(f"\n{'='*80}")
    print(f"Detecting file: {audio_file}")
    print(f"{'='*80}")

    # Detect notes
    result = detector.detect_notes(audio_file)

    # Print results
    detector.print_results(result)

    # If verbose mode is enabled, show all detector results
    if verbose:
        print("\nAll detector results (sorted by A/B ratio):")
        print("-" * 80)
        print(f"{'Note':<8} {'A/B Ratio':<12} {'Threshold':<12} {'DMD Freq (Hz)':<12} {'Detected':<8}")
        print("-" * 80)

        for note_result in result['all_results']:
            detected_str = "Yes" if note_result['detected'] else "No"
            print(f"{note_result['note_name']:<8} "
                  f"{note_result['ratio_mean']:<12.6f} "
                  f"{note_result['threshold']:<12.6f} "
                  f"{note_result['dmd_frequency']:<12.1f} "
                  f"{detected_str:<8}")

    return result


def interactive_mode():
    """Interactive mode."""
    print_separator()
    print("FLHO Pitch Detector - Interactive Mode")
    print_separator()

    # Get all audio files in the target directory
    audio_dir = AUDIO_DIR
    if not os.path.exists(audio_dir):
        print(f"Error: Directory '{audio_dir}' does not exist")
        sys.exit(1)

    audio_files = [f for f in os.listdir(audio_dir)
                   if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'))]

    if not audio_files:
        print(f"Error: No audio files found in '{audio_dir}'")
        sys.exit(1)

    # List all audio files
    print(f"\nFound {len(audio_files)} audio files:\n")
    for i, filename in enumerate(audio_files, 1):
        print(f"  {i}. {filename}")

    # File selection
    print(f"\nSelect files to detect (1-{len(audio_files)}), comma-separated for multiple, or 'all' for all:")
    print("(Press Enter to default to the first file)")
    choice = input("Enter: ").strip()

    if choice == '':
        # Default to first file
        selected_files = [os.path.join(audio_dir, audio_files[0])]
        print(f"Default selection: {audio_files[0]}")
    elif choice.lower() == 'all':
        # Select all files
        selected_files = [os.path.join(audio_dir, f) for f in audio_files]
        print(f"Selected all {len(audio_files)} files")
    else:
        # Select by index
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            valid_indices = [i for i in indices if 0 <= i < len(audio_files)]
            if valid_indices:
                selected_files = [os.path.join(audio_dir, audio_files[i]) for i in valid_indices]
                print(f"Selected: {', '.join(audio_files[i] for i in valid_indices)}")
            else:
                print("Invalid selection, defaulting to first file")
                selected_files = [os.path.join(audio_dir, audio_files[0])]
        except ValueError:
            print("Invalid input, defaulting to first file")
            selected_files = [os.path.join(audio_dir, audio_files[0])]

    # Choose number of parallel processes
    print(f"\nSet number of parallel processes (default: 32):")
    workers_input = input("Enter (press Enter for default): ").strip()
    if workers_input.isdigit():
        max_workers = int(workers_input)
    else:
        max_workers = 32
        print(f"Using default: {max_workers}")

    # Choose whether to show detailed results
    print(f"\nShow detailed results for all detectors? (y/N):")
    verbose_input = input("Enter (press Enter for No): ").strip().lower()
    verbose = verbose_input in ['y', 'yes']

    # Choose whether to save results
    print(f"\nSave detection results to file? (y/N):")
    save_input = input("Enter (press Enter for No): ").strip().lower()
    save_results = save_input in ['y', 'yes']

    output_dir = None
    if save_results:
        print(f"\nSpecify output directory (press Enter for output_flho_pitch_detector):")
        output_dir_input = input("Enter: ").strip()
        if output_dir_input:
            output_dir = output_dir_input
        else:
            output_dir = "output_flho_pitch_detector"
        os.makedirs(output_dir, exist_ok=True)

    # Create detector
    print(f"\n{'='*80}")
    print(f"Starting detection")
    print(f"{'='*80}")
    print(f"Files to process: {len(selected_files)}")
    print(f"Parallel processes: {max_workers}")
    print(f"Verbose mode: {'Yes' if verbose else 'No'}")

    detector = PitchDetector(
        config_path="flho_single_detector_config.json",
        audio_dir="piano_single_note/single_notes",
        max_workers=max_workers
    )

    # Batch detection
    start_time = time.time()
    all_results = []

    for i, audio_file in enumerate(selected_files, 1):
        print(f"\nProgress: {i}/{len(selected_files)}")
        result = detect_audio_file(detector, audio_file, verbose)
        all_results.append(result)

        # Save results (default to output_pitch_detection directory)
        if save_results:
            if not output_dir:
                output_dir = "output_pitch_detection"
                os.makedirs(output_dir, exist_ok=True)

            base_name = Path(audio_file).stem
            output_file = os.path.join(output_dir, f"pitch_detection_{base_name}.json")

            detector.save_results(result, output_file)

    # Summary
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"Detection complete!")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time:.2f} s")
    print(f"Average per file: {total_time/len(selected_files):.2f} s")
    print(f"\nDetection summary:")
    print("-" * 80)

    for result in all_results:
        detected_notes = ', '.join(result['detection_summary']['detected_note_names'])
        if detected_notes:
            print(f"  {result['audio_file']:<40} -> {detected_notes}")
        else:
            print(f"  {result['audio_file']:<40} -> (no notes detected)")

    print(f"{'='*80}")


def command_line_mode(args):
    """
    Command-line mode.

    Args:
        args: Command-line argument list
    """
    if not args:
        print("Usage: python flho_pitch_detector_runner.py run <audio_files...> [options]")
        print("Example: python flho_pitch_detector_runner.py run audio/C4.wav")
        print("         python flho_pitch_detector_runner.py run audio/C4D4G4A4.wav -w 16 -o results -v")
        sys.exit(1)

    # Parse arguments
    audio_files = []
    max_workers = 32
    output_dir = None
    verbose = False

    i = 0
    while i < len(args):
        arg = args[i]

        if arg.startswith('-'):
            if arg in ['-w', '--workers']:
                if i + 1 < len(args):
                    max_workers = int(args[i + 1])
                    i += 2
                else:
                    print(f"Error: {arg} requires a process count")
                    sys.exit(1)
            elif arg in ['-o', '--output-dir']:
                if i + 1 < len(args):
                    output_dir = args[i + 1]
                    os.makedirs(output_dir, exist_ok=True)
                    i += 2
                else:
                    print(f"Error: {arg} requires a directory")
                    sys.exit(1)
            elif arg in ['-v', '--verbose']:
                verbose = True
                i += 1
            else:
                print(f"Unknown option: {arg}")
                print("Available options: -w/--workers, -o/--output-dir, -v/--verbose")
                sys.exit(1)
        else:
            audio_files.append(arg)
            i += 1

    if not audio_files:
        print("Error: No audio files specified")
        sys.exit(1)

    # Validate file existence
    valid_files = []
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            valid_files.append(audio_file)
        else:
            print(f"Warning: File not found - {audio_file}")

    if not valid_files:
        print("Error: No valid audio files found")
        sys.exit(1)

    print("FLHO Pitch Detector")
    print_separator()
    print(f"Config file: flho_single_detector_config.json")
    print(f"Parallel processes: {max_workers}")
    print(f"Files to process: {len(valid_files)}")
    print(f"Verbose mode: {'Yes' if verbose else 'No'}")

    if output_dir:
        print(f"Output directory: {output_dir}")

    # Create detector
    detector = PitchDetector(
        config_path="flho_single_detector_config.json",
        audio_dir="piano_single_note/single_notes",
        max_workers=max_workers
    )

    # Batch detection
    start_time = time.time()
    all_results = []

    for i, audio_file in enumerate(valid_files, 1):
        print(f"\nProgress: {i}/{len(valid_files)}")
        result = detect_audio_file(detector, audio_file, verbose)
        all_results.append(result)

        # Save results (default to output_pitch_detection directory)
        if not output_dir:
            output_dir = "output_pitch_detection"
            os.makedirs(output_dir, exist_ok=True)

        base_name = Path(audio_file).stem
        output_file = os.path.join(output_dir, f"pitch_detection_{base_name}.json")

        detector.save_results(result, output_file)

    # Summary
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"Detection complete!")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time:.2f} s")
    print(f"Average per file: {total_time/len(valid_files):.2f} s")
    print(f"\nDetection summary:")
    print("-" * 80)

    for result in all_results:
        detected_notes = ', '.join(result['detection_summary']['detected_note_names'])
        if detected_notes:
            print(f"  {result['audio_file']:<40} -> {detected_notes}")
        else:
            print(f"  {result['audio_file']:<40} -> (no notes detected)")

    print(f"{'='*80}")


if __name__ == "__main__":
    # Determine whether to use command-line or interactive mode
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "run":
            # Command-line mode
            command_line_mode(sys.argv[2:])
        else:
            print(f"Unknown command: {command}")
            print("Available commands: run")
            print("\nOr run 'python flho_pitch_detector_runner.py' directly for interactive mode")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_mode()
