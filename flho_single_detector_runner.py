"""
FLHO Resonance Detector - Unified Entry Script

Usage example: demonstrates how to create a detector from JSON configuration and perform analysis.
"""

import sys
import os
from flho_single_detector_utils import FLHODetectorConfig, FLHODetector

# Global configuration
AUDIO_DIR = 'audio'  # Unified audio directory for all detectors
# AUDIO_DIR = 'single_note_and_synthesized_chords_trainset'  # Unified audio directory for all detectors


def print_separator(char="=", length=60):
    """Print a separator line."""
    print(char * length)


def run_single_detector(detector_name, files_to_process=None):
    """
    Run a single detector.

    Args:
        detector_name: Detector name (c1, c2, cs2, a3, a4)
        files_to_process: List of files to process; None uses default files
    """
    # Load configuration
    config_manager = FLHODetectorConfig()

    # Get detector configuration
    detector_config = config_manager.get_detector_config(detector_name)

    # Create detector instance
    detector = FLHODetector(detector_config, audio_dir=AUDIO_DIR)

    print_separator()
    print(f"FLHO Resonance Pitch Detector - {detector.target_note}")
    print_separator()
    print(f"Natural frequency f0 = {detector.f_natural} Hz")
    print(f"Natural angular frequency omega_0 = {detector.omega_0:.2f} rad/s")
    print(f"Drive strength F = {detector.F_drive}")
    print(f"Bandwidth = {detector.bandwidth_octaves} octaves")
    print(f"Hankel rank m = {detector.hankel_m}")
    print(f"Normalization type: {detector.normalization_type}")

    # Build preprocessing steps list
    preprocessing_steps = []
    if detector.rectification:
        preprocessing_steps.append("Rectification")
    preprocessing_steps.append("Bandpass Filter")
    if detector.centroid_weighted:
        preprocessing_steps.append("Spectral Centroid Weighting")

    print(f"Preprocessing steps: {' -> '.join(preprocessing_steps)}")
    print_separator()

    # Determine files to process
    if files_to_process is None:
        files_to_process = detector.default_files

    # Validate file existence
    valid_files = []
    for filename in files_to_process:
        file_path = os.path.join(detector.audio_dir, filename)
        if os.path.exists(file_path):
            valid_files.append(filename)
        else:
            print(f"Warning: File {filename} does not exist, skipped")

    if not valid_files:
        print("Error: No valid files")
        return

    # Process all files
    all_results = []
    for i, filename in enumerate(valid_files, 1):
        audio_file_path = os.path.join(detector.audio_dir, filename)

        if len(valid_files) > 1:
            print(f"\nProgress: {i}/{len(valid_files)}")

        result = detector.process_audio_file(audio_file_path)
        if result:
            all_results.append(result)

    # Output results summary
    if all_results:
        print("\n" + "=" * 60)
        print(f"Experiment Results Summary - FLHO Resonance Pitch Detection ({detector.target_note})")
        print("=" * 60)

        target_key = f'has_{detector.target_note.replace("#", "s")}'
        target_results = [r for r in all_results if r[target_key]]
        non_target_results = [r for r in all_results if not r[target_key]]

        print(f"\nFiles containing {detector.target_note} ({len(target_results)} files):")
        print("-" * 60)
        if target_results:
            for res in target_results:
                print(f"  {res['audio_name']:<25} A/B = {res['ratio_mean']:.6f}")
            avg_target = sum([r['ratio_mean'] for r in target_results]) / len(target_results)
            print(f"  Average A/B = {avg_target:.6f}")
        else:
            print("  None")

        print(f"\nFiles not containing {detector.target_note} ({len(non_target_results)} files):")
        print("-" * 60)
        if non_target_results:
            for res in non_target_results:
                print(f"  {res['audio_name']:<25} A/B = {res['ratio_mean']:.6f}")
            avg_non_target = sum([r['ratio_mean'] for r in non_target_results]) / len(non_target_results)
            print(f"  Average A/B = {avg_non_target:.6f}")
        else:
            print("  None")

        # Separation and margin analysis
        if target_results and non_target_results and len(all_results) > 1:
            print(f"\n{'='*60}")
            print(f"Separation and Margin Analysis")
            print(f"{'='*60}")

            target_ratios = [r['ratio_mean'] for r in target_results]
            non_target_ratios = [r['ratio_mean'] for r in non_target_results]

            min_target = min(target_ratios)
            max_target = max(target_ratios)
            mean_target = sum(target_ratios) / len(target_ratios)
            std_target = (sum((x - mean_target)**2 for x in target_ratios) / len(target_ratios)) ** 0.5

            min_non_target = min(non_target_ratios)
            max_non_target = max(non_target_ratios)
            mean_non_target = sum(non_target_ratios) / len(non_target_ratios)
            std_non_target = (sum((x - mean_non_target)**2 for x in non_target_ratios) / len(non_target_ratios)) ** 0.5

            separation = abs(mean_target - mean_non_target)
            margin = min_target - max_non_target
            relative_margin = margin / max_non_target * 100 if max_non_target != 0 else 0

            print(f"\nWith {detector.target_note} group ({len(target_results)} files):")
            print(f"  Min A/B = {min_target:.6f}")
            print(f"  Max A/B = {max_target:.6f}")
            print(f"  Mean A/B = {mean_target:.6f}")
            print(f"  Std dev  = {std_target:.6f}")

            print(f"\nWithout {detector.target_note} group ({len(non_target_results)} files):")
            print(f"  Min A/B = {min_non_target:.6f}")
            print(f"  Max A/B = {max_non_target:.6f}")
            print(f"  Mean A/B = {mean_non_target:.6f}")
            print(f"  Std dev  = {std_non_target:.6f}")

            print(f"\nSeparation metrics:")
            print(f"  Separation (|mu_target - mu_non|) = {separation:.6f}")
            print(f"  Margin (min_target - max_non)     = {margin:.6f}")
            print(f"  Relative margin                   = {relative_margin:.2f}%")

            if margin > 0:
                print(f"\n  OK - Groups are fully separated (margin > 0)")
                print(f"  Recommended threshold = {(min_target + max_non_target) / 2:.6f}")
            else:
                print(f"\n  WARN - Groups overlap (margin < 0)")
                print(f"  Cannot find a perfect threshold")

            print(f"{'='*60}")

        # Detailed data table
        print("\nDetailed data table:")
        print(f"  {'File':<25} {'Has ' + detector.target_note:<10} {'A/B':<12} {'DMD(Hz)':<12}")
        print("-" * 63)
        for res in all_results:
            target_str = "Yes" if res[target_key] else "No"
            print(f"  {res['audio_name']:<28} {target_str:<7} {res['ratio_mean']:<14.6f} {res['dmd_frequency']:<12.1f}")

        # Top 10 by A/B ratio
        print(f"\n{'='*63}")
        print(f"  Top 10 by A/B Ratio")
        print(f"{'='*63}")
        sorted_results = sorted(all_results, key=lambda x: x['ratio_mean'], reverse=True)
        top_n = min(10, len(sorted_results))
        for i, res in enumerate(sorted_results[:top_n], 1):
            target_str = "Yes" if res[target_key] else "No"
            print(f"  {i}. {res['audio_name']:<25} A/B = {res['ratio_mean']:.6f} (Has {detector.target_note}: {target_str})")
        print(f"{'='*63}")

        # Visualization
        detector.visualize_results(all_results)


def list_available_detectors():
    """List all available detectors."""
    config_manager = FLHODetectorConfig()
    detectors = config_manager.get_detector_names()

    print("\nAvailable pitch detectors:")
    print("-" * 60)
    for name in detectors:
        config = config_manager.get_detector_config(name)
        print(f"  {name:<8} - {config['description']}")
        print(f"           Normalization: {config['processing']['normalization_type']}")

        # Build preprocessing steps
        steps = []
        if config['processing'].get('rectification', False):
            steps.append("Rectification")
        steps.append("Bandpass Filter")
        if config['processing'].get('centroid_weighted', False):
            steps.append("Spectral Centroid Weighting")

        print(f"           Preprocessing: {' -> '.join(steps)}")
        print()


if __name__ == "__main__":
    # Command-line argument parsing
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "list":
            # List all detectors
            list_available_detectors()

        elif command == "run":
            # Run a specified detector
            if len(sys.argv) < 3:
                print("Usage: python flho_pitch_detector_runner.py run <detector_name> [file1 file2 ...]")
                print("Example: python flho_pitch_detector_runner.py run c1 C1.wav C4.wav")
                sys.exit(1)

            detector_name = sys.argv[2].upper()  # Convert to uppercase to match config keys
            files = sys.argv[3:] if len(sys.argv) > 3 else None
            run_single_detector(detector_name, files)

        else:
            print(f"Unknown command: {command}")
            print("Available commands: list, run")
            sys.exit(1)

    else:
        # Interactive mode
        print("=" * 60)
        print("FLHO Resonance Pitch Detector - Unified Tool System")
        print("=" * 60)

        config_manager = FLHODetectorConfig()
        detectors = config_manager.get_detector_names()

        print("\nAvailable detectors:")
        for i, name in enumerate(detectors, 1):
            config = config_manager.get_detector_config(name)
            print(f"  {i}. {name} - {config['description']}")

        print(f"\nSelect detector (1-{len(detectors)}) or enter name:")
        choice = input("Enter: ").strip()

        # Determine if input is a number or a name
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(detectors):
                detector_name = detectors[idx]
            else:
                print("Invalid selection")
                sys.exit(1)
        else:
            detector_name = choice.upper()  # Convert to uppercase to match config keys
            if detector_name not in detectors:
                print(f"Unknown detector: {detector_name}")
                sys.exit(1)

        # Get detector configuration (must happen before file selection)
        detector_config = config_manager.get_detector_config(detector_name)

        # Get all files in audio directory
        audio_files_list = [f for f in os.listdir(AUDIO_DIR)
                       if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'))]

        if not audio_files_list:
            print(f"\nError: No audio files found in {AUDIO_DIR}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Select files to process")
        print("=" * 60)
        print(f"\nFound the following audio files:\n")

        for i, filename in enumerate(audio_files_list, 1):
            print(f"  {i}. {filename}")

        default_files = detector_config['default_files']
        valid_defaults = [f for f in default_files if f in audio_files_list]
        default_str = ', '.join(valid_defaults) if valid_defaults else 'None'

        print(f"\nSelect files to process (1-{len(audio_files_list)}), comma-separated for multiple, or 'all' for all:")
        print(f"(Press Enter to use defaults: {default_str})")
        choice = input("Enter: ").strip().lower()

        if choice == '':
            # Use default files
            if valid_defaults:
                files_to_process = valid_defaults
                print(f"Using defaults: {', '.join(valid_defaults)}")
            else:
                files_to_process = [audio_files_list[0]]
                print(f"Defaulting to first file: {audio_files_list[0]}")
        elif choice == 'all':
            # Process all files
            files_to_process = audio_files_list
            print(f"Selected all {len(audio_files_list)} files")
        else:
            # Select by index
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                valid_indices = [i for i in indices if 0 <= i < len(audio_files_list)]
                if valid_indices:
                    files_to_process = [audio_files_list[i] for i in valid_indices]
                    print(f"Selected: {', '.join(files_to_process)}")
                else:
                    files_to_process = [audio_files_list[0]]
                    print(f"Invalid selection, defaulting to first file: {audio_files_list[0]}")
            except ValueError:
                files_to_process = [audio_files_list[0]]
                print(f"Invalid input, defaulting to first file: {audio_files_list[0]}")

        run_single_detector(detector_name, files_to_process)
