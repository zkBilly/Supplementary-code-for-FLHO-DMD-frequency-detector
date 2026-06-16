"""
FLHO Pitch Detector - Multi-Note Detection

Runs 88 parallel FLHO pitch detectors to detect notes in audio files.
Detection decisions are based on the ab_threshold parameter from the config file.
"""

import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from flho_single_detector_utils import FLHODetectorConfig, FLHODetector


class PitchDetector:
    """FLHO multi-note detector - runs all 88 detectors in parallel"""

    def __init__(self, config_path="flho_single_detector_config.json", audio_dir="piano_single_note/single_notes", max_workers=None):
        """
        Initialize the multi-note detector.

        Args:
            config_path: Path to the configuration file
            audio_dir: Directory containing audio files
            max_workers: Maximum number of parallel processes, defaults to 32 (optimized for 32-core CPU)
        """
        self.config_manager = FLHODetectorConfig(config_path)
        self.audio_dir = audio_dir
        # Optimized for 32-core CPU, default 32 processes
        self.max_workers = max_workers or 32
        self.detector_names = self.config_manager.get_detector_names()

    def _run_single_detector(self, detector_name, audio_file_path):
        """
        Run a single detector (for parallel processing).
        Redirects stdout to avoid garbled terminal output.
        """
        import sys
        import io

        # Temporarily redirect stdout to suppress verbose detector output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Get detector configuration
            detector_config = self.config_manager.get_detector_config(detector_name)

            # Create detector instance
            detector = FLHODetector(detector_config, audio_dir=self.audio_dir)

            # Process audio file
            result = detector.process_audio_file(audio_file_path)

            if result:
                # Add detection decision
                threshold = detector_config.get('ab_threshold', 1.0)
                result['detected'] = result['ratio_mean'] > threshold
                result['threshold'] = threshold
                result['note_name'] = detector.target_note

                return {
                    'success': True,
                    'note_name': detector.target_note,
                    'ratio_mean': result['ratio_mean'],
                    'threshold': threshold,
                    'detected': result['detected'],
                    'dmd_frequency': result['dmd_frequency']
                }
            else:
                return {
                    'success': False,
                    'note_name': detector_name,
                    'error': 'Processing failed'
                }

        except Exception as e:
            return {
                'success': False,
                'note_name': detector_name,
                'error': str(e)
            }
        finally:
            # Restore stdout
            sys.stdout = old_stdout

    def detect_notes(self, audio_file_path):
        """
        Detect notes in an audio file.

        Args:
            audio_file_path: Path to the audio file

        Returns:
            dict: Detection results
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        print(f"\nStarting detection on: {os.path.basename(audio_file_path)}")
        print(f"Using {self.max_workers} parallel processes for {len(self.detector_names)} detectors")
        print("-" * 60)

        start_time = time.time()
        results = []

        # Process all detectors in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_detector = {
                executor.submit(self._run_single_detector, detector_name, audio_file_path): detector_name
                for detector_name in self.detector_names
            }

            # Collect results
            completed = 0
            for future in as_completed(future_to_detector):
                detector_name = future_to_detector[future]
                completed += 1

                try:
                    result = future.result()
                    results.append(result)

                    # Show progress (every 10 completions or when all done)
                    if completed % 10 == 0 or completed == len(self.detector_names):
                        print(f"Progress: {completed}/{len(self.detector_names)} ({completed/len(self.detector_names)*100:.1f}%)")

                except Exception as e:
                    print(f"Detector {detector_name} execution exception: {e}")
                    results.append({
                        'success': False,
                        'note_name': detector_name,
                        'error': str(e)
                    })

        end_time = time.time()
        processing_time = end_time - start_time

        # Organize results
        detected_notes = []
        all_results = []

        for result in results:
            if result['success']:
                all_results.append(result)
                if result['detected']:
                    detected_notes.append(result)
            else:
                print(f"Warning: Detector {result['note_name']} failed - {result.get('error', 'Unknown error')}")

        # Sort by A/B ratio
        detected_notes.sort(key=lambda x: x['ratio_mean'], reverse=True)
        all_results.sort(key=lambda x: x['ratio_mean'], reverse=True)

        # Return complete results
        chord_result = {
            'audio_file': os.path.basename(audio_file_path),
            'processing_time': processing_time,
            'total_detectors': len(self.detector_names),
            'successful_detectors': len(all_results),
            'failed_detectors': len(results) - len(all_results),
            'detected_notes': detected_notes,
            'all_results': all_results,
            'detection_summary': self._generate_summary(detected_notes, all_results)
        }

        return chord_result

    def _generate_summary(self, detected_notes, all_results):
        """Generate a detection summary."""
        summary = {
            'detected_count': len(detected_notes),
            'detected_note_names': [note['note_name'] for note in detected_notes],
            'highest_ratio': detected_notes[0]['ratio_mean'] if detected_notes else 0,
            'lowest_ratio': detected_notes[-1]['ratio_mean'] if detected_notes else 0,
            'average_ratio_detected': np.mean([n['ratio_mean'] for n in detected_notes]) if detected_notes else 0,
            'average_ratio_undetected': np.mean([n['ratio_mean'] for n in all_results if not n['detected']]) if any(not n['detected'] for n in all_results) else 0
        }
        return summary

    def print_results(self, chord_result):
        """Print detection results."""
        print("\n" + "=" * 60)
        print(f"Detection results: {chord_result['audio_file']}")
        print("=" * 60)
        print(f"Processing time: {chord_result['processing_time']:.2f} s")
        print(f"Notes detected: {chord_result['detection_summary']['detected_count']}")

        if chord_result['detected_notes']:
            print(f"\nDetected notes (sorted by A/B ratio):")
            print("-" * 60)
            print(f"{'Note':<8} {'A/B Ratio':<12} {'Threshold':<12} {'DMD Freq (Hz)':<12}")
            print("-" * 60)

            for note in chord_result['detected_notes']:
                print(f"{note['note_name']:<8} {note['ratio_mean']:<12.6f} {note['threshold']:<12.6f} {note['dmd_frequency']:<12.1f}")

            print("-" * 60)
            print(f"Detected notes: {', '.join(chord_result['detection_summary']['detected_note_names'])}")
        else:
            print("\nNo notes detected")

        print("=" * 60)

    def save_results(self, chord_result, output_file=None):
        """Save detection results to a file."""
        if output_file is None:
            # Auto-generate output filename
            base_name = os.path.splitext(chord_result['audio_file'])[0]
            output_file = f"flho_pitch_detection_{base_name}.json"

        # Prepare serializable data
        serializable_result = self._make_serializable(chord_result)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

        print(f"\nDetection results saved to: {output_file}")
        return output_file

    def _make_serializable(self, obj):
        """Convert numpy data types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


def main():
    """Main function - command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python flho_pitch_detector_utils.py <audio_file> [max_workers] [output_file]")
        print("Example: python flho_pitch_detector_utils.py audio/C4D4G4A4.wav 32 result.json")
        sys.exit(1)

    audio_file = sys.argv[1]
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    # Create multi-note detector
    detector = PitchDetector(max_workers=max_workers)

    # Detect notes
    result = detector.detect_notes(audio_file)

    # Print results
    detector.print_results(result)

    # Save results
    if output_file:
        detector.save_results(result, output_file)
    else:
        detector.save_results(result)


if __name__ == "__main__":
    main()
