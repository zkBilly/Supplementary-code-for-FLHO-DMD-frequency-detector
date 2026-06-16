import os
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

# Configuration parameters
SINGLE_NOTES_DIR = r".\piano_single_note"
OUTPUT_DIR = r".\single_note_and_synthesized_chords_trainset"

# Note name mapping
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_note_index(note_name):
    """
    Convert a note name to a numeric index (e.g., C4 -> 48, C#4 -> 49).
    """
    # Separate note name and octave
    if '#' in note_name:
        base_note = note_name[:2]
        octave = int(note_name[2:])
    else:
        base_note = note_name[:1]
        octave = int(note_name[1:])

    # Compute number of semitones
    semitones = NOTE_NAMES.index(base_note)
    # MIDI note number = 12 * (octave + 1) + semitones
    return 12 * (octave + 1) + semitones

def load_audio(note_name):
    """
    Load a single-note audio file.
    """
    # Handle the # symbol in note names
    file_name = f"{note_name}.wav"
    file_path = os.path.join(SINGLE_NOTES_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    sample_rate, audio_data = wav.read(file_path)
    return sample_rate, audio_data

def synthesize_audio(audio_list, sample_rate):
    """
    Synthesize multiple audio signals using appropriate mixing techniques
    to avoid clipping distortion.

    Args:
        audio_list: List of audio signals
        sample_rate: Sample rate

    Returns:
        Synthesized audio signal
    """
    # Find the longest audio length
    max_length = max(len(audio) for audio in audio_list)

    # Create mixing buffer
    mixed_audio = np.zeros(max_length, dtype=np.float32)

    # Mix each audio signal into the buffer
    for audio in audio_list:
        # Zero-pad if audio length is insufficient
        if len(audio) < max_length:
            padded_audio = np.zeros(max_length, dtype=np.float32)
            padded_audio[:len(audio)] = audio
        else:
            padded_audio = audio[:max_length]

        mixed_audio += padded_audio

    # Normalize to avoid clipping distortion
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        # Use 0.9 as a safety factor to preserve some dynamic range
        mixed_audio = mixed_audio * (0.9 / max_val)

    return mixed_audio

def create_interval(note1, note2):
    """
    Synthesize a two-note interval.
    """
    print(f"Synthesizing interval: {note1} - {note2}")

    # Load audio
    sr1, audio1 = load_audio(note1)
    sr2, audio2 = load_audio(note2)

    # Ensure consistent sample rate
    if sr1 != sr2:
        raise ValueError(f"Sample rate mismatch: {note1}({sr1}) vs {note2}({sr2})")

    # Synthesize audio
    synthesized = synthesize_audio([audio1, audio2], sr1)

    # Generate filename (preserving the original # symbol)
    filename = f"{note1}{note2}.wav"
    return sr1, synthesized, filename

def create_triad(root_note, third_note, fifth_note, chord_type="major"):
    """
    Synthesize a triad chord.
    """
    print(f"Synthesizing {chord_type} triad: {root_note} - {third_note} - {fifth_note}")

    # Load audio
    sr1, audio1 = load_audio(root_note)
    sr2, audio2 = load_audio(third_note)
    sr3, audio3 = load_audio(fifth_note)

    # Ensure consistent sample rate
    if not (sr1 == sr2 == sr3):
        raise ValueError(f"Sample rate mismatch: {sr1}, {sr2}, {sr3}")

    # Synthesize audio
    synthesized = synthesize_audio([audio1, audio2, audio3], sr1)

    # Generate filename (preserving the original # symbol)
    filename = f"{root_note}{third_note}{fifth_note}.wav"
    return sr1, synthesized, filename

def save_audio(sample_rate, audio_data, filename):
    """
    Save an audio file.
    """
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save as WAV file
    wav.write(output_path, sample_rate, audio_data)
    print(f"Saved: {output_path}")
    return output_path

def generate_major_triads():
    """
    Generate all major triads.
    """
    triads = []
    for octave in range(0, 9):  # 0 to 8 octaves
        for root_idx in range(12):
            root_note = f"{NOTE_NAMES[root_idx]}{octave}"
            # Major triad: root + major 3rd (4 semitones) + perfect 5th (7 semitones)
            third_idx = (root_idx + 4) % 12
            fifth_idx = (root_idx + 7) % 12

            # Handle octave carry-over
            third_octave = octave
            fifth_octave = octave

            if third_idx < root_idx:
                third_octave += 1
            if fifth_idx < root_idx:
                fifth_octave += 1

            # Check if within range (allow C8)
            if fifth_octave <= 8:
                third_note = f"{NOTE_NAMES[third_idx]}{third_octave}"
                fifth_note = f"{NOTE_NAMES[fifth_idx]}{fifth_octave}"
                triads.append((root_note, third_note, fifth_note, "major"))

    return triads

def generate_minor_triads():
    """
    Generate all minor triads.
    """
    triads = []
    for octave in range(0, 9):  # 0 to 8 octaves
        for root_idx in range(12):
            root_note = f"{NOTE_NAMES[root_idx]}{octave}"
            # Minor triad: root + minor 3rd (3 semitones) + perfect 5th (7 semitones)
            third_idx = (root_idx + 3) % 12
            fifth_idx = (root_idx + 7) % 12

            # Handle octave carry-over
            third_octave = octave
            fifth_octave = octave

            if third_idx < root_idx:
                third_octave += 1
            if fifth_idx < root_idx:
                fifth_octave += 1

            # Check if within range (allow C8)
            if fifth_octave <= 8:
                third_note = f"{NOTE_NAMES[third_idx]}{third_octave}"
                fifth_note = f"{NOTE_NAMES[fifth_idx]}{fifth_octave}"
                triads.append((root_note, third_note, fifth_note, "minor"))

    return triads

def generate_intervals_within_octave():
    """
    Generate all intervals within one octave.
    """
    intervals = []
    for octave in range(0, 9):  # 0 to 8 octaves
        for root_idx in range(12):
            root_note = f"{NOTE_NAMES[root_idx]}{octave}"

            # Generate various intervals starting from the root (no more than one octave)
            for interval_semitones in range(1, 13):  # 1 to 12 semitones
                target_idx = (root_idx + interval_semitones) % 12
                target_octave = octave + (root_idx + interval_semitones) // 12

                if target_octave <= 8:  # Limit to within 8 octaves
                    target_note = f"{NOTE_NAMES[target_idx]}{target_octave}"
                    intervals.append((root_note, target_note))

    return intervals

def main():
    print("="*60)
    print("Audio Interval & Triad Synthesizer")
    print("="*60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # First synthesize specified test files
    print("\n--- Synthesizing test files ---")

    # 1. C4-G4 interval (perfect fifth)
    sr, audio, filename = create_interval("C4", "G4")
    save_audio(sr, audio, filename)

    # 2. C4-E4-G4 major triad
    sr, audio, filename = create_triad("C4", "E4", "G4", "major")
    save_audio(sr, audio, filename)

    print(f"\nTest files synthesized and saved to {OUTPUT_DIR} directory")
    print("Please verify the synthesis results. If correct, continue with batch synthesis...")

    # Ask whether to continue with batch synthesis
    response = input("\nContinue with batch synthesis of all intervals and triads? (y/n): ")
    if response.lower() != 'y':
        print("Program ended.")
        return

    print("\n--- Starting batch synthesis ---")

    # Generate and synthesize all intervals
    print("\nSynthesizing all intervals within one octave...")
    intervals = generate_intervals_within_octave()
    for i, (note1, note2) in enumerate(intervals):
        try:
            sr, audio, filename = create_interval(note1, note2)
            save_audio(sr, audio, filename)
            print(f"[{i+1}/{len(intervals)}] Done: {filename}")
        except Exception as e:
            print(f"[{i+1}/{len(intervals)}] Error: {note1}-{note2}: {str(e)}")

    # Generate and synthesize all major triads
    print("\nSynthesizing all major triads...")
    major_triads = generate_major_triads()
    for i, (root, third, fifth, chord_type) in enumerate(major_triads):
        try:
            sr, audio, filename = create_triad(root, third, fifth, chord_type)
            save_audio(sr, audio, filename)
            print(f"[{i+1}/{len(major_triads)}] Done: {filename}")
        except Exception as e:
            print(f"[{i+1}/{len(major_triads)}] Error: {root}-{third}-{fifth}: {str(e)}")

    # Generate and synthesize all minor triads
    print("\nSynthesizing all minor triads...")
    minor_triads = generate_minor_triads()
    for i, (root, third, fifth, chord_type) in enumerate(minor_triads):
        try:
            sr, audio, filename = create_triad(root, third, fifth, chord_type)
            save_audio(sr, audio, filename)
            print(f"[{i+1}/{len(minor_triads)}] Done: {filename}")
        except Exception as e:
            print(f"[{i+1}/{len(minor_triads)}] Error: {root}-{third}-{fifth}: {str(e)}")

    print("\nAll synthesis tasks complete!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
