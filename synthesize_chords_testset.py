import os
import numpy as np
import scipy.io.wavfile as wav

# ============================================================
# Configuration parameters
# ============================================================
SINGLE_NOTES_DIR = r".\piano_single_note"
OUTPUT_DIR = r".\synthesized_chords_testset"

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MIN_MIDI = 12 * (0 + 1) + 9   # A0  = MIDI 21, lowest allowed note
MAX_MIDI = 12 * (8 + 1) + 0   # C8  = MIDI 108, highest allowed note

# Standard 88-key piano root note list (A0 -> C8, ascending by pitch)
PIANO_ROOT_NOTES = (
    ['A0', 'A#0', 'B0']
    + [f"{n}{o}" for o in range(1, 8) for n in NOTE_NAMES]
    + ['C8']
)

# ============================================================
# Chord definitions: {abbreviation: root-position semitone interval list (relative to root)}
# ============================================================

# Triads (major, minor, augmented, diminished)
TRIAD_TYPES = {
    'major':      [0, 4, 7],    # Major triad: root + major 3rd + perfect 5th
    'minor':      [0, 3, 7],    # Minor triad: root + minor 3rd + perfect 5th
    'augmented':  [0, 4, 8],    # Augmented triad: root + major 3rd + augmented 5th
    'diminished': [0, 3, 6],    # Diminished triad: root + minor 3rd + diminished 5th
}

# Triad inversion requirements
#   maj / min: first and second inversions only
#   aug / dim: root position + first and second inversions
TRIAD_INVERSIONS = {
    'major':      [1, 2],       # Generate only first and second inversions
    'minor':      [1, 2],
    'augmented':  [0, 1, 2],    # Root position, first, second inversions
    'diminished': [0, 1, 2],
}

# Seventh chords (7 types x 4 inversions)
SEVENTH_TYPES = {
    'maj7':     [0, 4, 7, 11],     # major 7th
    'min7':     [0, 3, 7, 10],     # minor 7th
    'dom7':     [0, 4, 7, 10],     # dominant 7th
    'hdim7':    [0, 3, 6, 10],     # half diminished 7th
    'dim7':     [0, 3, 6, 9],      # diminished 7th
    'minmaj7':  [0, 3, 7, 11],     # minor major 7th
    'augmaj7':  [0, 4, 8, 11],     # augmented major 7th
}

# Ninth chords (10 types x 5 inversions)
NINTH_TYPES = {
    'dom7maj9':    [0, 4, 7, 10, 14],   # dominant 7th & major 9th
    'dom7min9':    [0, 4, 7, 10, 13],   # dominant 7th & minor 9th
    'min7maj9':    [0, 3, 7, 10, 14],   # minor 7th & major 9th
    'min7min9':    [0, 3, 7, 10, 13],   # minor 7th & minor 9th
    'hdim7min9':   [0, 3, 6, 10, 13],   # half diminished 7th & minor 9th
    'maj7maj9':    [0, 4, 7, 11, 14],   # major 7th & major 9th
    'maj7aug9':    [0, 4, 7, 11, 15],   # major 7th & augmented 9th
    'dim7min9':    [0, 3, 6, 9, 13],    # diminished 7th & minor 9th
    'minmaj7maj9': [0, 3, 7, 11, 14],   # minor major 7th & major 9th
    'aug7maj9':    [0, 4, 8, 11, 14],   # augmented 7th & major 9th
}

# ============================================================
# Utility functions
# ============================================================

def get_note_index(note_name):
    """
    Convert a note name to a MIDI number (e.g., C4 -> 60, C#4 -> 61).
    """
    if '#' in note_name:
        base_note = note_name[:2]
        octave = int(note_name[2:])
    else:
        base_note = note_name[:1]
        octave = int(note_name[1:])

    semitones = NOTE_NAMES.index(base_note)
    return 12 * (octave + 1) + semitones


def midi_to_note_name(midi_num):
    """
    Convert a MIDI number to a note name (e.g., 60 -> C4).
    """
    octave = midi_num // 12 - 1
    semitone = midi_num % 12
    return f"{NOTE_NAMES[semitone]}{octave}"


def load_audio(note_name):
    """
    Load a single-note audio file.
    """
    file_name = f"{note_name}.wav"
    file_path = os.path.join(SINGLE_NOTES_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    sample_rate, audio_data = wav.read(file_path)
    return sample_rate, audio_data


def synthesize_audio(audio_list, sample_rate):
    """
    Synthesize multiple audio signals, using normalized mixing to avoid clipping distortion.
    """
    max_length = max(len(audio) for audio in audio_list)
    mixed_audio = np.zeros(max_length, dtype=np.float32)

    for audio in audio_list:
        if len(audio) < max_length:
            padded_audio = np.zeros(max_length, dtype=np.float32)
            padded_audio[:len(audio)] = audio
        else:
            padded_audio = audio[:max_length]
        mixed_audio += padded_audio

    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio * (0.9 / max_val)

    return mixed_audio


def save_audio(sample_rate, audio_data, filename):
    """
    Save an audio file.
    """
    output_path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav.write(output_path, sample_rate, audio_data)
    print(f"Saved: {output_path}")
    return output_path


# ============================================================
# Chord note generation
# ============================================================

def intervals_to_notes(root_note, intervals):
    """
    Given a root note and a list of semitone intervals,
    compute the note names of all constituent notes (root position).

    Args:
        root_note: Root note name, e.g. 'C4'
        intervals: List of semitone intervals, e.g. [0, 4, 7]

    Returns:
        List of note names, e.g. ['C4', 'E4', 'G4']
    """
    root_midi = get_note_index(root_note)
    notes = []
    for offset in intervals:
        midi = root_midi + offset
        notes.append(midi_to_note_name(midi))
    return notes


def invert_notes(notes, inversion):
    """
    Invert a root-position note list.

    Args:
        notes: Root-position note name list
        inversion: Number of inversions
                   0 = root position
                   1 = first inversion (2nd note becomes bass)
                   2 = second inversion (3rd note becomes bass)
                   ...

    Returns:
        Inverted note name list
    """
    if inversion == 0:
        return list(notes)

    n = len(notes)
    bass_idx = inversion

    result = []
    # Starting from the bass, take subsequent notes from the original chord
    # (keeping octave unchanged)
    for i in range(bass_idx, n):
        result.append(notes[i])
    # Lower voice notes that were "flipped up" are raised by one octave
    for i in range(0, bass_idx):
        midi = get_note_index(notes[i])
        result.append(midi_to_note_name(midi + 12))

    return result


def check_notes_in_range(notes):
    """
    Check whether all notes are within the 88-key piano range (A0 ~ C8, MIDI 21 ~ 108).
    """
    for note in notes:
        midi = get_note_index(note)
        if midi < MIN_MIDI or midi > MAX_MIDI:
            return False
    return True


def compute_chord_notes(root_note, intervals, inversion):
    """
    Compute all constituent notes of a chord (after inversion).

    Args:
        root_note: Root note name
        intervals: Root-position semitone interval list
        inversion: Number of inversions

    Returns:
        List of note names; returns None if out of range
    """
    root_notes = intervals_to_notes(root_note, intervals)
    inverted = invert_notes(root_notes, inversion)

    if not check_notes_in_range(inverted):
        return None

    return inverted


# ============================================================
# Single chord synthesis
# ============================================================

def synthesize_chord(notes):
    """
    Synthesize a chord audio file and save it.
    The filename is formed by concatenating constituent notes from low to high pitch
    (e.g., E4G4C5.wav).

    Args:
        notes: List of constituent note names

    Returns:
        On success returns (sample_rate, audio_data, filename), None on failure
    """
    # Sort by pitch low-to-high and concatenate as filename
    sorted_notes = sorted(notes, key=get_note_index)
    filename = "".join(sorted_notes) + ".wav"

    # Load individual note audio
    audio_list = []
    sample_rate = None
    for note in sorted_notes:
        sr, audio = load_audio(note)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {note}({sr})")
        audio_list.append(audio)

    synthesized = synthesize_audio(audio_list, sample_rate)
    save_audio(sample_rate, synthesized, filename)
    return sample_rate, synthesized, filename


# ============================================================
# Batch generation functions
# ============================================================

def generate_triads():
    """
    Batch generate triad chord test set.

    Major/minor triads: first and second inversions only
    Augmented/diminished triads: root position + first and second inversions
    """
    print("\n" + "=" * 60)
    print("Generating triad test set (major/minor inversions + aug/dim root & inversions)")
    print("=" * 60)

    for root_note in PIANO_ROOT_NOTES:
        for triad_type, intervals in TRIAD_TYPES.items():
            for inv in TRIAD_INVERSIONS[triad_type]:
                notes = compute_chord_notes(root_note, intervals, inv)
                if notes is None:
                    continue
                synthesize_chord(notes)


def generate_seventh_chords():
    """
    Batch generate seventh chord test set (7 types x 4 inversions).
    """
    print("\n" + "=" * 60)
    print("Generating seventh chord test set (7 types x root + 3 inversions)")
    print("=" * 60)

    for root_note in PIANO_ROOT_NOTES:
        for seventh_type, intervals in SEVENTH_TYPES.items():
            for inv in range(4):    # 0=root, 1=first, 2=second, 3=third inversion
                notes = compute_chord_notes(root_note, intervals, inv)
                if notes is None:
                    continue
                synthesize_chord(notes)


def generate_ninth_chords():
    """
    Batch generate ninth chord test set (10 types x 5 inversions).
    """
    print("\n" + "=" * 60)
    print("Generating ninth chord test set (10 types x root + 4 inversions)")
    print("=" * 60)

    for root_note in PIANO_ROOT_NOTES:
        for ninth_type, intervals in NINTH_TYPES.items():
            for inv in range(5):    # 0=root, 1~4=four inversions
                notes = compute_chord_notes(root_note, intervals, inv)
                if notes is None:
                    continue
                synthesize_chord(notes)


# ============================================================
# Main function
# ============================================================

def main():
    print("=" * 60)
    print("Chord Test Set Audio Synthesizer")
    print("=" * 60)
    print(f"Single-note source dir: {SINGLE_NOTES_DIR}")
    print(f"Output dir:             {os.path.abspath(OUTPUT_DIR)}")
    print()
    print("Test set contents:")
    print("  1. Major/minor triads: first and second inversions")
    print("  2. Augmented/diminished triads: root + first and second inversions")
    print("  3. Seventh chords: 7 types x 4 inversions (root/first/second/third)")
    print("  4. Ninth chords: 10 types x 5 inversions (root/first/second/third/fourth)")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Test files ----
    print("--- Synthesizing test files ---")

    # Triad test: C4 major first inversion
    notes = compute_chord_notes("C4", TRIAD_TYPES['major'], 1)
    if notes:
        synthesize_chord(notes)

    # Triad test: C4 augmented root position
    notes = compute_chord_notes("C4", TRIAD_TYPES['augmented'], 0)
    if notes:
        synthesize_chord(notes)

    # Seventh chord test: C4 maj7 root position
    notes = compute_chord_notes("C4", SEVENTH_TYPES['maj7'], 0)
    if notes:
        synthesize_chord(notes)

    # Ninth chord test: C4 dom7maj9 root position
    notes = compute_chord_notes("C4", NINTH_TYPES['dom7maj9'], 0)
    if notes:
        synthesize_chord(notes)

    print(f"\nTest files synthesized and saved to {OUTPUT_DIR} directory")
    print("Please verify the synthesis results. If correct, continue with batch synthesis...")

    response = input("\nContinue with batch synthesis of all chord test sets? (y/n): ")
    if response.lower() != 'y':
        print("Program ended.")
        return

    # ---- Batch synthesis ----
    generate_triads()
    generate_seventh_chords()
    generate_ninth_chords()

    print("\n" + "=" * 60)
    print("All chord test set synthesis tasks complete!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
