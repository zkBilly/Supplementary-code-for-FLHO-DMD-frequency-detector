# Supplementary Code for FLHO-DMD Pitch Detector

This repository contains the supplementary code accompanying the paper:

> **FLHO-DMD: Forced Linear Harmonic Oscillator with Dynamic Mode Decomposition for Multi-Pitch Detection**

## Overview

The FLHO-DMD framework uses an array of 88 physically motivated forced linear harmonic oscillators (FLHO) as frequency-selective probes, combined with stacked Hankel Dynamic Mode Decomposition (DMD) to extract an A/B amplitude ratio that serves as a physically interpretable pitch presence indicator. Three independent physical mechanisms—inharmonicity, temporal envelope decay, and phase instability—collectively explain why overtone-driven resonance yields a systematically weaker response than fundamental-driven resonance.

## File Structure

```
Supplementary-code-for-FLHO-DMD-pitch-detector/
├── dmd_utils.py                         # DMD decomposition core (Hankel + SVD)
├── flho_single_detector_utils.py        # Single FLHO detector implementation
├── flho_single_detector_config.json     # Per-detector configuration (88 detectors, A0–C8)
├── flho_single_detector_runner.py       # Interactive single-detector runner
├── flho_pitch_detector_utils.py         # Multi-note detector (88 parallel detectors)
├── flho_pitch_detector_runner.py        # Interactive multi-pitch runner
├── flho_testset_evaluator.py            # Test set batch evaluation (precision/recall/F1)
├── synthesize_chords_testset.py         # Chord test set synthesizer (triads, seventh, ninth chords)
├── synthesize_chords_trainset.py        # Chord training set synthesizer (intervals, major/minor triads)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── piano_single_note/                   # Single-note audio files (88 piano keys)
    └── single_notes/
        ├── A0.wav ... C8.wav
```

## Key Components

### 1. Core Detection Pipeline

| File | Description |
|------|-------------|
| `dmd_utils.py` | SVD-based Dynamic Mode Decomposition with Hankel embedding. Provides `create_hankel()`, `dmd_decomposition()`, and frequency extraction utilities. |
| `flho_single_detector_utils.py` | Implements a single FLHO detector: audio loading, normalization (waveform_max / spectrum_max), spectral centroid weighting, bandpass filtering, RK4 oscillator integration, and stacked Hankel-DMD analysis via `analyze_flho_dmd_modes()`. |
| `flho_pitch_detector_utils.py` | Orchestrates 88 parallel FLHO detectors using `ProcessPoolExecutor`. Compares each detector's A/B ratio against its `ab_threshold` to decide pitch presence. |

### 2. Runners

| File | Description |
|------|-------------|
| `flho_single_detector_runner.py` | Run a single FLHO detector interactively or via CLI. Supports listing all detectors, selecting specific audio files, and visualizing results. |
| `flho_pitch_detector_runner.py` | Run the full 88-detector array on audio files. Supports interactive mode and command-line batch processing with configurable parallelism. |

### 3. Evaluation & Synthesis

| File | Description |
|------|-------------|
| `flho_testset_evaluator.py` | Batch evaluation on chord test sets. Computes precision, recall, and F1 scores per file and across the dataset. Supports concurrent file processing via thread pools. |
| `synthesize_chords_testset.py` | Synthesizes chord test sets from single-note audio: triads (4 types), seventh chords (7 types), and ninth chords (10 types) with all inversions. |
| `synthesize_chords_trainset.py` | Synthesizes training sets: all intervals within one octave, all major triads, and all minor triads across the 88-key piano range. |

### 4. Configuration

| File | Description |
|------|-------------|
| `flho_single_detector_config.json` | Per-detector configuration for all 88 detectors (A0–C8). Each entry specifies: `f_natural`, `drive_strength`, `centroid_beta`, `bandwidth_octaves`, `ab_threshold`, `normalization_type`, DMD parameters (`hankel_m`, `svd_rank`, `freq_window`), and preprocessing flags. |

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Supplementary-code-for-FLHO-DMD-pitch-detector

# Install dependencies
pip install -r requirements.txt
```

### Required Data

The FLHO detectors require single-note piano audio files as reference signals. The 88 piano note recordings (A0 through C8) are already provided in the `piano_single_note/single_notes/` directory, with filenames like `A0.wav`, `C4.wav`, etc.

## Usage

### Running a Single Detector

```bash
# Interactive mode
python flho_single_detector_runner.py

# List available detectors
python flho_single_detector_runner.py list

# Run a specific detector on an audio file
python flho_single_detector_runner.py run C4 path/to/audio.wav
```

### Running Multi-Note Detection

```bash
# Interactive mode
python flho_pitch_detector_runner.py

# Command-line mode
python flho_pitch_detector_runner.py run path/to/audio.wav

# Batch processing with custom parallelism
python flho_pitch_detector_runner.py run file1.wav file2.wav -w 16 -o results
```

### Evaluating on Test Sets

```bash
# Evaluate both datasets
python flho_testset_evaluator.py

# Evaluate a specific dataset
python flho_testset_evaluator.py --dataset testset

# With custom parallelism
python flho_testset_evaluator.py --workers 16 --parallel-files 4
```

### Synthesizing Datasets

```bash
# Synthesize chord test set
python synthesize_chords_testset.py

# Synthesize chord training set
python synthesize_chords_trainset.py
```

## Detection Pipeline

The FLHO-DMD detection pipeline consists of the following steps:

1. **Audio Loading & Normalization**: Load WAV file, optionally convert stereo to mono, and normalize using either waveform maximum or spectrum maximum normalization.

2. **Spectral Centroid Weighting** (optional): Compute the spectral centroid from peak magnitudes and apply a Gaussian reward coefficient based on distance to the target centroid.

3. **Bandpass Filtering**: Apply a geometrically symmetric bandpass filter centered at the detector's natural frequency with the configured bandwidth (in octaves).

4. **FLHO Integration**: Solve the forced linear harmonic oscillator equation via RK4 integration using the processed audio as the drive signal. Also solve the free (unforced) oscillator for reference.

5. **Stacked Hankel-DMD**: Construct Hankel matrices from both forced and free responses, stack them vertically, perform SVD-truncated DMD, and identify the mode closest to the target frequency.

6. **A/B Ratio Decision**: Compute the amplitude ratio of the forced (A) to free (B) mode components. A pitch is detected when this ratio exceeds the detector's `ab_threshold`.

## Citation

If you use this code in your research, please cite the corresponding paper.

## License

This code is provided for research and educational purposes. Please refer to the paper for licensing details.
