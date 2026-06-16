"""
FLHO Resonance Detector - General Utility Class

A forced linear harmonic oscillator + DMD resonance detection system based on JSON configuration.
Supports multiple normalization methods and preprocessing pipelines.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import json
import re
from dmd_utils import dmd_decomposition, create_hankel


class FLHODetectorConfig:
    """FLHO detector configuration class."""

    def __init__(self, config_path="flho_single_detector_config.json"):
        """Load configuration from a JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.detectors = self.config['detectors']

    def get_detector_names(self):
        """Get all available detector names."""
        return list(self.detectors.keys())

    def get_detector_config(self, detector_name):
        """Get the configuration for a specific detector."""
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}. Available: {list(self.detectors.keys())}")
        return self.detectors[detector_name]


class FLHODetector:
    """FLHO resonance detector utility class."""

    def __init__(self, detector_config, audio_dir=None):
        """
        Initialize the detector.

        Args:
            detector_config: Detector configuration dictionary
            audio_dir: Audio file directory (optional; uses default if not provided)
        """
        self.config = detector_config
        self.target_note = detector_config['target_note']
        self.f_natural = detector_config['f_natural']
        self.omega_0 = 2 * np.pi * self.f_natural
        self.audio_dir = audio_dir if audio_dir else detector_config.get('audio_dir', '.')
        self.output_dir = detector_config['output_dir']
        self.default_files = detector_config['default_files']

        # Processing parameters
        proc_params = detector_config['processing']
        self.normalization_type = proc_params['normalization_type']        # Normalization: "waveform_max" or "spectrum_max"
        self.rectification = proc_params.get('rectification', False)       # Whether to apply full-wave rectification
        self.centroid_weighted = proc_params.get('centroid_weighted', False)  # Whether to apply spectral centroid weighting
        self.centroid_beta = proc_params.get('centroid_beta', 20.0)        # Gaussian decay factor beta for centroid reward
        self.bandwidth_octaves = proc_params['bandwidth_octaves']          # Bandpass filter bandwidth (octaves)
        self.F_drive = proc_params['drive_strength']                       # Drive strength F
        self.interpolation = proc_params.get('interpolation', False)       # Whether to 8x interpolate the drive signal (reduce RK4 dt for high-frequency stability)

        # DMD parameters
        dmd_params = detector_config['dmd_params']
        self.hankel_m = dmd_params['hankel_m']
        self.svd_rank = dmd_params['svd_rank']
        self.freq_window = dmd_params['freq_window']

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_audio_file(self, filename):
        """
        Load an audio file and normalize according to configuration.

        Args:
            filename: Path to the audio file

        Returns:
            audio_data: Normalized audio data
            t_audio: Time array
            sample_rate: Sample rate
        """
        sample_rate, audio_data = wavfile.read(filename)
        audio_data = audio_data.astype(np.float64)

        # Convert stereo to mono
        if len(audio_data.shape) == 2:
            audio_data = audio_data.mean(axis=1)

        # Save original waveform before normalization
        audio_original = audio_data.copy()

        # Choose normalization method based on configuration
        if self.normalization_type == "waveform_max":
            audio_data = self._normalize_waveform_max(audio_data)
        elif self.normalization_type == "spectrum_max":
            audio_data = self._normalize_spectrum_max(audio_data, sample_rate)
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

        # Optional: spectral centroid weighting (independent step)
        if self.centroid_weighted:
            audio_data = self._apply_centroid_weighting(audio_data, sample_rate)

        duration = len(audio_data) / sample_rate
        t_audio = np.linspace(0, duration, len(audio_data))

        return audio_data, audio_original, t_audio, sample_rate

    def _normalize_waveform_max(self, audio_data):
        """Normalize by the maximum absolute waveform amplitude."""
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude
        return audio_data

    def _normalize_spectrum_max(self, audio_data, sample_rate):
        """Normalize by the maximum spectral magnitude."""
        n_fft = len(audio_data)
        fft_result = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_result)

        max_mag = np.max(magnitude)
        if max_mag > 0:
            target_max = n_fft / 30
            scale_factor = target_max / max_mag
            fft_result = fft_result * scale_factor

        audio_data = np.fft.irfft(fft_result, n=n_fft).astype(np.float64)
        return audio_data

    def _apply_centroid_weighting(self, audio_data, sample_rate):
        """Apply spectral centroid weighting (independent step).

        Core idea:
        1. Find peaks above threshold on the spectrum
        2. Compute spectral centroid based on these peaks
        3. Compute reward coefficient based on distance to target centroid
        """
        # Compute spectrum
        n_fft = len(audio_data)
        fft_result = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_result)
        freqs = np.fft.rfftfreq(n_fft, 1/sample_rate)

        # Normalize to 1000 (consistent with plot_single_notes_spectrum.py)
        max_mag = np.max(magnitude)
        if max_mag > 0:
            magnitude_norm = magnitude / max_mag * 1000.0
        else:
            magnitude_norm = magnitude

        # Find peaks: threshold at 5% of max (50 after normalization)
        threshold = 50.0
        freq_resolution = sample_rate / len(audio_data)
        min_distance = max(3, int(25 / freq_resolution))  # 25 Hz spacing
        peak_indices, _ = find_peaks(magnitude_norm, height=threshold, distance=min_distance)

        if len(peak_indices) > 0:
            # Compute spectral centroid based on peaks
            peak_freqs = freqs[peak_indices]
            peak_magnitudes = magnitude_norm[peak_indices]
            signal_centroid = np.sum(peak_freqs * peak_magnitudes) / np.sum(peak_magnitudes)
        else:
            # Fall back to full-spectrum centroid if no peaks found
            if np.sum(magnitude) > 0:
                signal_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                signal_centroid = 0.0

        # Compute reward coefficient
        target_centroid = self.config.get('spectral_centroid', None)

        if target_centroid is not None and target_centroid > 0:
            relative_distance = abs(signal_centroid - target_centroid) / target_centroid
            beta = self.centroid_beta
            reward = np.exp(-beta * relative_distance**2)
        else:
            reward = 1.0

        # Apply reward directly in the time domain (scalar reward, equivalent to frequency-domain multiplication)
        return audio_data * reward

    def apply_preprocessing(self, audio_signal, sample_rate):
        """
        Apply preprocessing steps (rectification, filtering).

        Args:
            audio_signal: Input signal
            sample_rate: Sample rate

        Returns:
            drive_signal: Processed drive signal
        """
        signal = audio_signal.copy()

        # Apply full-wave rectification if configured
        if self.rectification:
            signal = np.abs(signal)
            # Remove DC component after rectification (full-wave rectification shifts mean far above 0)
            signal = signal - np.mean(signal)

        # Bandpass filtering is always applied
        signal = self._apply_bandpass_filter(signal, sample_rate)

        return signal

    def _apply_bandpass_filter(self, signal, sample_rate):
        """Apply bandpass filter (geometrically symmetric bandwidth)."""
        half_octaves = self.bandwidth_octaves / 2.0
        low_cutoff = self.f_natural / (2 ** half_octaves)
        high_cutoff = self.f_natural * (2 ** half_octaves)

        nyquist = sample_rate / 2.0
        low_cutoff = max(low_cutoff, 1.0)
        high_cutoff = min(high_cutoff, nyquist * 0.95)

        low_normalized = low_cutoff / nyquist
        high_normalized = high_cutoff / nyquist

        if low_normalized <= 0 or low_normalized >= 1.0 or high_normalized <= 0 or high_normalized >= 1.0:
            return signal

        order = 2
        b, a = butter(order, [low_normalized, high_normalized], btype='band')
        padlen = min(3 * max(len(b), len(a)), len(signal) - 1)
        if padlen < 1:
            padlen = 1

        try:
            filtered = filtfilt(b, a, signal, padlen=padlen)
            if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                filtered = filtfilt(b, a, signal, padlen=0)
            return filtered
        except:
            return signal

    def solve_linear_oscillator_forced(self, s, dt, n_steps, x0=0.0, v0=0.0):
        """Solve the forced linear harmonic oscillator equation."""
        x = np.zeros(n_steps)
        v = np.zeros(n_steps)
        x[0] = x0
        v[0] = v0

        for step in range(1, n_steps):
            def derivatives(x_val, v_val, t_idx):
                dxdt = v_val
                dvdt = -self.omega_0**2 * x_val + self.F_drive * s[t_idx]
                return dxdt, dvdt

            dx1, dv1 = derivatives(x[step-1], v[step-1], step-1)
            dx2, dv2 = derivatives(x[step-1] + 0.5*dt*dx1, v[step-1] + 0.5*dt*dv1, step-1)
            dx3, dv3 = derivatives(x[step-1] + 0.5*dt*dx2, v[step-1] + 0.5*dt*dv2, step-1)
            dx4, dv4 = derivatives(x[step-1] + dt*dx3, v[step-1] + dt*dv3, step-1)

            x[step] = x[step-1] + (dt/6.0)*(dx1 + 2*dx2 + 2*dx3 + dx4)
            v[step] = v[step-1] + (dt/6.0)*(dv1 + 2*dv2 + 2*dv3 + dv4)

        return x, v

    def solve_linear_oscillator_free(self, dt, n_steps, x0=1.0, v0=0.0):
        """Solve the free (unforced) linear harmonic oscillator equation."""
        x = np.zeros(n_steps)
        v = np.zeros(n_steps)
        x[0] = x0
        v[0] = v0

        for step in range(1, n_steps):
            def derivatives(x_val, v_val):
                dxdt = v_val
                dvdt = -self.omega_0**2 * x_val
                return dxdt, dvdt

            dx1, dv1 = derivatives(x[step-1], v[step-1])
            dx2, dv2 = derivatives(x[step-1] + 0.5*dt*dx1, v[step-1] + 0.5*dt*dv1)
            dx3, dv3 = derivatives(x[step-1] + 0.5*dt*dx2, v[step-1] + 0.5*dt*dv2)
            dx4, dv4 = derivatives(x[step-1] + dt*dx3, v[step-1] + dt*dv3)

            x[step] = x[step-1] + (dt/6.0)*(dx1 + 2*dx2 + 2*dx3 + dx4)
            v[step] = v[step-1] + (dt/6.0)*(dv1 + 2*dv2 + 2*dv3 + dv4)

        return x, v

    def _solve_with_interpolation(self, drive_signal, dt, n_steps, interp_factor=8):
        """Solve oscillator response with interpolated (smaller) RK4 time step dt, then downsample to original resolution."""
        dt_interp = dt / interp_factor
        n_steps_interp = n_steps * interp_factor
        t_original = np.arange(n_steps) * dt
        t_interp = np.arange(n_steps_interp) * dt_interp
        interp_func = interpolate.interp1d(t_original, drive_signal, kind='cubic',
                                           fill_value='extrapolate')
        drive_signal_interp = interp_func(t_interp)
        print(f"  Interpolation: {interp_factor}x, dt: {dt:.2e} -> {dt_interp:.2e}, "
              f"omega_0*dt: {self.omega_0*dt:.4f} -> {self.omega_0*dt_interp:.4f}")
        x_forced_interp, _ = self.solve_linear_oscillator_forced(
            drive_signal_interp, dt_interp, n_steps_interp, x0=0.0, v0=0.0)
        x_free_interp, _ = self.solve_linear_oscillator_free(
            dt_interp, n_steps_interp, x0=1.0, v0=0.0)
        x_forced = x_forced_interp[::interp_factor]
        x_free = x_free_interp[::interp_factor]
        return x_forced, x_free

    def analyze_flho_dmd_modes(self, x_forced, x_free, dt):
        """
        Perform joint DMD analysis on forced and free oscillator responses.

        Returns:
            amplitude_ratio, dmd_frequency, all_frequencies, eigenvalue_norms,
            best_idx, mode_forced_part, mode_free_part, all_modes
        """
        H_A = create_hankel(x_forced, self.hankel_m)
        H_B = create_hankel(x_free, self.hankel_m)
        H_combined = np.vstack([H_A, H_B])

        eigenvalues, modes = dmd_decomposition(H_combined, svd_rank=self.svd_rank)

        frequencies = np.angle(eigenvalues) / (2 * np.pi * dt)
        eigenvalue_norms = np.abs(eigenvalues)

        freq_diffs = np.abs(frequencies - self.f_natural)
        best_idx = np.argmin(freq_diffs)
        closest_freq = frequencies[best_idx]

        if freq_diffs[best_idx] > self.freq_window:
            print(f"  Warning: No mode found within {self.f_natural:.1f} +/- {self.freq_window} Hz")

        selected_mode = modes[:, best_idx]
        mode_forced_part = selected_mode[:self.hankel_m]
        mode_free_part = selected_mode[self.hankel_m:]

        free_magnitudes = np.abs(mode_free_part)
        forced_magnitudes = np.abs(mode_forced_part)

        valid_mask = free_magnitudes > 1e-10
        if np.sum(valid_mask) > 0:
            ratios = forced_magnitudes[valid_mask] / free_magnitudes[valid_mask]
            amplitude_ratio = np.mean(ratios)
        else:
            amplitude_ratio = 0.0

        return (amplitude_ratio, closest_freq, frequencies, eigenvalue_norms,
                best_idx, mode_forced_part, mode_free_part, modes)

    def contains_target_note(self, filename):
        """Check whether the file contains the target note."""
        name_without_ext = os.path.splitext(os.path.basename(filename))[0]
        note_pattern = r'([A-Ga-g])(#|b)?(\d+)'
        matches = re.findall(note_pattern, name_without_ext)

        target_letter = self.target_note[0]
        target_accidental = '#' if '#' in self.target_note else ('b' if 'b' in self.target_note else '')
        target_octave = int(''.join(filter(str.isdigit, self.target_note)))

        for match in matches:
            note_letter = match[0].upper()
            accidental = match[1] or ''
            octave = int(match[2])

            if (note_letter == target_letter.upper() and
                accidental == target_accidental and
                octave == target_octave):
                return True, self.f_natural

        return False, 0.0

    def process_audio_file(self, audio_file_path):
        """
        Process a single audio file.

        Returns:
            result_dict or None
        """
        has_target, f_expected = self.contains_target_note(audio_file_path)
        print(f"\nFile: {os.path.basename(audio_file_path)}")
        print(f"Contains {self.target_note}: {has_target}")

        try:
            audio_signal, audio_original, t_audio, sample_rate = self.load_audio_file(audio_file_path)
            print(f"Sample rate: {sample_rate} Hz, Duration: {t_audio[-1]:.3f} s")
            print(f"Original signal range: [{np.min(audio_original):.6e}, {np.max(audio_original):.6e}]")
            print(f"Original signal energy: {np.mean(audio_original**2):.6e}")
        except Exception as e:
            print(f"Error: {e}")
            return None

        # Preprocessing
        # Generate preprocessing description
        if self.rectification:
            preprocessing_desc = "rectification -> bandpass_filter"
        else:
            preprocessing_desc = "bandpass_filter"
        print(f"Preprocessing: {preprocessing_desc}")
        drive_signal = self.apply_preprocessing(audio_signal, sample_rate)

        print(f"Drive signal range: [{np.min(drive_signal):.6e}, {np.max(drive_signal):.6e}]")
        print(f"Drive signal energy: {np.mean(drive_signal**2):.6e}")

        # Solve linear harmonic oscillator equation
        print(f"Solving linear harmonic oscillator: omega_0={self.omega_0:.1f} rad/s, F={self.F_drive}")
        dt = 1.0 / sample_rate
        n_steps = len(drive_signal)

        if self.interpolation:
            x_forced, x_free = self._solve_with_interpolation(drive_signal, dt, n_steps)
        else:
            x_forced, _ = self.solve_linear_oscillator_forced(drive_signal, dt, n_steps, x0=0.0, v0=0.0)
            x_free, _ = self.solve_linear_oscillator_free(dt, n_steps, x0=1.0, v0=0.0)

        # Check numerical stability
        if np.any(np.isnan(x_forced)) or np.any(np.isinf(x_forced)):
            print(f"  Error: Forced response contains NaN or Inf")
            return None

        if np.any(np.isnan(x_free)) or np.any(np.isinf(x_free)):
            print(f"  Error: Free response contains NaN or Inf")
            return None

        # DMD analysis
        print(f"DMD analysis: Hankel rank={self.hankel_m}")

        amplitude_ratio, dmd_freq, all_freqs, all_eigenvalue_norms, best_idx, mode_A, mode_B, all_modes = \
            self.analyze_flho_dmd_modes(x_forced, x_free, dt)

        # Print all modes
        print(f"All DMD modes ({len(all_freqs)} total):")
        print(f"  {'Mode':<6} {'Freq (Hz)':<12} {'|lambda|':<8} {'|A|':<10} {'|B|':<10}")
        print("-" * 53)
        for i in range(len(all_freqs)):
            mode_num = i + 1
            marker = " <-- selected" if i == best_idx else ""
            mode_i = all_modes[:, i]
            mode_A_i = mode_i[:self.hankel_m]
            mode_B_i = mode_i[self.hankel_m:]
            avg_norm_A = np.mean(np.abs(mode_A_i))
            avg_norm_B = np.mean(np.abs(mode_B_i))
            print(f"  Mode{mode_num:<5} {all_freqs[i]:<11.1f} {all_eigenvalue_norms[i]:<8.4f} {avg_norm_A:<11.6f} {avg_norm_B:<10.6f}{marker}")

        print(f"DMD result: Frequency={dmd_freq:.1f} Hz, |lambda|={all_eigenvalue_norms[best_idx]:.4f}")
        print(f"Amplitude ratio (A/B): {amplitude_ratio:.6f}")

        return {
            'audio_name': os.path.basename(audio_file_path),
            f'has_{self.target_note.replace("#", "s")}': has_target,
            'ratio_mean': amplitude_ratio,
            'dmd_frequency': dmd_freq,
            'resonance_indicator': amplitude_ratio,
            'response_forced': x_forced,
            'response_free': x_free,
            'drive_signal': drive_signal,
            'audio_signal': audio_signal,
            'audio_original': audio_original,
            't': t_audio,
            'all_frequencies': all_freqs,
            'eigenvalue_norms': all_eigenvalue_norms,
            'best_idx': best_idx,
            'mode_A': mode_A,
            'mode_B': mode_B
        }

    def visualize_results(self, all_results):
        """Visualize detection results."""
        if not all_results:
            return

        target_key = f'has_{self.target_note.replace("#", "s")}'

        # Separate results with and without the target note
        target_results = [r for r in all_results if r[target_key]]
        non_target_results = [r for r in all_results if not r[target_key]]

        # Detailed waveform plot - prefer default_files, fall back to processed files
        default_names = [os.path.basename(f) for f in self.default_files]
        result_names = [r['audio_name'] for r in all_results]

        # First pick default files from processed results (in default order)
        viz_results = []
        for df in default_names:
            for r in all_results:
                if r['audio_name'] == df:
                    viz_results.append(r)
                    break

        # Then add other non-default files
        for r in all_results:
            if r['audio_name'] not in default_names:
                viz_results.append(r)

        # Limit to at most 3
        viz_results = viz_results[:3]

        n_viz = len(viz_results)
        if n_viz > 0:
            fig = plt.figure(figsize=(20, 5*n_viz))

            for idx in range(n_viz):
                res = viz_results[idx]
                t_viz = res['t']
                audio_original_viz = res.get('audio_original', None)
                drive_viz = res['drive_signal']
                x_forced_viz = res['response_forced']
                x_free_viz = res['response_free']
                amp_ratio = res['ratio_mean']

                # Use full duration
                time_points = len(t_viz)

                ax1 = plt.subplot(n_viz, 3, idx*3 + 1)
                if audio_original_viz is not None:
                    ax1.plot(t_viz[:time_points], audio_original_viz[:time_points], 'b-', linewidth=0.5)
                    ax1.set_ylabel('Amplitude', fontsize=10)
                    orig_energy = np.mean(audio_original_viz[:time_points]**2)
                    title_text = f'Audio: {res["audio_name"]} (E={orig_energy:.4e})'
                else:
                    ax1.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax1.transAxes)
                    title_text = f'Audio: {res["audio_name"]}'
                ax1.set_xlabel('Time (s)', fontsize=10)
                ax1.set_title(title_text, fontsize=11)
                ax1.grid(True, alpha=0.3)

                ax2 = plt.subplot(n_viz, 3, idx*3 + 2)
                ax2.plot(t_viz[:time_points], drive_viz[:time_points], 'c-', linewidth=0.5)
                drive_energy = np.mean(drive_viz[:time_points]**2)
                ax2.set_xlabel('Time (s)', fontsize=10)
                ax2.set_ylabel('Amplitude', fontsize=10)
                ax2.set_title(f'Drive Signal (E={drive_energy:.4e})', fontsize=11)
                ax2.grid(True, alpha=0.3)

                ax3 = plt.subplot(n_viz, 3, idx*3 + 3)
                x_forced_max = np.max(np.abs(x_forced_viz[:time_points]))
                ax3.plot(t_viz[:time_points], x_free_viz[:time_points], 'gray', linewidth=0.8, alpha=0.3, label='Free')
                ax3.plot(t_viz[:time_points], x_forced_viz[:time_points], 'r-', linewidth=1.0, alpha=0.7, label='Forced')
                ax3.axhline(y=x_forced_max, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
                ax3.axhline(y=-x_forced_max, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
                ax3.set_xlabel('Time (s)', fontsize=10)
                ax3.set_ylabel('Displacement', fontsize=10)
                ax3.set_title(f'A/B={amp_ratio:.4f}, max={x_forced_max:.4f}', fontsize=11)
                ax3.legend(loc='upper right', fontsize=8)
                ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'flho_signal_pipeline.pdf')
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"\nDetailed visualization saved: {save_path}")
            plt.close()

        # Amplitude comparison chart - using all processed files
        if len(all_results) >= 2:
            fig_compare = plt.figure(figsize=(max(14, len(all_results)*0.8), 7))

            labels = []
            amp_ratios = []
            colors = []

            for res in all_results:
                labels.append(f"{res['audio_name']}\n({self.target_note}: {'Y' if res[target_key] else 'N'})")
                amp_ratios.append(res['ratio_mean'])
                colors.append('red' if res[target_key] else 'gray')

            x_pos = np.arange(len(labels))
            bars = plt.bar(x_pos, amp_ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

            for i, (bar, ratio) in enumerate(zip(bars, amp_ratios)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(amp_ratios)*0.01,
                        f'{ratio:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=45)

            plt.xlabel('Audio File', fontsize=12)
            plt.ylabel('DMD Mode Amplitude Ratio (forced/free)', fontsize=12)
            plt.title(f'DMD Mode Amplitude Ratio Comparison ({self.target_note})', fontsize=14, fontweight='bold')
            plt.xticks(x_pos, labels, fontsize=8, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y', linestyle='--')

            if target_results and non_target_results:
                min_target = np.min([r['ratio_mean'] for r in target_results])
                max_non_target = np.max([r['ratio_mean'] for r in non_target_results])

                if min_target >= max_non_target:
                    threshold = (min_target + max_non_target) / 2
                    plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
                    plt.legend(loc='upper right', fontsize=10)
                else:
                    plt.text(0.5, 0.95, f'No valid threshold (min_{self.target_note} < max_non{self.target_note})',
                            transform=plt.gca().transAxes,
                            ha='center', va='top', fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

            plt.tight_layout()
            save_path_compare = os.path.join(self.output_dir, 'flho_amplitude_comparison.pdf')
            plt.savefig(save_path_compare, format='pdf', bbox_inches='tight')
            print(f"Amplitude comparison chart saved: {save_path_compare}")
            plt.close()
