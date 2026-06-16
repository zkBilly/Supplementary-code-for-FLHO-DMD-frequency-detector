"""
DMD (Dynamic Mode Decomposition) Utility Functions - Stable Version

Based on an improved SVD-DMD algorithm suitable for complex-domain signal analysis.

Key improvements:
1. SVD performed directly on the Hankel matrix H (rather than H1)
2. Koopman operator constructed using the time-shift property of the V matrix
3. Pseudoinverse used to improve numerical stability

Usage example:
    from dmd_utils import dmd_decomposition, create_hankel

    # Build Hankel matrix
    H = create_hankel(signal, m=10)

    # DMD decomposition
    eigenvalues, modes = dmd_decomposition(H, svd_rank=5)

    # Compute frequencies
    frequencies = np.angle(eigenvalues) / (2 * np.pi * dt)
"""

import numpy as np
from typing import Tuple, Optional


def create_hankel(signal: np.ndarray, m: int) -> np.ndarray:
    """
    Build a Hankel matrix.

    Args:
        signal: Input signal (can be real or complex)
        m: Number of rows in the Hankel matrix

    Returns:
        Hankel matrix of shape (m, n-m+1), where n is the signal length
    """
    n = len(signal)
    if m > n:
        raise ValueError(f"Hankel row count m={m} cannot exceed signal length n={n}")

    H = np.zeros((m, n - m + 1), dtype=signal.dtype)
    for i in range(m):
        H[i, :] = signal[i:i + n - m + 1]

    return H


def dmd_decomposition(H: np.ndarray,
                      svd_rank: Optional[int] = None,
                      normalize_modes: bool = True,
                      singular_value_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform DMD decomposition (stable version).

    Algorithm steps:
    1. SVD on Hankel matrix H: H = U @ diag(S) @ V
    2. Truncate to rank r, obtaining Ur, Sr, Vr
    3. Construct Koopman operator: K = Y @ pinv(X)
       where X = diag(Sr) @ Vr[:, :-1]
             Y = diag(Sr) @ Vr[:, 1:]
    4. Eigendecomposition: K @ W = W @ Lambda
    5. Reconstruct DMD modes: Phi = Ur @ W

    Args:
        H: Hankel matrix of shape (features, snapshots)
        svd_rank: SVD truncation rank. If None, full rank is used or
                  rank is auto-selected based on singular value threshold
        normalize_modes: Whether to normalize mode vectors (L2 norm = 1 per column)
        singular_value_threshold: Singular value threshold. Retains singular values
                                  greater than max(S)*threshold.
                                  If provided, overrides the svd_rank parameter

    Returns:
        eigenvalues: DMD eigenvalues, shape (r,)
        modes: DMD modes, shape (features, r)
    """
    features, snapshots = H.shape

    # Step 0: Numerical stability preprocessing
    # Check for NaN or Inf in the matrix
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("Hankel matrix contains NaN or Inf values")

    # Normalize to improve numerical stability
    H_max = np.max(np.abs(H))
    if H_max > 0:
        H_normalized = H / H_max
    else:
        H_normalized = H.copy()

    # Step 1: SVD on H (complex domain)
    # Python's svd returns V as V* (conjugate transpose)
    try:
        U, S, V = np.linalg.svd(H_normalized, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print(f"Warning: SVD did not converge, adding regularization...")
        # Add a small regularization term to improve numerical stability
        epsilon = 1e-8
        H_reg = H_normalized + epsilon * np.random.randn(*H_normalized.shape).astype(H_normalized.dtype)
        U, S, V = np.linalg.svd(H_reg, full_matrices=False)

    # Step 2: Determine truncation rank
    if singular_value_threshold is not None:
        # Auto-select rank based on singular value threshold
        threshold = np.max(S) * singular_value_threshold
        svd_rank = np.sum(S > threshold)
        svd_rank = max(svd_rank, 1)  # Keep at least 1 mode
    else:
        # Use specified rank or full rank
        if svd_rank is None:
            svd_rank = min(features, snapshots)

    svd_rank = min(svd_rank, len(S))

    Ur = U[:, :svd_rank]
    Sr = S[:svd_rank]
    Vr = V[:svd_rank, :]

    # Step 3: Compute Koopman operator matrix in the basis of U
    # X = diag(S) @ Vr[:, :-1]  (r, snapshots-1)
    # Y = diag(S) @ Vr[:, 1:]   (r, snapshots-1)
    # K = Y @ pinv(X)          (r, r)
    X = np.diag(Sr) @ Vr[:, :-1]
    Y = np.diag(Sr) @ Vr[:, 1:]
    K = Y @ np.linalg.pinv(X)

    # Step 4: Eigendecomposition
    eigenvalues_tilde, eigenvectors_tilde = np.linalg.eig(K)

    # Step 5: Reconstruct DMD modes
    modes = Ur @ eigenvectors_tilde

    # Normalize modes (optional)
    if normalize_modes:
        for i in range(modes.shape[1]):
            norm = np.linalg.norm(modes[:, i])
            if norm > 1e-10:
                modes[:, i] = modes[:, i] / norm

    return eigenvalues_tilde, modes


def extract_dmd_frequencies(eigenvalues: np.ndarray, dt: float) -> np.ndarray:
    """
    Extract frequencies from DMD eigenvalues.

    Args:
        eigenvalues: DMD eigenvalues (discrete-time)
        dt: Sampling time interval (seconds)

    Returns:
        frequencies: Frequencies (Hz)
    """
    return np.angle(eigenvalues) / (2 * np.pi * dt)


def find_dominant_modes(eigenvalues: np.ndarray,
                        modes: np.ndarray,
                        target_freq: Optional[float] = None,
                        n_modes: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find dominant modes (by energy or proximity to target frequency).

    Args:
        eigenvalues: DMD eigenvalues
        modes: DMD modes
        target_freq: Target frequency (Hz). If provided, selects modes closest to it;
                     otherwise sorts by mode energy
        n_modes: Number of modes to return

    Returns:
        selected_eigenvalues: Selected eigenvalues
        selected_modes: Selected modes
        frequencies: Corresponding frequencies (dt parameter required)
    """
    if target_freq is not None:
        # Select by frequency proximity
        dt = 1.0  # Assume dt=1; actual frequency needs scaling by 1/dt
        freqs = extract_dmd_frequencies(eigenvalues, dt)
        freq_diffs = np.abs(freqs - target_freq)
        indices = np.argsort(freq_diffs)[:n_modes]
    else:
        # Select by mode energy (L2 norm)
        mode_energies = np.array([np.linalg.norm(modes[:, i])
                                  for i in range(modes.shape[1])])
        indices = np.argsort(mode_energies)[::-1][:n_modes]

    return (eigenvalues[indices],
            modes[:, indices],
            extract_dmd_frequencies(eigenvalues[indices], 1.0))


# ========== Test code ==========
if __name__ == "__main__":
    print("=" * 60)
    print("DMD Utility Functions Test")
    print("=" * 60)

    # Generate test signal: 440 Hz complex exponential signal
    f_test = 440.0  # Hz
    dt = 1e-3       # s
    T = 0.5         # s
    n = int(T / dt)
    t = np.arange(n) * dt

    z = np.exp(1j * 2 * np.pi * f_test * t)

    print(f"\nTest signal:")
    print(f"  Frequency: {f_test} Hz")
    print(f"  Length: {n} points")
    print(f"  Sample rate: {1/dt} Hz")

    # Build Hankel matrix
    hankel_m = 10
    H = create_hankel(z, hankel_m)
    print(f"\nHankel matrix:")
    print(f"  Rows m: {hankel_m}")
    print(f"  Shape: {H.shape}")

    # DMD decomposition
    print("\nDMD decomposition:")
    eigenvalues, modes = dmd_decomposition(H, svd_rank=None)

    print(f"  Number of eigenvalues: {len(eigenvalues)}")
    print(f"  Eigenvalue magnitude range: [{np.min(np.abs(eigenvalues)):.4f}, {np.max(np.abs(eigenvalues)):.4f}]")

    # Compute frequencies
    frequencies = extract_dmd_frequencies(eigenvalues, dt)

    print(f"\nDMD frequency results:")
    print(f"  Frequency range: [{np.min(frequencies):.1f}, {np.max(frequencies):.1f}] Hz")

    # Find mode closest to 440 Hz
    best_idx = np.argmin(np.abs(frequencies - f_test))
    print(f"\nClosest mode (index {best_idx}):")
    print(f"  Frequency: {frequencies[best_idx]:.2f} Hz")
    print(f"  Error: {abs(frequencies[best_idx] - f_test):.2f} Hz")
    print(f"  Mode shape: {modes[:, best_idx].shape}")
    print(f"  Mode magnitude: {np.abs(modes[:, best_idx])}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
