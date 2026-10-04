"""
AudioSentry DSP Engine
Lightweight, CPU-efficient Digital Signal Processing (DSP) pipeline for real-time
voice sentinel and synthetic speech / deepfake detection at 16kHz.

Key Features Analyzed:
1. Spectral Roll-off (> 5.8 kHz threshold & High Frequency ratio)
2. Phase Jitter / Phase Coherence (STFT phase derivative dynamics)
3. Pitch Micro-perturbations (Glottal cycle-to-cycle F0 jitter)
4. 64-Bin Mel-Spectrogram Energy Bank
5. Exponential Moving Average (EMA) ring buffer filter with alpha = 0.08
"""

import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq


class AudioSentryDSP:
    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_seconds: float = 1.0,
        ema_alpha: float = 0.08,
        rolloff_threshold_hz: float = 5800.0,
        n_mels: int = 64,
    ):
        """
        Initializes the DSP Sentinel Engine.
        :param sample_rate: Expected audio sampling frequency (default: 16000 Hz)
        :param buffer_seconds: Duration of circular ring buffer in seconds (default: 1.0s = 16,000 samples)
        :param ema_alpha: Exponential Moving Average smoothing factor (alpha = 0.08)
        :param rolloff_threshold_hz: Spectral roll-off critical frequency (> 5.8 kHz)
        :param n_mels: Number of Mel-scale frequency bins (default: 64)
        """
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.ema_alpha = ema_alpha
        self.rolloff_threshold_hz = rolloff_threshold_hz
        self.n_mels = n_mels

        # Ring buffer for active audio stream (Float32)
        self.ring_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.total_samples_received = 0

        # State tracking for EMA
        self.synthetic_risk_ema = 0.15
        self.last_verdict = "STANDBY"
        self.confidence_score = 85.0

        # Precompute window functions and frequencies for speed
        self.fft_size = 1024
        self.fft_window = get_window("hann", self.fft_size, fftbins=True).astype(np.float32)
        self.freq_bins = rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        # Precompute 64-bin Mel Filter Bank
        self.mel_filters = self._build_mel_filterbank(self.n_mels, self.fft_size, self.sample_rate)

    def _hz_to_mel(self, hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self, n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
        """Constructs an orthonormal 64-bin triangular Mel-scale filterbank matrix."""
        low_freq = 20.0
        high_freq = sample_rate / 2.0  # 8000 Hz
        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        n_bins = n_fft // 2 + 1
        filters = np.zeros((n_mels, n_bins), dtype=np.float32)

        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            if f_m > f_m_minus:
                filters[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            if f_m_plus > f_m:
                filters[m - 1, f_m:f_m_plus] = (f_m_plus - np.arange(f_m, f_m_plus)) / (f_m_plus - f_m)

        return filters

    def push_audio(self, chunk: np.ndarray) -> None:
        """Push incoming audio samples (50ms = 800 samples) into circular ring buffer."""
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        # Normalize if chunk is 16-bit integer range
        if np.max(np.abs(chunk)) > 1.0:
            chunk = np.clip(chunk / 32768.0, -1.0, 1.0)

        chunk_len = len(chunk)
        if chunk_len >= self.buffer_size:
            self.ring_buffer[:] = chunk[-self.buffer_size:]
        else:
            self.ring_buffer = np.roll(self.ring_buffer, -chunk_len)
            self.ring_buffer[-chunk_len:] = chunk

        self.total_samples_received += chunk_len

    def compute_spectral_features(self, audio: np.ndarray, percentile: float = 0.85) -> tuple[float, float, list[float]]:
        """
        Calculates spectral roll-off, high-frequency energy ratio (>5.8kHz),
        and 64-bin Mel-Spectrogram energy distribution.
        """
        if len(audio) < self.fft_size:
            return 0.0, 0.0, [0.0] * self.n_mels

        segment = audio[-self.fft_size:] * self.fft_window
        spec_mag = np.abs(rfft(segment))
        power = spec_mag ** 2
        total_power = np.sum(power)

        if total_power < 1e-9:
            return 0.0, 0.0, [0.0] * self.n_mels

        # 1. Spectral Roll-off
        cumulative_power = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumulative_power, percentile * total_power)
        rolloff_idx = min(rolloff_idx, len(self.freq_bins) - 1)
        rolloff_hz = float(self.freq_bins[rolloff_idx])

        # 2. Energy ratio above 5.8 kHz
        hf_mask = self.freq_bins >= self.rolloff_threshold_hz
        hf_power = np.sum(power[hf_mask])
        hf_ratio = float(hf_power / total_power)

        # 3. 64-Bin Mel Spectrogram
        mel_power = np.dot(self.mel_filters, power)
        log_mel = np.log10(np.maximum(mel_power, 1e-6))
        # Min-max normalization for visualization: -6.0 to 1.0
        normalized_mel = np.clip((log_mel + 6.0) / 7.0, 0.0, 1.0)
        mel_bins = [round(float(v), 3) for v in normalized_mel]

        return rolloff_hz, hf_ratio, mel_bins

    def compute_phase_jitter(self, audio: np.ndarray, frame_size: int = 512, hop_size: int = 256) -> float:
        """
        Measures STFT phase derivative jitter across consecutive frames.
        Natural human speech contains organic glottal turbulence and acoustic dispersion.
        Synthetic speech exhibits either rigid harmonic phase-locking (<0.12) or stochastic noise.
        """
        if len(audio) < frame_size * 2:
            return 0.0

        recent_audio = audio[-min(len(audio), 4096):]
        num_frames = (len(recent_audio) - frame_size) // hop_size
        if num_frames < 2:
            return 0.0

        window = get_window("hann", frame_size, fftbins=True).astype(np.float32)
        phases = []

        for i in range(num_frames):
            start = i * hop_size
            frame = recent_audio[start : start + frame_size] * window
            spec = rfft(frame)
            phases.append(np.angle(spec))

        phases = np.array(phases)
        phase_diff = np.diff(phases, axis=0)
        phase_diff_unwrapped = (phase_diff + np.pi) % (2 * np.pi) - np.pi

        bin_freqs = rfftfreq(frame_size, d=1.0 / self.sample_rate)
        voiced_mask = (bin_freqs >= 300) & (bin_freqs <= 4000)

        if not np.any(voiced_mask):
            return 0.0

        active_diffs = phase_diff_unwrapped[:, voiced_mask]
        phase_jitter_val = float(np.mean(np.std(active_diffs, axis=0)))
        return phase_jitter_val

    def compute_pitch_micro_perturbation(self, audio: np.ndarray) -> tuple[float, float]:
        """
        Computes F0 fundamental pitch and cycle-to-cycle relative perturbation (Jitter %).
        Natural human vocal folds display micro-tremor (~0.5% - 2.5%).
        TTS models show unnatural micro-stability (<0.25%) or abrupt vocoder glitches (>4.5%).
        """
        min_lag = int(self.sample_rate / 450)  # ~35 samples
        max_lag = int(self.sample_rate / 70)   # ~228 samples

        frame_len = 480  # 30ms
        hop = 160        # 10ms
        recent = audio[-min(len(audio), 4800):]

        num_frames = (len(recent) - frame_len) // hop
        if num_frames < 3:
            return 0.0, 0.0

        pitch_periods = []
        for i in range(num_frames):
            start = i * hop
            segment = recent[start : start + frame_len]
            seg_norm = segment - np.mean(segment)
            variance = np.sum(seg_norm ** 2)
            if variance < 1e-5:
                continue

            corr = np.correlate(seg_norm, seg_norm, mode="full")
            corr = corr[len(seg_norm) - 1 :]

            if len(corr) <= max_lag:
                continue

            search_window = corr[min_lag:max_lag]
            peak_rel_idx = np.argmax(search_window)
            peak_lag = min_lag + peak_rel_idx
            peak_val = search_window[peak_rel_idx]

            if corr[0] > 0 and (peak_val / corr[0]) > 0.35:
                pitch_periods.append(peak_lag)

        if len(pitch_periods) < 3:
            return 0.0, 0.0

        pitch_periods = np.array(pitch_periods, dtype=np.float32)
        avg_lag = np.mean(pitch_periods)
        estimated_f0 = float(self.sample_rate / avg_lag) if avg_lag > 0 else 0.0

        period_diffs = np.abs(np.diff(pitch_periods))
        jitter_percent = float((np.mean(period_diffs) / avg_lag) * 100.0)

        return estimated_f0, jitter_percent

    def analyze(self) -> dict:
        """
        Executes full DSP Sentinel verification on current ring buffer.
        Applies Exponential Moving Average (alpha=0.3) for smooth anti-flicker transitions.
        """
        if self.total_samples_received < 800:
            return {
                "is_speech": False,
                "verdict": "STANDBY",
                "confidence": 100.0,
                "synthetic_risk": 0.0,
                "ema_alpha": round(float(self.ema_alpha), 2),
                "spectral_rolloff_hz": 0.0,
                "hf_energy_ratio": 0.0,
                "phase_jitter": 0.0,
                "f0_hz": 0.0,
                "pitch_jitter_percent": 0.0,
                "rms_level": 0.0,
                "mel_bins": [0.0] * self.n_mels,
                "waveform_preview": [0.0] * 64,
            }

        audio = self.ring_buffer
        rms = float(np.sqrt(np.mean(audio[-1600:] ** 2)))
        is_speech = rms > 0.008

        # Downsample waveform for UI oscilloscope preview
        preview_samples = audio[-800:]
        downsampled = preview_samples[:: max(1, len(preview_samples) // 64)][:64]
        waveform_preview = [round(float(x), 4) for x in downsampled]

        # Spectral analysis & 64-bin Mel Spectrogram
        rolloff_hz, hf_ratio, mel_bins = self.compute_spectral_features(audio, percentile=0.85)

        if not is_speech:
            self.synthetic_risk_ema = (1.0 - self.ema_alpha) * self.synthetic_risk_ema + self.ema_alpha * 0.15
            return {
                "is_speech": False,
                "verdict": "STANDBY",
                "confidence": round(float((1.0 - self.synthetic_risk_ema) * 100.0), 1),
                "synthetic_risk": round(float(self.synthetic_risk_ema), 3),
                "ema_alpha": round(float(self.ema_alpha), 2),
                "spectral_rolloff_hz": 0.0,
                "hf_energy_ratio": 0.0,
                "phase_jitter": 0.0,
                "f0_hz": 0.0,
                "pitch_jitter_percent": 0.0,
                "rms_level": round(rms, 4),
                "mel_bins": mel_bins,
                "waveform_preview": waveform_preview,
            }

        # Phase jitter & Pitch micro-perturbations
        phase_jitter = self.compute_phase_jitter(audio)
        f0_hz, pitch_jitter_pct = self.compute_pitch_micro_perturbation(audio)

        # Deepfake anomaly penalties
        spectral_anomaly = 0.0
        if rolloff_hz < self.rolloff_threshold_hz:
            deficit = max(0.0, (self.rolloff_threshold_hz - rolloff_hz) / self.rolloff_threshold_hz)
            spectral_anomaly += 0.5 * deficit

        if hf_ratio < 0.025:
            spectral_anomaly += 0.3 * (1.0 - (hf_ratio / 0.025))
        elif hf_ratio > 0.35:
            spectral_anomaly += 0.35
        spectral_anomaly = min(1.0, spectral_anomaly)

        phase_anomaly = 0.0
        if phase_jitter < 0.12:
            phase_anomaly = (0.12 - phase_jitter) / 0.12
        elif phase_jitter > 1.20:
            phase_anomaly = min(1.0, (phase_jitter - 1.20) / 0.8)

        pitch_anomaly = 0.0
        if f0_hz > 65.0:
            if pitch_jitter_pct < 0.25:
                pitch_anomaly = (0.25 - pitch_jitter_pct) / 0.25
            elif pitch_jitter_pct > 4.5:
                pitch_anomaly = min(1.0, (pitch_jitter_pct - 4.5) / 3.0)

        # Multi-factor raw risk composite
        raw_synthetic_risk = float(np.clip(
            0.40 * spectral_anomaly + 0.30 * phase_anomaly + 0.30 * pitch_anomaly,
            0.0,
            1.0
        ))

        # Exponential Moving Average (EMA) Ring Buffer Filter: alpha = 0.08
        self.synthetic_risk_ema = float(
            self.ema_alpha * raw_synthetic_risk + (1.0 - self.ema_alpha) * self.synthetic_risk_ema
        )

        # Dynamic Verdict Determination
        if self.synthetic_risk_ema >= 0.50:
            verdict = "ALERT"
            confidence = 50.0 + (self.synthetic_risk_ema - 0.50) * 100.0
        else:
            verdict = "AUTHENTIC"
            confidence = 50.0 + (0.50 - self.synthetic_risk_ema) * 100.0

        confidence = float(np.clip(confidence, 50.0, 99.8))
        self.last_verdict = verdict
        self.confidence_score = confidence

        return {
            "is_speech": True,
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "synthetic_risk": round(float(self.synthetic_risk_ema), 3),
            "ema_alpha": round(float(self.ema_alpha), 2),
            "spectral_rolloff_hz": round(rolloff_hz, 1),
            "hf_energy_ratio": round(hf_ratio, 4),
            "phase_jitter": round(phase_jitter, 3),
            "f0_hz": round(f0_hz, 1),
            "pitch_jitter_percent": round(pitch_jitter_pct, 2),
            "rms_level": round(rms, 4),
            "mel_bins": mel_bins,
            "waveform_preview": waveform_preview,
        }
