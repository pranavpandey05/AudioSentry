# AudioSentry 🎙️🛡️
> **Real-Time Voice Sentinel & Deepfake Detection Dashboard**  
> Pure NumPy & SciPy Digital Signal Processing (DSP) • Zero Heavy GPU Dependencies • 16 kHz Real-Time WebSocket Telemetry

---

## 📌 Overview

**AudioSentry** is a real-time voice sentinel and synthetic deepfake detection system designed for high-efficiency audio verification. It inspects incoming microphone audio streams at **16 kHz** in **50 ms frames (800 samples)** over a bi-directional WebSocket connection.

Unlike resource-heavy deep learning transformer models that require GPUs, AudioSentry uses **lightweight, mathematically rigorous DSP algorithms** implemented in pure NumPy and SciPy:
1. **Spectral Roll-Off (> 5.8 kHz Analysis):** Detects brick-wall frequency suppression, unnatural band cutoffs, and deficient high-frequency energy typical of neural vocoders (e.g., HiFi-GAN, WaveGlow).
2. **Phase Jitter Dynamics:** Evaluates short-time Fourier transform (STFT) phase velocity and dispersion across harmonic bands to identify synthetic phase locks or random phase incoherence.
3. **Pitch Micro-Perturbations (Glottal Jitter):** Measures cycle-to-cycle fundamental frequency ($F_0$) perturbation. Biological human vocal cords exhibit natural micro-tremor ($0.5\% - 2.5\%$ jitter), whereas synthetic voices often have mathematical perfection ($<0.25\%$) or unphysical vocoder glitches.
4. **Exponential Moving Average (EMA) Ring Buffer ($\alpha = 0.08$):** Mitigates frame-by-frame volatility and flicker, ensuring smooth, natural confidence transitions and stable verdicts without jumpy oscillations.

---

## 🗂️ Project Structure

```text
AudioSentry Project/
├── backend/
│   ├── requirements.txt      # FastAPI, WebSockets, NumPy, SciPy, Uvicorn
│   ├── main.py               # FastAPI server + 16kHz WebSocket audio ingestion endpoint
│   └── dsp_engine.py         # Spectral roll-off, phase jitter, pitch micro-perturbation, EMA filter
├── frontend/
│   ├── index.html            # Dark security-themed dashboard (Tailwind CSS, SVG gauge, HUD)
│   └── app.js                # Web Audio API 16kHz streaming, 50ms chunking, dual canvas visualizers
└── README.md                 # Project documentation and local execution instructions
```

---


## 🔬 DSP Methodology & Mathematics

### A. Spectral Roll-Off ($> 5.8\text{ kHz}$)
The roll-off frequency $f_r$ is defined as the frequency below which $85\%$ of the total spectral power is concentrated:
$$\sum_{f=0}^{f_r} |X(f)|^2 = 0.85 \times \sum_{f=0}^{f_s/2} |X(f)|^2$$
Most TTS systems and vocoders exhibit high-frequency deficits ($f_r < 5.8\text{ kHz}$) or abnormal energy ratios:
$$R_{\text{hf}} = \frac{\sum_{f \ge 5.8\text{kHz}} |X(f)|^2}{\sum |X(f)|^2}$$

### B. Phase Jitter
Phase angles $\theta(t, \omega)$ from consecutive STFT frames are unwrapped to calculate phase velocity:
$$\Delta \theta(t, \omega) = \text{unwrap}(\theta(t, \omega) - \theta(t-1, \omega))$$
The phase jitter index measures the standard deviation of $\Delta \theta$ across voiced harmonic bands ($300\text{ Hz} - 4,000\text{ Hz}$).

### C. Pitch Micro-Perturbation (Glottal Jitter %)
Using normalized autocorrelation over a 30 ms sliding window, the fundamental period $T_i$ is estimated for consecutive voiced frames:
$$\text{Jitter} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N}\sum_{i=1}^{N} T_i} \times 100\%$$
Human vocal cords produce involuntary cycle-to-cycle tremors ($0.5\% \le \text{Jitter} \le 2.5\%$).

### D. Exponential Moving Average (EMA) Filter
$$\bar{S}_t = \alpha \cdot S_t + (1 - \alpha) \cdot \bar{S}_{t-1} \quad (\alpha = 0.08)$$
This suppresses single-frame transient spikes and stabilizes the confidence gauge.

---

## 📄 License
MIT License. Built for real-time security telemetry and acoustic research.
