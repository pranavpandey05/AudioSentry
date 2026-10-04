/**
 * AudioSentry Frontend Client
 * - Web Audio API 16kHz capture & resampler
 * - 50ms audio chunking (800 Float32 samples per frame)
 * - Raw PCM WebSocket streaming to FastAPI backend
 * - Smooth anti-flicker Confidence Gauge & Dynamic Verdict toggle ("AUTHENTIC" vs "ALERT")
 * - Dual Canvas Visualizers: Live 16kHz Oscilloscope + 64-Bin Mel-Spectrogram Heatmap (>5.8kHz)
 */

(() => {
  // --- DOM Elements ---
  const wsStatusBadge = document.getElementById("wsStatusBadge");
  const wsStatusDot = document.getElementById("wsStatusDot");
  const wsStatusText = document.getElementById("wsStatusText");
  const toggleMicBtn = document.getElementById("toggleMicBtn");
  const micBtnText = document.getElementById("micBtnText");

  // Verdict & Gauge Elements
  const verdictCard = document.getElementById("verdictCard");
  const verdictTitle = document.getElementById("verdictTitle");
  const verdictDescription = document.getElementById("verdictDescription");
  const verdictIconWrap = document.getElementById("verdictIconWrap");
  const verdictIcon = document.getElementById("verdictIcon");
  const gaugeProgressCircle = document.getElementById("gaugeProgressCircle");
  const confidencePercentage = document.getElementById("confidencePercentage");
  const confidenceLabel = document.getElementById("confidenceLabel");

  // Telemetry Metric Elements
  const quickRolloff = document.getElementById("quickRolloff");
  const quickPhase = document.getElementById("quickPhase");
  const quickPitch = document.getElementById("quickPitch");
  const quickHf = document.getElementById("quickHf");
  const metricRolloffVal = document.getElementById("metricRolloffVal");
  const metricPhaseVal = document.getElementById("metricPhaseVal");
  const metricPitchVal = document.getElementById("metricPitchVal");
  const metricRiskVal = document.getElementById("metricRiskVal");
  const barRolloff = document.getElementById("barRolloff");
  const barPhase = document.getElementById("barPhase");
  const barPitch = document.getElementById("barPitch");
  const barRisk = document.getElementById("barRisk");
  const rmsVal = document.getElementById("rmsVal");

  // Canvases
  const oscCanvas = document.getElementById("oscilloscopeCanvas");
  const oscCtx = oscCanvas.getContext("2d");
  const melCanvas = document.getElementById("melHeatmapCanvas");
  const melCtx = melCanvas.getContext("2d");

  // Simulation Buttons
  const simSyntheticBtn = document.getElementById("simSyntheticBtn");
  const simHumanBtn = document.getElementById("simHumanBtn");

  // --- Constants & Config ---
  const TARGET_SAMPLE_RATE = 16000;
  const CHUNK_DURATION_SEC = 0.05; // 50ms
  const SAMPLES_PER_CHUNK = Math.round(TARGET_SAMPLE_RATE * CHUNK_DURATION_SEC); // 800 samples
  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 50; // r=50 -> ~314.159
  const N_MELS = 64;

  // --- State Variables ---
  let socket = null;
  let isRecording = false;
  let audioContext = null;
  let mediaStream = null;
  let scriptProcessor = null;
  let simulationInterval = null;

  // Audio Chunk Aggregator
  let pcmBuffer = [];

  // Live Visualizer Data Stores
  let oscilloscopeWaveform = new Float32Array(512);
  let liveMelBins = new Float32Array(N_MELS);
  let liveRms = 0.0;
  let latestRolloffHz = 0;
  let displayedConfidence = 0.0;

  // --- WebSocket Setup ---
  function getWebSocketUrl() {
    const loc = window.location;
    const protocol = loc.protocol === "https:" ? "wss:" : "ws:";
    const host = loc.port && loc.port !== "8000" && loc.hostname ? `${loc.hostname}:8000` : (loc.host || "localhost:8000");
    return `${protocol}//${host}/ws/audio`;
  }

  function initWebSocket() {
    const wsUrl = getWebSocketUrl();
    console.log(`[AudioSentry] Connecting WebSocket to: ${wsUrl}`);
    setWsStatus("CONNECTING", "amber");

    try {
      socket = new WebSocket(wsUrl);
      socket.binaryType = "arraybuffer";

      socket.onopen = () => {
        console.log("[AudioSentry] WebSocket connected successfully.");
        setWsStatus("CONNECTED", "emerald");
      };

      socket.onclose = () => {
        console.warn("[AudioSentry] WebSocket closed. Retrying in 2.5s...");
        setWsStatus("OFFLINE", "rose");
        setTimeout(initWebSocket, 2500);
      };

      socket.onerror = (err) => {
        console.error("[AudioSentry] WebSocket error:", err);
      };

      socket.onmessage = (event) => {
        try {
          const telemetry = JSON.parse(event.data);
          handleTelemetry(telemetry);
        } catch (e) {
          console.error("Failed to parse incoming telemetry:", e);
        }
      };
    } catch (e) {
      console.error("Failed to construct WebSocket:", e);
      setTimeout(initWebSocket, 3000);
    }
  }

  function setWsStatus(text, color) {
    wsStatusText.innerText = text;
    if (color === "emerald") {
      wsStatusBadge.className = "flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400";
      wsStatusDot.className = "w-2 h-2 rounded-full bg-emerald-400";
    } else if (color === "rose") {
      wsStatusBadge.className = "flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400";
      wsStatusDot.className = "w-2 h-2 rounded-full bg-rose-400";
    } else {
      wsStatusBadge.className = "flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400";
      wsStatusDot.className = "w-2 h-2 rounded-full bg-amber-400 animate-pulse";
    }
  }

  // --- Telemetry Dispatch & UI Updates ---
  function handleTelemetry(t) {
    if (!t) return;

    latestRolloffHz = t.spectral_rolloff_hz || 0;
    liveRms = t.rms_level || 0;
    rmsVal.innerText = liveRms.toFixed(4);

    // Update 64-bin Mel data
    if (t.mel_bins && t.mel_bins.length === N_MELS) {
      for (let i = 0; i < N_MELS; i++) {
        liveMelBins[i] = t.mel_bins[i];
      }
    }

    // Update Telemetry metric cards
    quickRolloff.innerText = `${t.spectral_rolloff_hz} Hz`;
    quickPhase.innerText = `${t.phase_jitter}`;
    quickPitch.innerText = `${t.pitch_jitter_percent} %`;
    quickHf.innerText = `${(t.hf_energy_ratio * 100).toFixed(1)}%`;

    metricRolloffVal.innerText = `${t.spectral_rolloff_hz} Hz`;
    metricPhaseVal.innerText = `${t.phase_jitter}`;
    metricPitchVal.innerText = `${t.pitch_jitter_percent} %`;
    metricRiskVal.innerText = `${t.synthetic_risk}`;

    // Progress bar updates
    barRolloff.style.width = `${Math.min(100, (t.spectral_rolloff_hz / 8000) * 100)}%`;
    barPhase.style.width = `${Math.min(100, (t.phase_jitter / 1.5) * 100)}%`;
    barPitch.style.width = `${Math.min(100, (t.pitch_jitter_percent / 4.0) * 100)}%`;
    barRisk.style.width = `${Math.min(100, t.synthetic_risk * 100)}%`;

    // Smooth confidence gauge animation
    updateConfidenceGauge(t.confidence, t.verdict);

    // Update dynamic verdict indicator
    updateVerdictUI(t.verdict);
  }

  function updateConfidenceGauge(confidence, verdict) {
    const target = Math.max(0, Math.min(100, confidence));
    
    // Smooth anti-flicker glide interpolator
    displayedConfidence += (target - displayedConfidence) * 0.18;
    const rounded = Math.round(displayedConfidence * 10) / 10;
    confidencePercentage.innerText = `${rounded.toFixed(1)}%`;

    // SVG dashoffset calculation
    const offset = GAUGE_CIRCUMFERENCE - (target / 100) * GAUGE_CIRCUMFERENCE;
    gaugeProgressCircle.style.strokeDashoffset = offset;

    // Tint gauge circle and text strictly matching classification
    if (verdict === "ALERT") {
      gaugeProgressCircle.setAttribute("stroke", "#f43f5e"); // Vibrant Crimson
      confidencePercentage.className = "text-4xl font-extrabold font-mono tracking-tight text-rose-400";
      confidenceLabel.innerText = "AI CONFIDENCE";
      confidenceLabel.className = "text-[10px] font-mono uppercase tracking-wider text-rose-400 mt-0.5";
    } else if (verdict === "AUTHENTIC") {
      gaugeProgressCircle.setAttribute("stroke", "#10b981"); // Bright Emerald
      confidencePercentage.className = "text-4xl font-extrabold font-mono tracking-tight text-emerald-400";
      confidenceLabel.innerText = "HUMAN CONFIDENCE";
      confidenceLabel.className = "text-[10px] font-mono uppercase tracking-wider text-emerald-400 mt-0.5";
    } else {
      gaugeProgressCircle.setAttribute("stroke", "#06b6d4"); // Cyan standby
      confidencePercentage.className = "text-4xl font-extrabold font-mono tracking-tight text-white";
      confidenceLabel.innerText = "CONFIDENCE";
      confidenceLabel.className = "text-[10px] font-mono uppercase tracking-wider text-slate-400 mt-0.5";
    }
  }

  function updateVerdictUI(verdict) {
    if (verdict === "ALERT") {
      verdictTitle.innerText = "AI (SYNTHETIC ALERT)";
      verdictTitle.className = "text-3xl sm:text-4xl md:text-5xl font-black tracking-tight uppercase transition-colors duration-200 text-rose-500";
      
      verdictCard.className = "lg:col-span-8 hud-border rounded-2xl p-6 relative overflow-hidden transition-all duration-300 flex flex-col justify-between min-h-[260px] border-rose-500 bg-rose-950/30 shadow-2xl shadow-rose-950/60 animate-glow-red";
      
      verdictDescription.innerHTML = `
        <span class="text-rose-400 font-semibold tracking-wide">AI SYNTHETIC SPEECH DETECTED:</span> Neural vocoder artifacts, abnormal spectral roll-off boundary below 5.8kHz, and rigid pitch micro-stability identified.
      `;

      verdictIconWrap.className = "w-16 h-16 rounded-2xl flex items-center justify-center border transition-all duration-300 bg-rose-500/20 border-rose-500 text-rose-400 shadow-lg shadow-rose-500/40";
      verdictIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />`;

    } else if (verdict === "AUTHENTIC") {
      verdictTitle.innerText = "HUMAN (AUTHENTIC)";
      verdictTitle.className = "text-3xl sm:text-4xl md:text-5xl font-black tracking-tight uppercase transition-colors duration-200 text-emerald-400";
      
      verdictCard.className = "lg:col-span-8 hud-border rounded-2xl p-6 relative overflow-hidden transition-all duration-300 flex flex-col justify-between min-h-[260px] border-emerald-500 bg-emerald-950/30 shadow-2xl shadow-emerald-950/60 animate-glow-green";
      
      verdictDescription.innerHTML = `
        <span class="text-emerald-400 font-semibold tracking-wide">AUTHENTIC HUMAN VOICE:</span> Biological glottal micro-tremor and rich natural high-frequency acoustics confirmed.
      `;

      verdictIconWrap.className = "w-16 h-16 rounded-2xl flex items-center justify-center border transition-all duration-300 bg-emerald-500/20 border-emerald-500 text-emerald-400 shadow-lg shadow-emerald-500/40";
      verdictIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />`;

    } else {
      verdictTitle.innerText = "STANDBY";
      verdictTitle.className = "text-3xl sm:text-4xl md:text-5xl font-black tracking-tight uppercase transition-colors duration-200 text-slate-500";
      
      verdictCard.className = "lg:col-span-8 hud-border rounded-2xl p-6 relative overflow-hidden transition-all duration-300 flex flex-col justify-between min-h-[260px]";
      
      verdictDescription.innerText = isRecording 
        ? "Sentinel active. Listening for voice input..."
        : 'Sentinel in standby. Press "START SENTINEL" or run simulations below.';

      verdictIconWrap.className = "w-16 h-16 rounded-2xl flex items-center justify-center border transition-all duration-300 bg-slate-800/40 border-slate-700/60 text-slate-500";
      verdictIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z" />`;
    }
  }

  // --- Web Audio API 16kHz Real-Time Streaming ---
  async function startAudioStream() {
    stopSimulation();

    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: TARGET_SAMPLE_RATE,
        latencyHint: "interactive",
      });

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      const source = audioContext.createMediaStreamSource(mediaStream);
      const actualSampleRate = audioContext.sampleRate;
      console.log(`[AudioSentry] AudioContext sample rate: ${actualSampleRate} Hz`);

      const bufferSize = 2048;
      scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      pcmBuffer = [];

      scriptProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);

        // Update oscilloscope inspection buffer
        for (let i = 0; i < oscilloscopeWaveform.length; i++) {
          const idx = Math.floor((i / oscilloscopeWaveform.length) * inputData.length);
          oscilloscopeWaveform[i] = inputData[idx];
        }

        // Resample if native audio hardware is locked at 44.1k/48k
        let resampled;
        if (actualSampleRate === TARGET_SAMPLE_RATE) {
          resampled = inputData;
        } else {
          resampled = resampleLinear(inputData, actualSampleRate, TARGET_SAMPLE_RATE);
        }

        // Chunk audio into 50ms frames (800 samples at 16kHz)
        for (let j = 0; j < resampled.length; j++) {
          pcmBuffer.push(resampled[j]);

          if (pcmBuffer.length >= SAMPLES_PER_CHUNK) {
            const chunk = new Float32Array(pcmBuffer.slice(0, SAMPLES_PER_CHUNK));
            pcmBuffer = pcmBuffer.slice(SAMPLES_PER_CHUNK);

            // Stream raw Float32Array PCM buffer over WebSocket
            if (socket && socket.readyState === WebSocket.OPEN) {
              socket.send(chunk.buffer);
            }
          }
        }
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      isRecording = true;
      toggleMicBtn.className = "flex items-center space-x-2 px-4 py-2 rounded-lg font-mono text-xs font-semibold uppercase tracking-wider transition-all duration-200 bg-rose-500 hover:bg-rose-400 text-white shadow-lg shadow-rose-500/25 active:scale-95";
      micBtnText.innerText = "STOP SENTINEL";
      verdictDescription.innerText = "Sentinel active. Analyzing incoming 16kHz audio stream...";

    } catch (err) {
      console.error("Microphone access failed:", err);
      alert(`Could not start microphone: ${err.message}. Please allow mic access.`);
      stopAudioStream();
    }
  }

  function stopAudioStream() {
    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }

    pcmBuffer = [];
    isRecording = false;

    toggleMicBtn.className = "flex items-center space-x-2 px-4 py-2 rounded-lg font-mono text-xs font-semibold uppercase tracking-wider transition-all duration-200 bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/25 active:scale-95";
    micBtnText.innerText = "START SENTINEL";

    handleTelemetry({
      is_speech: false,
      verdict: "STANDBY",
      confidence: 100.0,
      synthetic_risk: 0.0,
      spectral_rolloff_hz: 0,
      phase_jitter: 0,
      pitch_jitter_percent: 0,
      hf_energy_ratio: 0,
      rms_level: 0,
      mel_bins: new Array(N_MELS).fill(0.0),
    });
  }

  function resampleLinear(input, inRate, outRate) {
    const ratio = inRate / outRate;
    const outLength = Math.round(input.length / ratio);
    const output = new Float32Array(outLength);

    for (let i = 0; i < outLength; i++) {
      const srcIndex = i * ratio;
      const indexFloor = Math.floor(srcIndex);
      const frac = srcIndex - indexFloor;
      const s0 = input[indexFloor] || 0;
      const s1 = input[indexFloor + 1] !== undefined ? input[indexFloor + 1] : s0;
      output[i] = s0 + frac * (s1 - s0);
    }
    return output;
  }

  // --- Synthetic & Authentic Speech Simulators ---
  function startSimulation(mode) {
    stopAudioStream();
    stopSimulation();

    let simPhase = 0;
    let frameCount = 0;
    const baseF0 = 145.0; // Fundamental pitch in Hz

    simulationInterval = setInterval(() => {
      frameCount++;
      const chunk = new Float32Array(SAMPLES_PER_CHUNK); // 800 samples
      const dt = 1.0 / TARGET_SAMPLE_RATE;

      if (mode === "synthetic") {
        // Neural Vocoder Simulation:
        // 1. Rigid pitch contour (zero glottal jitter)
        // 2. High-frequency brick-wall suppression < 5.5 kHz
        // 3. Static harmonic phase locks
        const currentF0 = baseF0;

        for (let i = 0; i < SAMPLES_PER_CHUNK; i++) {
          simPhase += 2 * Math.PI * currentF0 * dt;
          let s = 0.22 * Math.sin(simPhase);
          s += 0.14 * Math.sin(simPhase * 2.0);
          s += 0.08 * Math.sin(simPhase * 3.0);
          s += 0.04 * Math.sin(simPhase * 4.0);
          chunk[i] = s + (Math.random() - 0.5) * 0.005;
        }
      } else {
        // Authentic Human Phonation Simulation:
        // 1. Glottal micro-tremor (1.2% cycle-to-cycle jitter)
        // 2. High frequency fricative dispersion extending to 7.5 kHz
        // 3. Natural phase turbulence
        const jitterF0 = baseF0 * (1.0 + Math.sin(frameCount * 0.4) * 0.012 + (Math.random() - 0.5) * 0.015);

        for (let i = 0; i < SAMPLES_PER_CHUNK; i++) {
          simPhase += 2 * Math.PI * jitterF0 * dt;
          let s = 0.20 * Math.sin(simPhase);
          s += 0.12 * Math.sin(simPhase * 2.0 + Math.sin(i * 0.02));
          s += 0.07 * Math.sin(simPhase * 3.0);
          s += 0.04 * Math.sin(simPhase * 4.0);
          s += 0.02 * Math.sin(simPhase * 6.0);
          s += (Math.random() - 0.5) * 0.05; // Air fricatives
          chunk[i] = s;
        }
      }

      // Update oscilloscope buffer
      for (let i = 0; i < oscilloscopeWaveform.length; i++) {
        const idx = Math.floor((i / oscilloscopeWaveform.length) * chunk.length);
        oscilloscopeWaveform[i] = chunk[idx];
      }

      // Send to WebSocket
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(chunk.buffer);
      }
    }, 50);
  }

  function stopSimulation() {
    if (simulationInterval) {
      clearInterval(simulationInterval);
      simulationInterval = null;
    }
  }

  // --- Canvas Rendering Loop ---
  function resizeCanvases() {
    const dpr = window.devicePixelRatio || 1;
    [oscCanvas, melCanvas].forEach((canvas) => {
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
      }
    });
  }

  // 1. Draw Live 16kHz Oscilloscope
  function drawOscilloscope() {
    const width = oscCanvas.width;
    const height = oscCanvas.height;
    if (width === 0 || height === 0) return;

    oscCtx.clearRect(0, 0, width, height);
    oscCtx.fillStyle = "#030712";
    oscCtx.fillRect(0, 0, width, height);

    // Phosphor grid
    oscCtx.strokeStyle = "rgba(15, 23, 42, 0.9)";
    oscCtx.lineWidth = 1;

    for (let i = 1; i < 6; i++) {
      const y = (height / 6) * i;
      oscCtx.beginPath();
      oscCtx.moveTo(0, y);
      oscCtx.lineTo(width, y);
      oscCtx.stroke();
    }
    for (let i = 1; i < 8; i++) {
      const x = (width / 8) * i;
      oscCtx.beginPath();
      oscCtx.moveTo(x, 0);
      oscCtx.lineTo(x, height);
      oscCtx.stroke();
    }

    // Zero-axis line
    oscCtx.strokeStyle = "rgba(30, 41, 59, 0.8)";
    oscCtx.beginPath();
    oscCtx.moveTo(0, height / 2);
    oscCtx.lineTo(width, height / 2);
    oscCtx.stroke();

    // Waveform trace
    oscCtx.beginPath();
    const len = oscilloscopeWaveform.length;
    const step = width / len;

    for (let i = 0; i < len; i++) {
      const sample = oscilloscopeWaveform[i];
      const y = height / 2 - sample * (height * 0.42);
      const x = i * step;
      if (i === 0) oscCtx.moveTo(x, y);
      else oscCtx.lineTo(x, y);
    }

    oscCtx.strokeStyle = "#06b6d4"; // Phosphor Cyan
    oscCtx.lineWidth = 2.2;
    oscCtx.shadowColor = "#06b6d4";
    oscCtx.shadowBlur = 8;
    oscCtx.stroke();
    oscCtx.shadowBlur = 0;
  }

  // 2. Draw 64-Bin Mel-Spectrogram Heatmap
  function drawMelSpectrogram() {
    const width = melCanvas.width;
    const height = melCanvas.height;
    if (width === 0 || height === 0) return;

    melCtx.clearRect(0, 0, width, height);
    melCtx.fillStyle = "#030712";
    melCtx.fillRect(0, 0, width, height);

    const barWidth = width / N_MELS;
    const criticalBinIndex = Math.floor(N_MELS * (5800 / 8000)); // Index corresponding to ~5.8kHz

    for (let i = 0; i < N_MELS; i++) {
      const x = i * barWidth;
      const energy = liveMelBins[i] || 0.0;
      const barHeight = Math.max(3, energy * height * 0.88);
      const y = height - barHeight;

      // Colormap gradient (Deep Purple -> Cyan -> Emerald -> Yellow/Rose)
      let fill;
      if (i >= criticalBinIndex) {
        // High frequency band (> 5.8kHz)
        fill = energy > 0.25 ? "#10b981" : "#f43f5e";
      } else {
        if (energy > 0.65) fill = "#38bdf8";
        else if (energy > 0.35) fill = "#818cf8";
        else fill = "#6366f1";
      }

      melCtx.fillStyle = fill;
      melCtx.fillRect(x + 1, y, barWidth - 2, barHeight);

      // Top intensity heat cap
      melCtx.fillStyle = "rgba(255, 255, 255, 0.4)";
      melCtx.fillRect(x + 1, y, barWidth - 2, 2);
    }

    // Critical 5.8kHz Cutoff Marker Line
    const cutoffX = criticalBinIndex * barWidth;
    melCtx.save();
    melCtx.strokeStyle = "#f43f5e";
    melCtx.lineWidth = 1.8;
    melCtx.setLineDash([4, 3]);
    melCtx.shadowColor = "#f43f5e";
    melCtx.shadowBlur = 6;
    melCtx.beginPath();
    melCtx.moveTo(cutoffX, 0);
    melCtx.lineTo(cutoffX, height);
    melCtx.stroke();
    melCtx.restore();
  }

  function animationLoop() {
    resizeCanvases();
    drawOscilloscope();
    drawMelSpectrogram();
    requestAnimationFrame(animationLoop);
  }

  // --- Event Listeners ---
  toggleMicBtn.addEventListener("click", () => {
    if (!isRecording) {
      startAudioStream();
    } else {
      stopAudioStream();
    }
  });

  simSyntheticBtn.addEventListener("click", () => {
    startSimulation("synthetic");
  });

  simHumanBtn.addEventListener("click", () => {
    startSimulation("human");
  });

  // Start on load
  window.addEventListener("DOMContentLoaded", () => {
    initWebSocket();
    requestAnimationFrame(animationLoop);
  });
})();
