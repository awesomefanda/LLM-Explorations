# AI-Scene-Creation-Take1 🎬

A multimodal AI pipeline that transforms written scripts into narrated visual scenes. This project currently generates **high-quality images with synchronized AI voiceovers**, designed to evolve into a full text-to-motion cinema engine.

## 🚀 Features
* **🎙️ Text-to-Speech:** Uses the `Kokoro` pipeline for fast, high-quality narration.
* **🎨 Text-to-Image:** Generates visuals using `Diffusers` (defaulting to `tiny-sd` for speed and low VRAM usage).
* **🎞️ Auto-Assembly:** Automatically stitches assets into a `.mp4` video with captions using `MoviePy`.
* **🧠 Smart Caching:** Checks if images or audio already exist before generating to save time and compute.

## 🛠️ Tech Stack
* **Language:** Python
* **AI Models:** Kokoro (TTS), Stable Diffusion (Vision)
* **Video Processing:** MoviePy 2.0
* **Hardware:** Optimized for CUDA (NVIDIA GPU) with CPU fallback.

## 📋 Prerequisites

### 1. eSpeak NG
You must have **eSpeak NG** installed for the phonemizer to work.
* **Windows:** [Download eSpeak-NG-64.msi](https://github.com/espeak-ng/espeak-ng/releases)
* Ensure your script points to the correct `.dll` and `.exe` paths.

### 2. Dependencies
```bash
pip install torch soundfile kokoro diffusers moviepy