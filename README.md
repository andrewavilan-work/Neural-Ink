# 🧠 Neural-Ink V5 — AMD Pervasive AI Edition

![License](https://img.shields.io/badge/License-GPL%20v3-green.svg)
![Unity](https://img.shields.io/badge/Unity-6000.0+-black.svg?logo=unity)
![AMD](https://img.shields.io/badge/AMD-MI300X%20ROCm-ED1C24.svg?logo=amd)
![Sentis](https://img.shields.io/badge/AI-Pervasive%20AI-0078D4.svg)

**Neural-Ink V5** is a high-fidelity **Pervasive AI** pipeline for real-time Neural Style Transfer. It bridges the gap from **Cloud to Client**, utilizing **AMD Instinct™ MI300X** for high-performance training and **AMD Radeon™/Ryzen™ AI** hardware for seamless, real-time inference in Unity via **Unity Sentis**.

This project solves the core challenges of temporal stability and depth awareness in generative graphics. **Highly scalable by design**, the pipeline enables a seamless transition from massive cloud-based training to local execution on consumer devices, including laptops equipped with **AMD Ryzen™ AI NPUs**, thanks to its standard ONNX-based deployment.

---

## 📂 Repository Structure

The project is structured to demonstrate a complete **Pervasive AI** lifecycle:

1.  **`1_AMD_Cloud_Training/`**: The **Cloud** engine. Contains scripts for ROCm 6.2 and MI300X-optimized training.
2.  **`2_Model_Export/`**: The **Optimization** bridge. Tools to convert and quantize models (FP16) for edge deployment.
3.  **`3_Unity_Package/`**: The **Client** runtime. Plug-and-play Unity assets (C#, Shaders, Compute).
4.  **`4_Final_Models/`**: Pre-trained production models optimized for AMD hardware.

---

## 🎮 Unity Deployment Guide (Step-by-Step)

### 1. Requirements
- **Unity 6 (6000.0+)** or **Unity 2022/2023 LTS**.
- **Universal Render Pipeline (URP)**.
- **Unity Sentis** (v2.x recommended for Unity 6).

### 2. Installation
1.  **Install Sentis**: Go to `Window > Package Manager`, search for "Sentis" or "Inference Engine" and install it.
2.  **Import Assets**: Copy the `3_Unity_Package` folder into your project's `Assets` directory.
3.  **Setup Camera**:
    - Select your **Main Camera**.
    - Attach the `NeuralSentisV5.cs` script.
    - Assign the `NeuralInk_V5_FP16.onnx` model (from `4_Final_Models`) to the **Onnx Model** slot.
    - Assign the `Pack7Channel` Compute Shader and `TemporalWarp` Material (included in the `Shaders` folder).

### 3. Configure URP Render Features
The pipeline requires two Render Features to capture data and display the result:
1.  Locate your **Universal Renderer Data** (usually in `Settings/`).
2.  Click **Add Renderer Feature** and add:
    - **`Neural Capture Feature`**: Set it to execute `After Transparents`.
    - **`Neural Visor Feature`**: Set it to execute `After Post Process`.

### 4. Critical Settings for Quality
- **Depth Texture**: Ensure "Depth Texture" is enabled in your URP Asset and on your Main Camera.
- **Anti-Aliasing**: If using **TAA**, the AI might flicker due to camera jitter. We recommend using **SMAA** or disabling AA for the cleanest neural output.

---

## ☁️ Cloud Training Guide (AMD MI300X)

### 1. Environment Setup
1.  Upload `1_AMD_Cloud_Training` to your ROCm cloud instance.
2.  Run the setup script: `chmod +x setup_mi300x.sh && ./setup_mi300x.sh`.
    - *This configures ROCm 6.2, GFX942 architecture, and installs PyTorch.*

### 2. Training
1.  Place your reference style images in the `styles/` folder.
2.  Execute: `python train_mi300x_hackathon.py`.
3.  Monitor GPU usage with `watch -n 1 rocm-smi`.

---

## 🛠️ Optimization & Scalability
Use the tools in `2_Model_Export` to prepare the model for different hardware:
- **`export_to_onnx_v5.py`**: Generates a dynamic ONNX graph.
- **`optimize_to_fp16.py`**: Converts to Half-Precision. This is **essential** for real-time performance on **Ryzen™ AI NPUs** and **Radeon™ GPUs**.

---

## 🏆 AMD Pervasive AI Developer Contest 2026
This project showcases the full spectrum of AMD's AI ecosystem—from massive-scale training on **CDNA™ 3** accelerators to real-time execution on consumer hardware via **DirectML**.

---
*Built for the future of pervasive generative graphics.*
