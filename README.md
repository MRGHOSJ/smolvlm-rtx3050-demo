# 🤖 SmolVLM: GPU-Optimized Real-Time Image Captioning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**Run powerful Vision-Language Models (VLMs) locally on consumer-grade hardware.**

Optimized specifically for 6GB VRAM constraints (e.g., NVIDIA RTX 3050), this project provides real-time image captioning and analysis using **HuggingFaceTB/SmolVLM-Instruct**.

</div>

---

## 🎯 Project Goal

**Note:** This project was developed as a personal learning exercise to explore the capabilities of modern Vision-Language Models (VLMs). 

The primary objectives were to:
1.  **Test the SmolVLM model** in a real-time environment.
2.  **Understand the practical limitations** of running AI models on consumer-grade hardware (6GB VRAM).
3.  **Implement memory optimization techniques** (quantization, precision handling, and garbage collection) to ensure stable inference.

---

## ✨ Features

- **🚀 GPU Optimized**: Configured specifically for 6GB VRAM constraints without sacrificing quality.
- **🧠 Multiple Inference Modes**: Includes 3 distinct processing methods (Chat Template, Simple Chat, Direct Prompt) to find the optimal balance between speed and accuracy for your hardware.
- **📹 Live Webcam Feed**: Real-time video capture and processing using OpenCV with a clean on-screen display (OSD).
- **⚡ Automatic Precision Handling**: Intelligently attempts to load in `float16` for speed and memory efficiency, with a robust fallback to `float32` for older architectures.
- **🧹 Smart Memory Management**: Active garbage collection and cache clearing to prevent Out-Of-Memory (OOM) errors during continuous use.
- **🔌 Hardware Auto-Detection**: Automatically detects your specific GPU model and adjusts system metrics accordingly.

---

## 📋 Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Tested on RTX 3050 6GB; compatible with 4GB+ VRAM).
- **RAM**: 8GB+ System RAM recommended.
- **Camera**: Webcam or USB camera connected to the system.

### Software
- **Python**: 3.8 or higher.
- **CUDA**: Toolkit 11.8 or 12.x installed.
- **Drivers**: Up-to-date NVIDIA GPU drivers.

---

## 🚀 Installation

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/smolvlm-realtime.git
cd smolvlm-realtime
\`\`\`

### 2. Install PyTorch (Critical Step)
**Do not install PyTorch via requirements.txt.** You must install the version compiled for your specific CUDA version to enable GPU acceleration.

**For CUDA 11.8:**
\`\`\`bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
\`\`\`

**For CUDA 12.x:**
\`\`\`bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
\`\`\`

### 3. Install Python Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## 💻 Usage

#### 1. Run the Script
\`\`\`bash
python smolvlm_test.py
\`\`\`

#### 2. Controls
Once the window opens, use the following keys:

- **\`SPACE\`**: Capture a frame and analyze it.
- **\`1\`, \`2\`, \`3\`**: Switch between processing methods in real-time (see below).
- **\`q\`**: Quit the application.

---

## 🔧 Processing Methods

The script includes three distinct ways to prompt the model. You can switch between them in real-time to see which works best for your hardware.

| Method | Description | Speed | Stability |
| :--- | :--- | :--- | :--- |
| **1. Chat Template** | Uses the standard \`apply_chat_template\` function. Strictly formatted prompt with "user" and "assistant" roles. | Medium | High |
| **2. Simple Chat** | A simplified conversation dictionary passed directly to the processor. | Medium | Medium |
| **3. Direct Prompt** | **(Recommended)** Bypasses complex chat templates for a raw image-to-text approach. Fastest inference method. | **Fastest** | High |

---

## ⚙️ Configuration

You can tweak these variables at the top of \`smolvlm_test.py\` to fit your specific hardware limits:

- **\`MODEL_ID\`**: The HuggingFace model ID (Default: \`HuggingFaceTB/SmolVLM-Instruct\`).
- **\`MAX_TOKENS\`**: Limits the length of the generated response (Default: 50).
- **\`IMAGE_SIZE\`**: Resizes the input image to save memory. If you encounter OOM errors, lower this to \`256\` or \`224\` (Default: 384).

---

## 📊 Performance Benchmarks

*Tested on NVIDIA RTX 3050 6GB Laptop*

| Configuration | VRAM Usage | Inference Speed |
| :--- | :--- | :--- |
| **Float16 (Method 3)** | ~4.2 GB | ~2.0 seconds |
| **Float16 (Method 1)** | ~4.8 GB | ~4.5 seconds |
| **Float32 (Fallback)** | ~5.8 GB | ~6.0+ seconds |

> *Note: Performance varies based on image complexity, background processes, and thermal throttling.*

---

## 🐛 Troubleshooting

### "CUDA out of memory"
1.  **Lower \`IMAGE_SIZE\`**: Edit the script and set \`IMAGE_SIZE = 256\`.
2.  **Close apps**: Close browser tabs or games using the GPU.
3.  **Restart script**: This clears the VRAM cache completely.

### "Float16 failed"
The script will automatically fallback to \`float32\`. This uses more VRAM but is more stable on older GPU architectures. If this happens, ensure no other heavy GPU tasks are running.

### Latency is too high
1.  Switch to **Method 3** (Direct Prompt) by pressing \`3\`.
2.  Lower \`MAX_TOKENS\` in the script config if you only need short descriptions.
3.  Check if your GPU is thermal throttling (high temps reduce performance).

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
