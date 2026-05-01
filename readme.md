# Telkom University Complaint Classification System

Sistem klasifikasi keluhan mahasiswa Telkom University menggunakan LLM (Large Language Model).

## 🚀 Fitur

- Klasifikasi otomatis keluhan ke 7 direktorat
- Support multiple LLM models (Qwen, Llama, Mistral, dll)
- Web interface dengan real-time progress
- Custom prompt management
- Export hasil ke Excel

## 📋 Requirements

Python 3.8+ dengan dependencies di `requirements.txt`

## 🛠 Installation

1. Clone repository:
```bash
git clone https://github.com/username/your-repo.git
cd your-repo
```
# aktivasi environment
```bash
python -m venv venv
venv\Scripts\activate
```
```bash
pip install -r requirements.txt 

```
# install untuk pengguna nvidia
```bash
$env:CMAKE_ARGS='-DGGML_CUDA=on'
$env:FORCE_CMAKE='1'
pip install --force-reinstall --no-cache-dir llama-cpp-python
```
# Download LLM Models

## Required Models

Download model GGUF format dan letakkan di folder `model/MODEL_NAME/`

### Recommended Model:
- **Qwen2.5-4B-Instruct-GGUF**
  - Download: [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-4B-Instruct-GGUF)
  - File: `qwen2.5-4b-instruct-q4_0.gguf`
  - Folder: `model/Qwen2.5-4B-Instruct/`

### Alternative Models:
- Llama 3.2 3B Instruct
- Mistral 7B Instruct
- Phi-3 Mini

## Setup Steps:

1. Create model folder:
```bash
mkdir -p model/Qwen2.5-4B-Instruct/