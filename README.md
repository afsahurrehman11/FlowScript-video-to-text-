 <img src="./Images-for-ReadMe/1.png" >
 

# 🎥 FlowScript: Unleashing Video Stories with 3D Motion Capture, Sequential Analysis, and AI Precision ✍️  
**_Far Beyond Basic Video-to-Text—Every Movement, Every Detail, Perfected_**

**FlowScript** isn’t just another video captioning tool—it’s a sophisticated video analysis system that captures **3D movements** through sequential frame processing, understands every aspect of your video, and delivers pinpoint-accurate marketing narratives. Powered by **OpenAI's Whisper**, **Salesforce's BLIP**, **Google's Flan-T5**, and **OpenAI's CLIP**, it goes beyond static descriptions to reveal the full story, with CLIP ensuring unmatched precision by selecting the best description from multiple candidates.

---

## Overview

- **Audio Transcription**: FFmpeg extracts audio; **OpenAI's Whisper Medium** transcribes with high fidelity (avg_logprob > -1.2) → `whisper_out/filtered.txt`.
- **Unique Frames**: **OpenAI's CLIP (laion/CLIP-ViT-B-32-laion2B-s34B-b79K)** identifies diverse frames (similarity < 0.85) → `keyframes/`.
- **Independent Captioning**: **Salesforce's BLIP (blip-image-captioning-base)** captions each frame → `captions/blip_captions.txt`.
- **3D Motion Capture**: **Google's Flan-T5 Small** analyzes sequences of 5 frames for movement → `captions/window_summaries.txt`.
- **On-Screen Text**: **EasyOCR** extracts visible text → `ocr/all_ocr.txt`.
- **Comprehensive Synthesis**: Combines all data → `facts/facts.txt`.
- **Precision Descriptions**:  
  - **Google's Flan-T5 Large** generates 3 candidate narratives.  
  - **OpenAI's CLIP (clip-vit-base-patch32)** selects the most accurate by matching to keyframes.  
- **Final Output**: Adds hashtags & contact info → `final_description.txt`, `expanded_with_hashtags.txt`.

---

## Detailed Process: How FlowScript Stands Out

Unlike basic video-to-text tools, **FlowScript** captures the full essence of your video—static scenes, dynamic 3D movements, and all contextual details—while ensuring top-tier accuracy. Here’s how:

### 1. 🔊 Audio Transcription  
- **Tools**: FFmpeg, **OpenAI's Whisper Medium**  
- **How It Works**: FFmpeg converts video audio into 16kHz WAV format. **OpenAI's Whisper Medium**, a transformer-based automatic speech recognition model, processes the audio with a confidence filter (avg_logprob > -1.2) to ensure high-quality transcription.  
- **Output**: `whisper_out/filtered.txt`  
 <img src="./Images-for-ReadMe/2.png">

### 2. 🖼️ Independent Frame Captioning  
- **Models**: **OpenAI's CLIP (laion/CLIP-ViT-B-32-laion2B-s34B-b79K)**, **Salesforce's BLIP (blip-image-captioning-base)**  
- **How It Works**: **OpenAI's CLIP**, a Vision Transformer model trained on image-text pairs, computes embeddings to detect unique frames (similarity < 0.85). **Salesforce's BLIP**, a vision-language model, generates detailed captions for each selected frame.  
- **Output**: `keyframes/`, `captions/blip_captions.txt`  
<table>
  <tr>
    <td><img src="./Images-for-ReadMe/3.png"></td>
    <td><img src="./Images-for-ReadMe/4.png"></td>
  </tr>
  <tr>
    <td align="center"><strong>Frame 10 of Video</strong></td>
    <td align="center"><strong>Genrated caption: "a person is using a video camera to take pictures" </strong></td>
  </tr>
</table>


### 3. 🔄 Sequential 3D Motion Analysis  
- **Model**: **Google's Flan-T5 Small**  
- **How It Works**: **Google's Flan-T5 Small**, a text-to-text transformer fine-tuned for instruction tasks, analyzes sequences of 5 frames to summarize motion and context, enabling 3D movement tracking. 
- **Output**: `captions/window_summaries.txt`
<img src="./Images-for-ReadMe/5.png" >
 

### 4. 📖 On-Screen Text Extraction  
- **Tool**: **EasyOCR**  
- **How It Works**: **EasyOCR**, a deep learning-based OCR tool, detects and extracts text from frames using a neural network, followed by post-processing for accuracy.  
- **Output**: `ocr/all_ocr.txt`  
<img src="./Images-for-ReadMe/6.png" >

### 5. 📂 Data Integration  
- **How It Works**: Combines transcripts, captions, motion summaries, and OCR text into a single cohesive file for downstream processing.  
- **Output**: `facts/facts.txt`  
  ![Data Integration Output](path/to/data_integration_screenshot.png)

### 6. ✍️ AI-Powered Narrative Generation  
- **Model**: **Google's Flan-T5 Large**  
- **How It Works**: **Google's Flan-T5 Large**, a larger transformer model optimized for text generation, creates 3 marketing-friendly narrative candidates based on the integrated data.  
- **Output**: Internal candidates for selection.  
  ![Narrative Generation Output](path/to/narrative_generation_screenshot.png)

### 7. ✅ Precision Selection with CLIP  
- **Model**: **OpenAI's CLIP (clip-vit-base-patch32)**  
- **How It Works**: **OpenAI's CLIP**, with its dual image-text encoder, scores the 3 narratives against keyframe embeddings, selecting the most visually aligned description.  
- **Output**: Best description selected.  
  ![CLIP Selection Output](path/to/clip_selection_screenshot.png)

### 8. 🏷️ Final Polish  
- **How It Works**: Extracts key terms from the description to generate relevant hashtags and appends contact info (e.g., email or social media) for a professional finish.  
- **Output**: `final_description.txt`, `expanded_with_hashtags.txt`  
  ![Final Output](path/to/final_output_screenshot.png)

---

## What Makes FlowScript Unique  
- **3D Motion Capture**: Sequential analysis tracks movement, not just static frames.  
- **Holistic Analysis**: Combines audio, visuals, text, and motion for a complete picture.  
- **CLIP-Powered Precision**: Selects the best description by visual matching—accuracy guaranteed.  

---

## Setup  
```bash
pip install opencv-python torch whisper easyocr numpy pillow tqdm torchvision transformers
```  
- Install FFmpeg.  
- Run `main_code.ipynb` step-by-step.

---

## Outputs  
- `final_description.txt`: Core narrative.  
- `expanded_with_hashtags.txt`: Enhanced with hashtags.

---

## Web App  
Upload videos online via our single-file Python app—check the repo!
.
---

