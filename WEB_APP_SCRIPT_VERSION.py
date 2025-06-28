import os
import tempfile
import subprocess

import cv2
import torch
import whisper
import easyocr
import numpy as np
from PIL import Image
from tqdm import tqdm
import gradio as gr

from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
)


# 1. Audio extraction + Whisper transcription
def transcribe_video(video_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    # extract single‐channel 16 kHz audio
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    model = whisper.load_model("medium")
    # remove unsupported 'device' kwarg
    result = model.transcribe(audio_path, fp16=False, verbose=False)
    os.remove(audio_path)

    lines = []
    for seg in result["segments"]:
        if seg.get("avg_logprob", -100) > -1.2:
            lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text'].strip()}")
    return lines


# 2. Keyframe extraction with CLIP
def extract_keyframes(video_path, similarity_threshold=0.85):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cpu()
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    prev_emb = None
    frames, idx = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # sample one frame per second (as originally)
        if idx % int(fps) == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = clip_proc(images=img, return_tensors="pt")
            emb = clip_model.get_image_features(**inputs).detach()
            if prev_emb is None or torch.cosine_similarity(emb, prev_emb) < similarity_threshold:
                frames.append(img)
                prev_emb = emb
        idx += 1
    cap.release()
    return frames


# 3. BLIP captions
def caption_frames(frames):
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).cpu()

    captions = []
    for img in frames:
        inpt = proc(images=img, return_tensors="pt")
        out = model.generate(**inpt)
        captions.append(proc.decode(out[0], skip_special_tokens=True))
    return captions


# 4. Windowed summarization with T5 (large)
def summarize_windows(captions, window_size=5):
    tok = T5Tokenizer.from_pretrained("google/flan-t5-large", use_fast=True)
    mdl = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").cpu()

    summaries = []
    for i in range(0, len(captions), window_size):
        chunk = captions[i:i+window_size]
        prompt = "summarize: " + " ; ".join(chunk)
        ids = tok(prompt, return_tensors="pt").input_ids
        out = mdl.generate(ids, max_length=60, num_beams=4)
        summaries.append(tok.decode(out[0], skip_special_tokens=True))
    return summaries


# 5. OCR via EasyOCR
def ocr_frames(frames):
    reader = easyocr.Reader(['en'], gpu=False)
    lines = []
    for img in frames:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        txts = reader.readtext(arr, detail=0)
        txts = [t.strip() for t in txts if t.strip()]
        if txts:
            lines.append(" | ".join(txts))
    return lines


# 6. Aggregate “facts”
def aggregate(transcript, summaries, ocr_lines):
    out = ["=== Transcript ==="] + [f"- {l}" for l in transcript]
    out += ["\n=== Frame Summaries ==="] + [f"- {s}" for s in summaries]
    out += ["\n=== OCR Text ==="]       + [f"- {o}" for o in ocr_lines]
    return "\n".join(out)


# 7. Final description: T5‑large + CLIP re‑ranking
def generate_description(facts, keyframes):
    tok = T5Tokenizer.from_pretrained("google/flan-t5-large", use_fast=True)
    mdl = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").cpu().eval()
    input_ids = tok("generate a concise video description:\n\n" + facts,
                    return_tensors="pt").input_ids
    outs = mdl.generate(input_ids, max_length=200, num_beams=5)
    desc = tok.decode(outs[0], skip_special_tokens=True)

    desc += "\n\nContact: info@iboothme.com | +971 4 448 8563 | https://www.iboothme.com"
    return desc


# 8. Expand + hashtags
def expand_and_hashtag(text):
    tok = T5Tokenizer.from_pretrained("google/flan-t5-large", use_fast=True)
    mdl = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").cpu().eval()
    sents = [s for s in text.split('.') if s]
    prompt = "expand and enrich: " + ". ".join(sents[:3])
    inp = tok(prompt, return_tensors="pt").input_ids
    out = mdl.generate(inp, max_new_tokens=100, num_beams=4, no_repeat_ngram_size=2)
    exp = tok.decode(out[0], skip_special_tokens=True).strip()

    from collections import Counter
    words = [w.lower() for w in exp.split() if w.isalpha()]
    common = Counter(words).most_common(15)
    tags = " ".join(f"#{w}" for w,_ in common)
    return exp + "\n\n" + tags


# Gradio interface
def process(video_path: str):
    # note: we now take the path directly
    trans = transcribe_video(video_path)
    frames = extract_keyframes(video_path)
    caps   = caption_frames(frames)
    sums   = summarize_windows(caps)
    ocr_l  = ocr_frames(frames)
    facts  = aggregate(trans, sums, ocr_l)
    desc   = generate_description(facts, frames)
    final  = expand_and_hashtag(desc)
    return final


iface = gr.Interface(
    fn=process,
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Textbox(label="Generated Description"),
    title="Gardio: Video → Auto Description",
    description="Upload any video and get an AI‑driven, richly expanded description."
)

if __name__ == "__main__":
    iface.launch(share=True)

