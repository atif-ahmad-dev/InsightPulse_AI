# utils/audio_transcriber.py
"""
InsightPulse AI - Server-Side Speech-to-Text (STT) Engine
Provides cross-browser fallback audio transcription (e.g. for Mozilla Firefox)
using standard 16kHz PCM WAV audio streams.

Engines:
1. SpeechRecognition (Google Free Web Speech API - default, zero API keys required)
2. Google Gemini multimodal audio transcription (if GEMINI_API_KEY provided)
3. OpenAI Whisper transcription (if OPENAI_API_KEY provided)
"""

import os
import sys
import io
import base64
from typing import Tuple, Optional

# Ensure terminal stdout/stderr are UTF-8 safe on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def transcribe_audio_b64(
    audio_b64: str,
    language: str = "en-US",
    custom_gemini_key: Optional[str] = None,
    custom_openai_key: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Transcribes a base64-encoded WAV audio string into text.
    Returns:
        (success: bool, text_or_error_message: str)
    """
    if not audio_b64:
        return False, "No audio data received."

    try:
        # Strip data URL prefix if present (e.g. "data:audio/wav;base64,...")
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]

        audio_bytes = base64.b64decode(audio_b64)
        if len(audio_bytes) < 44:  # WAV header is 44 bytes
            return False, "Audio recording is empty or corrupt."

        return transcribe_audio_bytes(
            audio_bytes,
            language=language,
            custom_gemini_key=custom_gemini_key,
            custom_openai_key=custom_openai_key
        )
    except Exception as e:
        sys.stderr.write(f"[STT ERROR] Failed to decode audio base64: {e}\n")
        return False, f"Failed to decode audio: {str(e)}"

def transcribe_audio_bytes(
    audio_bytes: bytes,
    language: str = "en-US",
    custom_gemini_key: Optional[str] = None,
    custom_openai_key: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Transcribes raw WAV audio bytes into text.
    """
    # Map component language code to STT language codes
    lang_map = {
        "en-US": "en-US",
        "en-IN": "en-IN",
        "hi-IN": "hi-IN",
        "en": "en-US",
        "hi": "hi-IN",
        "hinglish": "en-IN"
    }
    stt_lang = lang_map.get(language, "en-US")

    # Tier 1: Try OpenAI Whisper if user provided key or in env
    openai_key = custom_openai_key or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            wav_file = io.BytesIO(audio_bytes)
            wav_file.name = "speech.wav"
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_file,
                language="hi" if "hi" in stt_lang.lower() else "en"
            )
            if transcript and transcript.text:
                return True, transcript.text.strip()
        except Exception as oai_err:
            sys.stderr.write(f"[STT OPENAI WARNING] Whisper transcription fallback: {oai_err}\n")

    # Tier 2: Free SpeechRecognition Google backend (default, zero keys needed)
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True

        wav_io = io.BytesIO(audio_bytes)
        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)

        try:
            recognized_text = r.recognize_google(audio_data, language=stt_lang)
            if recognized_text and recognized_text.strip():
                return True, recognized_text.strip()
            return False, "No speech detected in audio."
        except sr.UnknownValueError:
            return False, "Could not understand audio. Please speak clearly into your microphone."
        except sr.RequestError as req_err:
            sys.stderr.write(f"[STT GOOGLE REQUEST ERROR] {req_err}\n")
            # If Google request fails and Gemini key exists, try Gemini
    except ImportError:
        sys.stderr.write("[STT WARNING] speech_recognition package not available.\n")
    except Exception as sr_err:
        sys.stderr.write(f"[STT SR ERROR] {sr_err}\n")

    # Tier 3: Gemini multimodal audio transcription (if key exists)
    gemini_key = custom_gemini_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"Listen to this audio recording and transcribe exactly what was spoken in {stt_lang}. "
                "Output ONLY the exact transcribed text, without commentary, quotes, or timestamps."
            )
            response = model.generate_content([
                prompt,
                {"mime_type": "audio/wav", "data": audio_bytes}
            ])
            if response and response.text:
                return True, response.text.strip()
        except Exception as gem_err:
            sys.stderr.write(f"[STT GEMINI ERROR] {gem_err}\n")

    return False, "Could not recognize speech. Please ensure your microphone is working and speak clearly."
