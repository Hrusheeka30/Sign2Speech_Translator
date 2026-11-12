"""
Multilingual translation + text-to-speech for the Sign-to-Speech app.

Primary  : Azure Translator  (text) + Azure Speech (speech)  -- via REST, no SDK.
Fallback : deep-translator + gTTS -- used automatically ONLY when Azure keys
           are not configured, so the app still runs while you set Azure up.

Configuration (create a file named `.env` in the project root):

    AZURE_TRANSLATOR_KEY=your_translator_key
    AZURE_TRANSLATOR_REGION=centralindia
    AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com
    AZURE_SPEECH_KEY=your_speech_key
    AZURE_SPEECH_REGION=centralindia

Both resources are "Azure AI services" -> Translator and Speech, on the free tier.
"""

import io
import json
import os
import threading

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LANGS = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
}

# Azure Speech neural voices (Indian English + the four Indian languages)
AZURE_VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "kn": "kn-IN-GaganNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
}

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "centralindia")
AZURE_TRANSLATOR_ENDPOINT = os.getenv(
    "AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "centralindia")

HAVE_AZURE_TRANSLATOR = bool(AZURE_TRANSLATOR_KEY)
HAVE_AZURE_SPEECH = bool(AZURE_SPEECH_KEY)

# ---------------------------------------------------------------------------
# On-disk translation cache: the sign vocabulary is a small, fixed list, so we
# translate each (word, language) once and reuse it forever. This keeps results
# stable and stays clear of the free providers' rate limits.
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "translations_cache.json")
_cache_lock = threading.Lock()


def _load_cache():
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_TRANSLATION_CACHE = _load_cache()

# provider error text / obvious junk that must never reach the UI
_BAD_MARKERS = ("translation failed", "too many requests", "server error",
                "quota", "you are allowed to make")


def _plausible(out, src):
    """Reject empty results, provider error text, and wildly-expanded output
    (e.g. MyMemory turning the word "Bedroom" into a whole sentence)."""
    if not out or not out.strip():
        return False
    out = out.strip()
    low = out.lower()
    if any(m in low for m in _BAD_MARKERS):
        return False
    src_words = max(1, len(src.split()))
    out_words = len(out.split())
    if len(src) <= 16 and out_words > src_words + 2:
        return False
    if len(out) > 3 * len(src) + 18:
        return False
    return True


def _azure_translate(text, target_lang):
    url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version=3.0&from=en&to={target_lang}"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=[{"Text": text}], timeout=15)
    r.raise_for_status()
    return r.json()[0]["translations"][0]["text"]


def _azure_speak_bytes(text, lang):
    voice = AZURE_VOICES[lang]
    ssml = (
        f"<speak version='1.0' xml:lang='{lang}-IN'>"
        f"<voice name='{voice}'>{text}</voice></speak>"
    )
    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        "User-Agent": "sign-to-speech-app",
    }
    r = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=20)
    r.raise_for_status()
    return r.content


def _google_free(text, target_lang):
    """Google's public gtx endpoint (the one the mobile app uses). More reliable
    than deep-translator's scrape endpoint, which is rate-limited to 5 req/s."""
    r = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "en", "tl": target_lang, "dt": "t", "q": text},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return "".join(seg[0] for seg in data[0] if seg and seg[0])


def _fallback_translate(text, target_lang):
    """Try free providers in order: Google gtx -> Google scrape -> MyMemory.
    MyMemory is last because it often returns a whole-sentence match that is
    wrong for a single sign word."""
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    errors = []
    for name, fn in (
        ("google", lambda t: _google_free(t, target_lang)),
        ("google-scrape", lambda t: GoogleTranslator(source="en", target=target_lang).translate(t)),
        ("mymemory", lambda t: MyMemoryTranslator(source="en-US", target=f"{target_lang}-IN").translate(t)),
    ):
        try:
            out = fn(text)
        except Exception as e:  # noqa: BLE001 - try the next provider
            errors.append(f"{name}: {e}")
            continue
        if _plausible(out, text):
            return out.strip()
        errors.append(f"{name}: implausible result {out!r}")
    raise RuntimeError("; ".join(errors))


def translate(text, target_lang):
    """Translate English `text` into `target_lang` (en/hi/kn/ta/te)."""
    if not text:
        return text
    if target_lang not in LANGS:
        raise ValueError(f"Unsupported language: {target_lang}")
    if target_lang == "en":
        return text

    with _cache_lock:
        hit = _TRANSLATION_CACHE.get(target_lang, {}).get(text)
    if hit:
        return hit

    if HAVE_AZURE_TRANSLATOR:
        out = _azure_translate(text, target_lang)
    else:
        out = _fallback_translate(text, target_lang)

    if not _plausible(out, text):
        raise RuntimeError(f"unreliable translation: {text!r} -> {target_lang}")

    out = out.strip()
    with _cache_lock:
        _TRANSLATION_CACHE.setdefault(target_lang, {})[text] = out
        _save_cache(_TRANSLATION_CACHE)
    return out


def speak_bytes(text, lang):
    """Return speech audio bytes (mp3) for `text` in `lang`."""
    if lang not in LANGS:
        raise ValueError(f"Unsupported language: {lang}")
    if HAVE_AZURE_SPEECH:
        return _azure_speak_bytes(text, lang)
    # fallback (no Azure keys configured)
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def backend_info():
    return {
        "translator": "azure" if HAVE_AZURE_TRANSLATOR else "deep-translator (fallback)",
        "speech": "azure" if HAVE_AZURE_SPEECH else "gTTS (fallback)",
    }


if __name__ == "__main__":
    print("Backends:", backend_info())
    for code in LANGS:
        try:
            print(code, "->", translate("Thank you", code))
        except Exception as e:
            print(code, "translation failed:", e)
