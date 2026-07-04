# 자막 학습 플레이어 — 단독 로컬 백엔드
# faster-whisper STT(GPU/CPU 자동) + 번역(DeepL 우선 → 로컬 LLM 폴백) + 잡 API + 플레이어 서빙
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Windows CUDA DLL 부트스트랩 — Pinokio 앱 공통 모듈(축 B). 단일 원본=content_tools/pinokio_shared/cuda_dll.py
from cuda_dll import ensure_cuda_dlls
ensure_cuda_dlls()

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)

# yt-dlp의 YouTube JS 챌린지(nsig)용 deno 런타임이 설치돼 있으면 PATH에 추가(없으면 무시).
_deno_bin = os.path.expanduser("~/.deno/bin")
if os.path.isdir(_deno_bin) and _deno_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _deno_bin + os.pathsep + os.environ.get("PATH", "")

OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")  # 로컬 LLM 폴백(선택)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "")                    # 지정 시 우선, 비우면 설치된 모델 자동 선택
STT_MODEL       = os.getenv("STT_MODEL", "")                          # 비우면 GPU=large-v3 / CPU=small 자동

_executor = ThreadPoolExecutor(max_workers=2)
_lock = asyncio.Lock()              # GPU 직렬화
_jobs: dict = {}
_model_cache: dict = {}

# ── STT 결과 캐시 (같은 영상 재요청 시 다운로드·STT 건너뜀) ──
CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _ytid(url):
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})", url)
    return m.group(1) if m else None


def _file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(key):
    return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")


def _load_cache(key):
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(key, data):
    try:
        _cache_path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[캐시] 저장 실패: {e}", flush=True)


# ──────────────────────────────── STT ────────────────────────────────
def _get_model(size, device, compute):
    key = (size, device, compute)
    if key in _model_cache:
        return _model_cache[key]
    if device == "cuda":
        ensure_cuda_dlls()                       # 로드 직전 CUDA DLL 경로 재확인(임포트 순서 무관)
    from faster_whisper import WhisperModel
    print(f"[STT] 모델 로딩: {size} ({device}/{compute})", flush=True)
    m = WhisperModel(size, device=device, compute_type=compute)
    _model_cache[key] = m
    return m


def transcribe_segments(audio_path):
    """GPU(cuda, 큰 모델) 시도 → 실패 시 CPU(작은 모델) 폴백. 반환 (segments, lang)."""
    if STT_MODEL:
        attempts = [(STT_MODEL, "cuda", "float16"), (STT_MODEL, "cpu", "int8")]
    else:
        attempts = [("large-v3", "cuda", "float16"), ("small", "cpu", "int8")]
    last = None
    for size, device, compute in attempts:
        try:
            model = _get_model(size, device, compute)
            # word_timestamps=True → 각 세그먼트에 단어별 시각 포함(문장 단위 재분할·단어 하이라이트용)
            segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True,
                                              word_timestamps=True)
            total = getattr(info, "duration", 0) or 0
            print(f"[STT] 받아쓰기 시작 ({device}, 원어 {info.language}, 길이 {total:.0f}초)", flush=True)
            out = []
            next_log = 30.0                          # 30초 분량마다 진행률 출력(transcribe는 lazy 생성)
            for s in segments:
                t = s.text.strip()
                if t:
                    item = {"start": round(s.start, 3), "end": round(s.end, 3), "text": t}
                    ws = getattr(s, "words", None) or []
                    item["words"] = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                                     for w in ws if w.start is not None and w.end is not None]
                    out.append(item)
                if s.end >= next_log:
                    pct = f" ({100 * s.end / total:.0f}%)" if total else ""
                    print(f"[STT] 받아쓰기 중… {s.end:.0f}/{total:.0f}초{pct}", flush=True)
                    next_log = s.end + 30.0
            print(f"[STT] 완료: {len(out)}세그먼트 ({device}, 원어 {info.language})", flush=True)
            return out, info.language
        except Exception as e:
            print(f"[STT] {device} 실패 → 다음 시도: {type(e).__name__}: {e}", flush=True)
            last = e
    raise last


def download_audio(url):
    """yt-dlp로 오디오만 다운로드(후처리 없음 → ffmpeg 불필요). 반환 (path, title).
    간헐적 403 대응: ① 자동 재시도 + ② 플레이어 클라이언트 폴백(기본→web_safari→tv→mweb)."""
    import yt_dlp
    base = {"format": "bestaudio/best", "outtmpl": str(WORK_DIR / "%(id)s.%(ext)s"),
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "retries": 5, "fragment_retries": 5, "extractor_retries": 3}   # ① 재시도
    client_variants = [None, ["web_safari"], ["tv"], ["mweb"]]             # ② 클라이언트 폴백
    last = None
    for i, clients in enumerate(client_variants):
        opts = dict(base)
        if clients:
            opts["extractor_args"] = {"youtube": {"player_client": clients}}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if i:
                    print(f"[yt-dlp] 클라이언트 폴백 성공: {clients}", flush=True)
                return Path(ydl.prepare_filename(info)), info.get("title", "video")
        except Exception as e:
            last = e
            print(f"[yt-dlp] 다운로드 실패(client={clients or 'default'}): "
                  f"{type(e).__name__}: {e} → 다음 클라이언트 시도", flush=True)
    raise last if last else RuntimeError("오디오 다운로드 실패(모든 클라이언트 시도)")


# ─────────────────────────────── 번역 ───────────────────────────────
async def _deepl(texts, key):
    endpoint = ("https://api-free.deepl.com/v2/translate" if key.endswith(":fx")
                else "https://api.deepl.com/v2/translate")
    headers = {"Authorization": f"DeepL-Auth-Key {key}"}
    out = []
    async with httpx.AsyncClient() as c:
        for i in range(0, len(texts), 50):                 # DeepL 배치 최대 50, 순서 보존
            r = await c.post(endpoint, headers=headers,
                             json={"text": texts[i:i + 50], "target_lang": "KO"}, timeout=60)
            r.raise_for_status()
            out.extend(t["text"] for t in r.json()["translations"])
    return out


async def _local_llm(texts, model):
    """로컬 Ollama 폴백. 40줄 배치 1:1 번역, 실패 줄은 원문 유지."""
    out = list(texts)
    async with httpx.AsyncClient() as c:
        for i in range(0, len(texts), 40):
            batch = texts[i:i + 40]
            numbered = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(batch))
            prompt = ("다음 영상 자막을 자연스러운 한국어로 번역하세요. "
                      "'번호. 번역문' 형식으로 줄 수를 그대로 유지하고 번역문만 출력.\n\n" + numbered)
            try:
                r = await c.post(f"{OLLAMA_HOST}/api/generate",
                                 json={"model": model, "prompt": prompt, "stream": False,
                                       "options": {"temperature": 0.2}}, timeout=600)
                r.raise_for_status()
                resp = r.json().get("response", "")
            except Exception as e:
                print(f"[번역] 로컬 LLM 실패(원문 유지): {e}", flush=True)
                continue
            for line in resp.splitlines():
                mt = re.match(r"\s*(\d+)\s*[.)]\s*(.+)", line)
                if mt:
                    idx = int(mt.group(1)) - 1
                    if 0 <= idx < len(batch):
                        out[i + idx] = mt.group(2).strip()
    return out


async def _ollama_models():
    """Ollama에 설치된 모델 이름 목록(꺼져있거나 없으면 [])."""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _auto_model(models):
    """모델 목록에서 번역에 무난한 것 자동 선택(LOCAL_LLM_MODEL 지정 시 우선)."""
    if not models:
        return None
    if LOCAL_LLM_MODEL and LOCAL_LLM_MODEL in models:
        return LOCAL_LLM_MODEL
    for pref in ("exaone", "gemma", "qwen", "llama", "mistral", "phi"):
        for m in models:
            if m.lower().startswith(pref):
                return m
    return models[0]


# ── 로컬 번역 모델(CTranslate2 NMT) — faster-whisper와 같은 엔진(GPU/CPU, 오프라인·무료) ──
_CT2_REGISTRY = {
    "nllb-600m": {"hf": "facebook/nllb-200-distilled-600M", "arch": "nllb",   "label": "NLLB-600M (가벼움)"},
    "nllb-1.3b": {"hf": "facebook/nllb-200-distilled-1.3B", "arch": "nllb",   "label": "NLLB-1.3B"},
    "nllb-3.3b": {"hf": "facebook/nllb-200-3.3B",           "arch": "nllb",   "label": "NLLB-3.3B (고품질·대용량)"},
    "madlad-3b": {"hf": "google/madlad400-3b-mt",           "arch": "madlad", "label": "MADLAD-400-3B (Google)"},
}
LOCAL_MODELS = [m.strip() for m in os.getenv(
    "LOCAL_TRANSLATE_MODELS", "nllb-600m,nllb-1.3b,nllb-3.3b,madlad-3b").split(",")
    if m.strip() in _CT2_REGISTRY]
LOCAL_DEFAULT = os.getenv("LOCAL_TRANSLATE_DEFAULT", LOCAL_MODELS[0] if LOCAL_MODELS else "nllb-600m")
CT2_DIR_BASE  = os.getenv("CT2_DIR_BASE", str(APP_DIR / "models" / "ct2"))
NLLB_QUANT    = os.getenv("NLLB_QUANT", "int8_float16")    # GPU: int8_float16 / CPU: int8

_NLLB_LANG = {
    "en": "eng_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans", "es": "spa_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "ru": "rus_Cyrl", "it": "ita_Latn",
    "pt": "por_Latn", "vi": "vie_Latn", "th": "tha_Thai", "id": "ind_Latn",
    "ar": "arb_Arab", "hi": "hin_Deva", "tr": "tur_Latn", "pl": "pol_Latn",
    "nl": "nld_Latn", "uk": "ukr_Cyrl", "ko": "kor_Hang",
}
_NLLB_TGT = "kor_Hang"
_MADLAD_TGT = "<2ko>"
_ct2_cache = {}     # model_id → {"translator","tokenizer","arch"}
_ct2_err = {}


def _nllb_src(lang):
    return _NLLB_LANG.get((lang or "en").lower().split("-")[0], "eng_Latn")


def _ct2_device():
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _ct2_deps_ok():
    try:
        import ctranslate2  # noqa
        import transformers  # noqa
        return True
    except Exception:
        return False


def _ct2_get(model_id):
    """model_id의 CT2 번역기 lazy 로드/캐시(없으면 최초 1회 변환). 성공 dict / 실패 None."""
    if model_id in _ct2_cache:
        return _ct2_cache[model_id]
    spec = _CT2_REGISTRY.get(model_id)
    if not spec:
        return None
    try:
        import ctranslate2
        import transformers
        ensure_cuda_dlls()                       # CT2 GPU 추론용 CUDA DLL 경로 보장
        out = Path(CT2_DIR_BASE) / model_id
        if not (out / "model.bin").exists():
            out.mkdir(parents=True, exist_ok=True)
            from ctranslate2.converters import TransformersConverter
            print(f"[CT2] 변환 시작: {spec['hf']} → {out} (최초 1회, 모델 다운로드 포함)", flush=True)
            # load_as_float16: 샤딩 대형 모델(3.3B 등)의 meta-tensor 로딩 버그 회피 + 변환 메모리 절감
            TransformersConverter(spec["hf"], load_as_float16=True).convert(str(out), quantization=NLLB_QUANT, force=True)
            print(f"[CT2] 변환 완료: {model_id}", flush=True)
        device = _ct2_device()
        compute = NLLB_QUANT if device == "cuda" else "int8"
        entry = {
            "translator": ctranslate2.Translator(str(out), device=device, compute_type=compute),
            "tokenizer":  transformers.AutoTokenizer.from_pretrained(spec["hf"]),
            "arch": spec["arch"],
        }
        _ct2_cache[model_id] = entry
        _ct2_err.pop(model_id, None)
        print(f"[CT2] 로드 완료: {model_id} (device={device})", flush=True)
        return entry
    except Exception as e:
        _ct2_err[model_id] = f"{type(e).__name__}: {e}"
        print(f"[CT2] {model_id} 사용 불가 → {_ct2_err[model_id]}", flush=True)
        return None


def _ct2_translate_sync(model_id, texts, detected):
    """CTranslate2로 배치 번역(동기). nllb/madlad 아키텍처 모두 지원."""
    entry = _ct2_cache[model_id]
    tok, tr, arch = entry["tokenizer"], entry["translator"], entry["arch"]
    if arch == "madlad":
        sources = [tok.convert_ids_to_tokens(tok.encode(f"{_MADLAD_TGT} {t}")) for t in texts]
        prefix = None
    else:
        tok.src_lang = _nllb_src(detected)
        sources = [tok.convert_ids_to_tokens(tok.encode(t)) for t in texts]
        prefix = [[_NLLB_TGT]] * len(sources)
    results = tr.translate_batch(sources, batch_type="tokens", max_batch_size=2048, target_prefix=prefix)
    out = []
    for src_text, r in zip(texts, results):
        hyp = r.hypotheses[0] if r.hypotheses else []
        if arch != "madlad" and hyp and hyp[0] == _NLLB_TGT:
            hyp = hyp[1:]
        decoded = tok.decode(tok.convert_tokens_to_ids(hyp)).strip()
        out.append(decoded or src_text)
    return out


def _local_models_info():
    deps = _ct2_deps_ok()
    items = []
    for mid in LOCAL_MODELS:
        spec = _CT2_REGISTRY[mid]
        ready = mid in _ct2_cache or (Path(CT2_DIR_BASE) / mid / "model.bin").exists()
        items.append({"id": mid, "label": spec["label"], "ready": bool(ready)})
    return {"deps": deps, "default": LOCAL_DEFAULT, "models": items}


# ── 자막 조각 병합 — 너무 짧은 STT 세그먼트를 문장/구 단위로 합쳐 번역 정확도를 높인다 ──
SUB_MERGE           = os.getenv("SUBTITLE_MERGE", "1") not in ("0", "false", "False")
SUB_MAX_CHARS       = int(os.getenv("SUBTITLE_MAX_CHARS", "80"))         # CJK(글자 조밀)
SUB_MAX_CHARS_LATIN = int(os.getenv("SUBTITLE_MAX_CHARS_LATIN", "190"))  # 라틴 등(공백분리) — 문장 보존
SUB_MAX_GAP         = float(os.getenv("SUBTITLE_MAX_GAP", "1.0"))
SUB_MAX_DUR         = float(os.getenv("SUBTITLE_MAX_DUR", "12.0"))       # 긴 문장 보존(8→12)
# 단어 사이 '쉼(pause)' 경계 — 문장부호가 없는 언어(일본어 등)에서 문장/구를 나누는 핵심 신호.
SUB_WORD_GAP        = float(os.getenv("SUBTITLE_WORD_GAP", "0.45"))
_SENT_END = ("。", "．", ".", "!", "?", "！", "？", "…", "」", "”", "』")


def _is_cjk_text(detected="", sample=""):
    """CJK(공백 미분리) 여부 — joiner·글자수 상한 결정용. detected 우선, 없으면 텍스트 문자비율."""
    if (detected or "").lower()[:2] in ("ja", "zh", "ko"):
        return True
    if not sample:
        return False
    cjk = sum(1 for c in sample if
              "぀" <= c <= "ヿ" or "一" <= c <= "鿿" or "㐀" <= c <= "䶿" or "가" <= c <= "힣")
    return cjk >= max(1, len(sample.replace(" ", ""))) * 0.2


def merge_segments(segments, detected=""):
    """짧은 STT 조각을 문장/구 단위로 병합(번역 정확도↑). SUBTITLE_MERGE=0이면 비활성."""
    if not SUB_MERGE or len(segments) < 2:
        return segments
    joiner = "" if (detected or "").lower()[:2] in ("ja", "zh", "ko") else " "
    out, cur = [], None
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "text": text}
            continue
        gap = s["start"] - cur["end"]
        boundary = (cur["text"].endswith(_SENT_END) or gap > SUB_MAX_GAP
                    or len(cur["text"]) + len(text) > SUB_MAX_CHARS
                    or s["end"] - cur["start"] > SUB_MAX_DUR)
        if boundary:
            out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": text}
        else:
            cur["text"] += joiner + text
            cur["end"] = s["end"]
    if cur:
        out.append(cur)
    return out


def sentence_segments(segments, detected=""):
    """단어 타임스탬프(words)가 있으면 '문장 단위'로 재분할(시작/끝 시각 정확). 없으면 merge_segments 폴백.
    Whisper 원본 세그먼트는 문장 경계와 무관하게 끊기므로, 단어로 평탄화 후 문장부호에서 자른다."""
    if not SUB_MERGE or not segments:
        return segments
    if not any(s.get("words") for s in segments):
        return merge_segments(segments, detected)     # 단어정보 없으면 기존 방식
    sample = " ".join((s.get("text") or "") for s in segments[:30])
    cjk = _is_cjk_text(detected, sample)
    max_chars = SUB_MAX_CHARS if cjk else SUB_MAX_CHARS_LATIN

    words = []
    for s in segments:
        ws = s.get("words") or []
        if ws:
            words.extend(w for w in ws if (w.get("word") or "").strip())
        else:
            t = (s.get("text") or "").strip()
            if t:
                words.append({"start": s["start"], "end": s["end"], "word": " " + t})
    if not words:
        return merge_segments(segments, detected)

    out, cur = [], []

    def _flush():
        if not cur:
            return
        text = "".join(w["word"] for w in cur).strip()   # 단어 토큰이 공백 포함 → 자연스럽게 복원
        if text:
            out.append({"start": round(cur[0]["start"], 3),
                        "end":   round(cur[-1]["end"], 3), "text": text})

    for w in words:
        # 단어 사이 큰 쉼 → 문장 경계(문장부호 없는 언어 대응). 쉼 앞까지 먼저 끊고 새 조각 시작.
        if cur and (w["start"] - cur[-1]["end"]) >= SUB_WORD_GAP:
            _flush()
            cur = []
        cur.append(w)
        ct = "".join(x["word"] for x in cur).strip()
        ends_sentence = ct.endswith(_SENT_END) and len(cur) >= 2   # 약어/소수점 단독 분할 방지
        if ends_sentence or len(ct) >= max_chars or (w["end"] - cur[0]["start"]) >= SUB_MAX_DUR:
            _flush()
            cur = []
    _flush()
    return out


# ── 텍스트 기반 언어 감지 (lingua) — Whisper 오디오감지 오판(ja→ko 등) 보정 ──
_LINGUA_LANG_NAMES = ("ENGLISH", "CHINESE", "SPANISH", "HINDI", "ARABIC", "PORTUGUESE",
                      "RUSSIAN", "JAPANESE", "GERMAN", "FRENCH", "KOREAN")
_lingua_detector = None
_lingua_off = False


def _get_lingua():
    """lingua 감지기 싱글톤. 미설치/실패 시 None(→ Whisper 감지로 폴백)."""
    global _lingua_detector, _lingua_off
    if _lingua_detector is not None or _lingua_off:
        return _lingua_detector
    try:
        from lingua import Language, LanguageDetectorBuilder
        langs = [getattr(Language, n) for n in _LINGUA_LANG_NAMES]
        _lingua_detector = LanguageDetectorBuilder.from_languages(*langs).build()
        print(f"[번역] lingua 언어감지 로드 ({len(langs)}개 언어)", flush=True)
    except Exception as e:
        _lingua_off = True
        print(f"[번역] lingua 미사용 → Whisper 감지로 폴백: {type(e).__name__}: {e}", flush=True)
    return _lingua_detector


def _detect_lang_text(texts, fallback=""):
    """자막 텍스트로 소스 언어(ISO 639-1, 예 'ja')를 감지. 실패 시 fallback(Whisper 감지값)."""
    det = _get_lingua()
    if not det:
        return fallback
    sample = " ".join(t for t in texts if t).strip()[:3000]
    if not sample:
        return fallback
    try:
        lang = det.detect_language_of(sample)
        return lang.iso_code_639_1.name.lower() if lang else fallback
    except Exception:
        return fallback


async def translate(texts, lang, key, model=""):
    """반환 (번역결과, 상태: translated | source_korean | deepl_quota | no_engine).
    DeepL 키 있으면 최우선(품질 최상) → 실패/한도소진(456) 시 선택 로컬 모델로 폴백.
    키 없으면 바로 선택 로컬 모델. model 빈 값=서버 기본값. 'ollama:<명>'이면 Ollama LLM."""
    # 소스 언어를 '받아쓴 텍스트' 기반(lingua)으로 재확정 — Whisper 오디오감지 오판 보정.
    text_lang = _detect_lang_text(texts, fallback=(lang or ""))
    if text_lang and text_lang != (lang or "").lower().split("-")[0]:
        print(f"[번역] 소스언어 보정: STT감지='{lang}' → 텍스트감지='{text_lang}'", flush=True)
    lang = text_lang or lang

    if lang and lang.lower().startswith("ko"):
        return texts, "source_korean"
    loop = asyncio.get_event_loop()
    quota = False

    # 1) DeepL — 요청 키 있으면 최우선
    if key:
        try:
            return await _deepl(texts, key), "translated"
        except httpx.HTTPStatusError as e:
            quota = e.response.status_code in (429, 456)
            print(f"[번역] DeepL {e.response.status_code} → 로컬 폴백" + (" (한도 소진)" if quota else ""), flush=True)
        except Exception as e:
            print(f"[번역] DeepL 실패 → 로컬 폴백: {type(e).__name__}: {e}", flush=True)

    model = model or LOCAL_DEFAULT
    ok = lambda out: (out, "deepl_quota" if quota else "translated")

    # 2a) Ollama LLM
    ollama_name = model[7:] if model.startswith("ollama:") else (model if model not in _CT2_REGISTRY else "")
    if ollama_name:
        installed = await _ollama_models()
        chosen = ollama_name if ollama_name in installed else _auto_model(installed)
        if chosen:
            print(f"[번역] 로컬 Ollama 모델 사용: {chosen}", flush=True)
            return ok(await _local_llm(texts, chosen))
        model = LOCAL_DEFAULT if LOCAL_DEFAULT in _CT2_REGISTRY else "nllb-600m"

    # 2b) CT2 NMT (NLLB / MADLAD)
    if model not in _CT2_REGISTRY:
        model = LOCAL_DEFAULT if LOCAL_DEFAULT in _CT2_REGISTRY else "nllb-600m"
    if await loop.run_in_executor(_executor, _ct2_get, model) is None:
        return texts, "no_engine"
    try:
        out = await loop.run_in_executor(_executor, _ct2_translate_sync, model, texts, lang)
        return ok(out)
    except Exception as e:
        print(f"[번역] CT2({model}) 실패: {type(e).__name__}: {e}", flush=True)
        return texts, "no_engine"


# ─────────────────────────────── 잡 ───────────────────────────────
async def run_job(job_id, *, url=None, file_path=None, title="video", key="", model=""):
    job = _jobs[job_id]
    cleanup = []                       # 다운로드한 임시 오디오 정리용
    async with _lock:
        try:
            loop = asyncio.get_event_loop()
            # 캐시 키: 유튜브=영상ID, 로컬=파일 내용 해시
            if url:
                cache_key = "yt:" + (_ytid(url) or url)
            else:
                cache_key = "file:" + await loop.run_in_executor(_executor, _file_hash, file_path)

            cached = _load_cache(cache_key)
            from_cache = False
            if cached:                 # 이전 STT 결과 재사용 → 다운로드·STT 건너뜀
                segs = cached["segments"]
                lang = cached["lang"]
                title = cached.get("title", title)
                from_cache = True
                print(f"[잡] STT 캐시 재사용 {job_id}: {len(segs)}줄 ({title})", flush=True)
            else:
                if url:
                    job["status"] = "download"
                    audio, title = await loop.run_in_executor(_executor, download_audio, url)
                    cleanup.append(audio)
                else:
                    audio = Path(file_path)
                job["status"] = "stt"
                segs, lang = await loop.run_in_executor(_executor, transcribe_segments, audio)
                if not segs:
                    job.update(status="error", error="자막을 추출하지 못했어요 (STT 실패)")
                    return
                _save_cache(cache_key, {"title": title, "lang": lang, "segments": segs})
            job["title"] = title

            segs = sentence_segments(segs, lang)  # 단어 타임스탬프 기준 문장 단위 재분할
            job.update(status="translate", count=len(segs))
            originals = [s["text"] for s in segs]
            ko, tstatus = await translate(originals, lang, key, model)

            job["result"] = {
                "title": title, "source_lang": lang, "translate_status": tstatus, "cached": from_cache,
                "segments": [{"start": s["start"], "end": s["end"], "orig": o, "ko": k}
                             for s, o, k in zip(segs, originals, ko)],
            }
            job["status"] = "done"
            print(f"[잡] 완료 {job_id}: {len(segs)}줄 ({title}){' [캐시]' if from_cache else ''}", flush=True)
        except Exception as e:
            print(f"[잡] 오류 {job_id}: {type(e).__name__}: {e}", flush=True)
            job.update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            if file_path:              # 업로드 임시파일
                try:
                    Path(file_path).unlink()
                except OSError:
                    pass
            for p in cleanup:
                try:
                    p.unlink()
                except OSError:
                    pass


# ─────────────────────────────── API ───────────────────────────────
app = FastAPI(title="자막 학습 플레이어")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def cleanup_orphan_uploads(max_age_sec: int = 0) -> None:
    """WORK_DIR의 up_* 임시 업로드 정리. 0=전부(시작 시), >0=그보다 오래된 것만(새 잡 시작 시).
    정상 종료 시 잡 finally가 지우지만, 크래시로 남은 고아 파일을 청소한다."""
    if not WORK_DIR.exists():
        return
    now, removed = time.time(), 0
    for p in WORK_DIR.glob("up_*"):
        try:
            if max_age_sec and (now - p.stat().st_mtime) < max_age_sec:
                continue
            p.unlink(); removed += 1
        except OSError:
            pass
    if removed:
        print(f"[정리] 고아 업로드 파일 {removed}개 삭제", flush=True)


@app.post("/subtitle/jobs")
async def create_job(bg: BackgroundTasks,
                     url: Optional[str] = Form(default=None),
                     deepl_key: str = Form(default=""),
                     model: str = Form(default=""),
                     file: Optional[UploadFile] = File(default=None)):
    jid = uuid.uuid4().hex[:12]
    _jobs[jid] = {"status": "pending"}
    cleanup_orphan_uploads(2 * 3600)    # 2시간 넘게 남은 고아 업로드 정리(진행 중 파일은 보존)
    if file is not None and (file.filename or ""):
        ext = Path(file.filename).suffix or ".mp4"
        tmp = WORK_DIR / f"up_{jid}{ext}"
        tmp.write_bytes(await file.read())
        bg.add_task(run_job, jid, file_path=str(tmp), title=Path(file.filename).stem, key=deepl_key, model=model)
    elif url:
        bg.add_task(run_job, jid, url=url, key=deepl_key, model=model)
    else:
        _jobs.pop(jid, None)
        raise HTTPException(400, "url 또는 file이 필요합니다.")
    return {"job_id": jid}


@app.get("/subtitle/jobs/{jid}")
async def job_status(jid: str):
    j = _jobs.get(jid)
    if not j:
        raise HTTPException(404, "job을 찾을 수 없습니다.")
    return j


class RetranslateReq(BaseModel):
    texts: list[str]
    source_lang: str = ""
    deepl_key: str = ""
    model: str = ""


@app.post("/subtitle/translate")
async def retranslate(req: RetranslateReq):
    """이미 추출된 자막(원문)을 STT 없이 번역만 다시 한다 (키/모델 변경 후 재번역용)."""
    ko, status = await translate(req.texts, req.source_lang, req.deepl_key, req.model)
    return {"ko": ko, "translate_status": status}


@app.get("/translate/models")
async def list_translate_models():
    """플레이어 드롭다운용 — 노출 로컬 번역 모델 목록 + 기본값(환경변수로 설정)."""
    return _local_models_info()


# ── 단어/표현 뜻 조회 (학습용) — 번역엔진(DeepL 키 있으면 DeepL, 없으면 로컬 NLLB). 즉시 응답 ──
class LookupReq(BaseModel):
    text: str
    source_lang: str = ""
    deepl_key: str = ""         # 있으면 DeepL(단어도 정확), 없으면 로컬 모델


@app.post("/lookup")
async def lookup(req: LookupReq):
    text = (req.text or "").strip()[:120]
    if not text:
        return {"ko": ""}
    ko, _ = await translate([text], req.source_lang, req.deepl_key, "")
    return {"ko": ko[0] if ko else ""}


@app.get("/", response_class=HTMLResponse)
@app.get("/player", response_class=HTMLResponse)
async def player():
    return HTMLResponse((APP_DIR / "subtitle_player.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    # 외부 브라우저로 열도록 눈에 띄게 안내(YouTube 임베드는 일반 브라우저에서만 정상)
    print("\n" + "=" * 60, flush=True)
    print("  🌐 외부 브라우저(Chrome/Edge)에서 아래 주소를 여세요:", flush=True)
    print(f"     {url}", flush=True)
    print("=" * 60 + "\n", flush=True)
    # Pinokio가 이 URL을 캡처해 "플레이어 열기" 버튼을 만든다(이 줄은 형식 유지 필요)
    print(f"Server running on {url}", flush=True)
    cleanup_orphan_uploads()   # 시작 시 이전 크래시로 남은 업로드 고아 정리
    # access_log=False: 브라우저의 잡 상태 폴링(GET /subtitle/jobs/...) 접속 로그 노이즈 제거 → [STT]/[잡] 로그만 보이게
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
