"""Pinokio 앱 공통 — Windows CUDA DLL 부트스트랩 (축 B: Pinokio 전용 공통 코드).

faster-whisper(ctranslate2)·Qwen3-TTS(torch) 등이 venv 안 nvidia/*/bin DLL을 찾도록,
의존성 순서(cudart → cublasLt → cublas → cudnn)대로 전체경로를 ctypes로 preload 한다.
os.add_dll_directory만으로는 ctranslate2 내부 LoadLibrary가 user-dir를 안 봐서 cublas를 못 찾는
("cublas64_12.dll cannot be loaded") 문제를 해결한다.

※ 이 모듈은 Pinokio(로컬·Windows) 앱 전용이다. FastAPI(리눅스) 서버에는 이 문제가 없어 사용하지 않는다.
※ 단일 원본: content_tools/pinokio_shared/cuda_dll.py — 수정 후 `bash content_tools/sync_pinokio.sh`로
   각 Pinokio 앱(app/cuda_dll.py)에 동일 반영. 앱 쪽 복사본은 직접 수정 금지.
"""
import os
import sys
from pathlib import Path

_dll_handles = []   # add_dll_directory / WinDLL 핸들 유지(참조 보존)


def ensure_cuda_dlls():
    """venv의 nvidia CUDA DLL을 로드. Windows 전용, 여러 번 호출해도 안전(최초 1회만 수행).

    GPU 모델 로드/추론 직전에 호출하면 임포트 순서와 무관하게 cublas/cudnn이 해결된다.
    """
    if sys.platform != "win32" or getattr(ensure_cuda_dlls, "_done", False):
        return
    try:
        import importlib.util
        import ctypes
        spec = importlib.util.find_spec("nvidia")
        if not (spec and spec.submodule_search_locations):
            return
        bins = []
        for base in spec.submodule_search_locations:
            for binp in Path(base).glob("*/bin"):          # cublas/bin, cudnn/bin, cuda_runtime/bin …
                if binp.is_dir():
                    _dll_handles.append(os.add_dll_directory(str(binp)))
                    bins.append(binp)
        # 의존성 순서대로 전체경로 preload (cudart가 cublas의 의존이므로 반드시 먼저)
        for pattern in ("cudart64_*.dll", "cublasLt64_*.dll", "cublas64_*.dll", "cudnn*.dll"):
            for binp in bins:
                for dll in sorted(binp.glob(pattern)):
                    try:
                        _dll_handles.append(ctypes.WinDLL(str(dll)))
                    except OSError:
                        pass
        ensure_cuda_dlls._done = True
    except Exception:
        pass
