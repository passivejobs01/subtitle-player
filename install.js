module.exports = {
  // AI 번들 미사용 — 의존성이 전부 prebuilt 휠이라 Visual C++ Build Tools / CUDA 툴킷 불필요.
  // GPU 가속은 pip CUDA 라이브러리(nvidia-cublas/cudnn)로 처리하므로 설치가 가볍다.
  run: [
    // 1) 파이썬 의존성 (faster-whisper는 av로 오디오 디코딩 → 시스템 ffmpeg 불필요)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // 2) NVIDIA GPU면 faster-whisper(ctranslate2)용 CUDA 라이브러리 설치 → GPU 가속 STT
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12"
      }
    },
    // 3) NLLB(무료 GPU 번역) 모델을 CTranslate2로 변환할 때 쓰는 CPU torch (가벼운 prebuilt 휠, 컴파일 없음)
    //    추론은 ctranslate2(GPU)로 하고, torch는 최초 1회 모델 변환에만 사용.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
      }
    },
    {
      method: "input",
      params: {
        title: "설치 완료!",
        description: "Start를 눌러 자막 학습 플레이어를 실행하세요. (GPU 없으면 CPU로 동작 · NLLB는 처음 한 번 모델을 내려받습니다)"
      }
    }
  ]
}
