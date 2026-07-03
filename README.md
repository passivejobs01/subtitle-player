# 자막 학습 플레이어 (Subtitle Player)

유튜브·로컬 영상에 **원어 + 한글 이중 자막**을 만들어 외국어를 공부하는 **로컬** 플레이어입니다.
음성을 받아쓰고(STT) 번역해 자막을 생성하고, 영상과 함께 재생합니다. — 잇츠매거진

## 무엇을 하나
- **유튜브 링크** 또는 **내 PC 영상 파일** → 자막 자동 생성
- **원어 + 한글 이중 자막**(표시 토글), 가독성(반투명 배경 + 외곽선)
- 학습 기능: **구간 반복(A·B) · 한 문장 반복 · 한 문장 멈춤 · 단어 클릭 뜻 팝업 · 한글 가리기 · 이전/다음 문장 · 재생 속도 · .srt 내보내기**
- **문장 단위 자막**: 단어 타임스탬프 기반으로 **문장 단위로 정확히 분할**(학습·반복에 최적)
- **자동 언어 감지**: 원어를 텍스트 기반으로 정확히 판별 → 올바른 번역
- **STT**: faster-whisper — GPU(NVIDIA) 있으면 `large-v3`, 없으면 CPU `small` 자동 폴백
- **번역**: **DeepL**(키 입력 시, 최고 품질) → 없으면 **로컬 NMT/LLM(Ollama)** 폴백

## 설치 (Pinokio)
Pinokio → **Discover / Download** → **"Download from URL"** 에 이 저장소 주소를 붙여넣고 설치하세요:
```
https://github.com/passivejobs01/subtitle-player
```

## 사용법
1. Pinokio에서 **Install** → **Start**
2. **플레이어 열기** 클릭 → 브라우저에 플레이어가 뜸
3. (선택) **DeepL 무료키** 입력 — [deepl.com/pro-api](https://www.deepl.com/pro-api) 발급(월 50만자 무료). 키는 브라우저에만 저장.
4. **유튜브 링크 붙여넣기 → 자막 생성**, 또는 **영상 파일 드래그**
5. 잠시 후 이중 자막이 영상과 함께 재생됨 (긴 영상은 STT에 시간 소요)

> 로컬 LLM 폴백을 쓰려면 [Ollama](https://ollama.com) 설치 + 아무 모델(예: `ollama pull gemma2:9b`)만 있으면 됩니다 — **설치된 모델을 자동 선택**합니다(번역 무난한 모델 우선). 특정 모델을 강제하려면 환경변수 `LOCAL_LLM_MODEL`(+`OLLAMA_HOST`). DeepL 키가 있으면 폴백은 필요 없습니다.

## API (프로그램에서 호출)
서버는 localhost에서 동작(포트는 Pinokio가 자동 할당, 예시는 `7860`).

**자막 생성 잡 시작 → 폴링 → 결과(JSON)**

### curl
```bash
# 유튜브
curl -s -X POST http://127.0.0.1:7860/subtitle/jobs -F "url=https://youtu.be/XXXX" -F "deepl_key=YOUR_KEY:fx"
# 로컬 파일
curl -s -X POST http://127.0.0.1:7860/subtitle/jobs -F "file=@video.mp4" -F "deepl_key=YOUR_KEY:fx"
# 상태/결과
curl -s http://127.0.0.1:7860/subtitle/jobs/<job_id>
```

### Python
```python
import requests, time
B = "http://127.0.0.1:7860"
jid = requests.post(f"{B}/subtitle/jobs",
        data={"url": "https://youtu.be/XXXX", "deepl_key": "YOUR_KEY:fx"}).json()["job_id"]
while True:
    j = requests.get(f"{B}/subtitle/jobs/{jid}").json()
    if j["status"] in ("done", "error"): break
    time.sleep(2)
print(j["result"]["segments"][:3])   # [{start, end, orig, ko}, ...]
```

### JavaScript
```javascript
const B = "http://127.0.0.1:7860";
const fd = new FormData();
fd.append("url", "https://youtu.be/XXXX");
fd.append("deepl_key", "YOUR_KEY:fx");
const { job_id } = await (await fetch(`${B}/subtitle/jobs`, { method: "POST", body: fd })).json();
let j;
do { await new Promise(r => setTimeout(r, 2000));
     j = await (await fetch(`${B}/subtitle/jobs/${job_id}`)).json(); }
while (!["done", "error"].includes(j.status));
console.log(j.result.segments);   // [{start, end, orig, ko}, ...]
```

**응답 결과 형식**
```json
{ "status": "done",
  "result": { "title": "...", "source_lang": "ja",
    "segments": [ { "start": 0.0, "end": 2.2, "orig": "原文", "ko": "번역" } ] } }
```

## 관리 (Pinokio)
- **Start / Update / Install / Reset** — 사이드바 메뉴
- **Reset** = 가상환경(`app/env`) 삭제 후 재설치 가능

## 라이선스 / 출처
잇츠매거진 · [YouTube](https://www.youtube.com/@its-magazine)
