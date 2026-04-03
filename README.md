# 🇰🇷 Kanana Korean Embedding Server

카카오의 [Kanana-Nano-2.1b-Embedding](https://huggingface.co/kakaocorp/kanana-nano-2.1b-embedding) 모델을 로컬에서 서빙하는 경량 FastAPI 서버입니다.

## 왜 이걸 만들었나?

한국어 임베딩이 필요한데, 기존 영어 중심 모델(`nomic-embed-text` 등)로는 한국어 검색 품질이 부족했습니다.

Kanana-Nano-2.1b-Embedding은 **한국어 임베딩 벤치마크 1위** (MTEB Korean subset 65.0)이면서 2.1B 파라미터로 로컬에서 충분히 돌릴 수 있는 크기입니다.

## 특징

- 🍎 **Apple Silicon MPS 지원** — M1/M2/M3/M4 Mac에서 GPU 가속
- 🔥 **CUDA 지원** — NVIDIA GPU 자동 감지
- 🔌 **Ollama 호환 API** — Ollama를 사용하는 도구에 바로 연결 가능
- 🪶 **경량** — FastAPI + uvicorn, 의존성 최소화
- 📏 **1792차원** 고품질 임베딩

## 빠른 시작

### 1. 모델 다운로드

```bash
# HuggingFace에서 모델 다운로드 (약 4.2GB)
git lfs install
git clone https://huggingface.co/kakaocorp/kanana-nano-2.1b-embedding ./model

# 또는 huggingface-cli 사용
pip install huggingface_hub
huggingface-cli download kakaocorp/kanana-nano-2.1b-embedding --local-dir ./model
```

### 2. 설치 & 실행

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
MODEL_DIR=./model python embed_server.py
```

서버가 `http://localhost:11435`에서 시작됩니다.

### 3. 테스트

```bash
# 한국어 임베딩 테스트
curl -s http://localhost:11435/embed \
  -H "Content-Type: application/json" \
  -d '{"input": ["안녕하세요", "한국어 임베딩 테스트입니다"]}' | python3 -m json.tool | head -5

# 헬스체크
curl http://localhost:11435/health
```

## API 엔드포인트

| 엔드포인트 | 방식 | 설명 |
|-----------|------|------|
| `POST /embed` | Native | 배열 입력 → 배열 임베딩 |
| `POST /api/embeddings` | Ollama 호환 | 단일 prompt → 단일 embedding |
| `POST /api/embed` | Ollama 호환 | 배열 input → 배열 embeddings |
| `GET /api/tags` | Ollama 호환 | 모델 목록 (자동 감지용) |
| `GET /health` | - | 상태 확인 |

### Native API 예시

```bash
curl -X POST http://localhost:11435/embed \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["서울 맛집 추천해주세요", "강남역 근처 한식당"],
    "instruction": "Given a question, retrieve passages that answer the question"
  }'
```

### Ollama 호환 API 예시

```bash
# /api/embeddings (Ollama v1 호환)
curl -X POST http://localhost:11435/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "kanana-nano-2.1b-embedding", "prompt": "한국어 테스트"}'

# /api/embed (Ollama v2 호환)
curl -X POST http://localhost:11435/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "kanana-nano-2.1b-embedding", "input": ["텍스트1", "텍스트2"]}'
```

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_DIR` | 스크립트 디렉토리 | 모델 파일 경로 |
| `PORT` | `11435` | 서버 포트 |

## macOS LaunchAgent (자동 시작)

Mac에서 부팅 시 자동 실행하려면:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.kanana.embedding</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/venv/bin/python</string>
    <string>/path/to/embed_server.py</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>MODEL_DIR</key>
    <string>/path/to/model</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/kanana-embedding.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/kanana-embedding.log</string>
</dict>
</plist>
```

`~/Library/LaunchAgents/ai.kanana.embedding.plist`에 저장 후:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.kanana.embedding.plist
```

## 시스템 요구사항

- Python 3.10+
- RAM: ~4GB (모델 로딩 시)
- 디스크: ~4.2GB (모델 파일)
- 권장: Apple Silicon Mac (MPS) 또는 NVIDIA GPU (CUDA)

## 모델 정보

- **모델**: [kakaocorp/kanana-nano-2.1b-embedding](https://huggingface.co/kakaocorp/kanana-nano-2.1b-embedding)
- **파라미터**: 2.1B
- **임베딩 차원**: 1792
- **최대 토큰**: 512 (서버 기본값) / 8192 (모델 한계)
- **한국어 MTEB**: 65.0 (같은 크기 모델 중 1위)
- **라이선스**: Apache 2.0

## 감사 (Acknowledgments)

이 프로젝트는 [카카오(Kakao)](https://www.kakaocorp.com/)의 [Kanana](https://huggingface.co/kakaocorp) 모델 패밀리 덕분에 가능했습니다.

카카오 팀이 고품질 한국어 임베딩 모델을 Apache 2.0 라이선스로 공개해주신 덕분에, 누구나 로컬에서 한국어 의미 검색을 구축할 수 있게 되었습니다. 🙏

> **Kanana-Nano-2.1b-Embedding** by [kakaocorp](https://huggingface.co/kakaocorp/kanana-nano-2.1b-embedding) — 한국어 임베딩 벤치마크 1위, Apache 2.0

## 라이선스

이 서버 코드는 MIT 라이선스입니다.
모델 자체는 [Kakao의 Apache 2.0 라이선스](https://huggingface.co/kakaocorp/kanana-nano-2.1b-embedding)를 따릅니다.
