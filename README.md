# LoRA Fine-tuning Demo Project

GPT-2 Small 모델을 LoRA(Low-Rank Adaptation)를 통해 효율적으로 파인튜닝하는 데모 프로젝트입니다.

## 🎯 프로젝트 개요

- **모델**: GPT-2 Small (124M parameters)
- **방법**: LoRA (Parameter-Efficient Fine-Tuning)
- **환경**: RTX 3070 (8GB VRAM)
- **예상 학습 시간**: 30-45분 (10K 샘플, 3 epochs)

## 📁 프로젝트 구조

```
lora/
├── README.md                    # 프로젝트 설명서
├── PROJECT_PLAN.md             # 상세 설계 문서
├── EVALUATION_STRATEGY.md      # 평가 전략
├── requirements.txt            # Python 의존성
├── Dockerfile                  # Docker 이미지 정의
├── config/
│   ├── model_config.yaml      # 모델 설정
│   └── training_config.yaml   # 학습 설정
├── data/
│   ├── raw/                   # 원본 데이터
│   ├── processed/             # 전처리된 데이터
│   └── prepare_dataset.py     # 데이터 전처리 스크립트
├── src/
│   ├── __init__.py
│   ├── model.py               # 모델 로딩 및 LoRA 설정
│   ├── dataset.py             # 데이터셋 클래스
│   ├── trainer.py             # 학습 로직
│   └── utils.py               # 유틸리티 함수
├── scripts/
│   ├── train.py               # 학습 실행
│   ├── inference.py           # 추론 테스트
│   ├── evaluate.py            # 평가 스크립트
│   ├── compare_results.py     # 결과 비교
│   └── merge_lora.py          # LoRA 가중치 병합
├── notebooks/
│   └── demo.ipynb             # 데모 노트북
├── outputs/
│   ├── checkpoints/           # 학습 체크포인트
│   ├── logs/                  # 학습 로그
│   ├── eval/                  # 평가 결과
│   └── merged_models/         # 병합된 모델
└── tests/
    └── test_model.py          # 단위 테스트
```

## 🚀 빠른 시작

### 1. Docker 환경 설정

#### Docker 이미지 빌드
```bash
docker build -t lora-training .
```

#### Docker 컨테이너 실행
```bash
docker run -it --gpus all \
    -v $(pwd):/workspace \
    --name lora_demo \
    lora-training
```

### 2. 로컬 환경 설정 (대안)

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 3. 데이터 준비

```bash
# 샘플 데이터 다운로드 및 전처리
python data/prepare_dataset.py --dataset koalpaca --num_samples 10000
```

### 4. 학습 실행

```bash
# Baseline 평가 (학습 전)
python scripts/evaluate.py --mode baseline

# LoRA 학습
python scripts/train.py

# Fine-tuned 평가 (학습 후)
python scripts/evaluate.py --mode finetuned

# 결과 비교
python scripts/compare_results.py
```

### 5. 추론 테스트

```bash
python scripts/inference.py \
    --base_model gpt2 \
    --lora_weights outputs/checkpoints/final \
    --prompt "한국의 수도는 어디인가요?"
```

## ⚙️ 설정 커스터마이징

### 모델 설정 (`config/model_config.yaml`)
- LoRA rank, alpha, dropout
- Target modules
- Generation parameters

### 학습 설정 (`config/training_config.yaml`)
- Batch size, learning rate
- Epochs, warmup steps
- Logging, checkpointing

## 📊 평가 메트릭

- **Perplexity**: 언어 모델 성능 (낮을수록 좋음)
- **BLEU Score**: 생성 품질 (높을수록 좋음)
- **ROUGE Score**: 요약 품질
- **Human Evaluation**: 정성적 평가

## 🎯 성공 기준

- ✅ Perplexity 10-20% 감소
- ✅ BLEU Score 5-10점 증가
- ✅ 샘플 생성 품질 향상

## 📝 주요 명령어

```bash
# GPU 확인
nvidia-smi

# 학습 모니터링
tensorboard --logdir outputs/logs

# 체크포인트 확인
ls -lh outputs/checkpoints/

# LoRA 가중치 병합
python scripts/merge_lora.py \
    --base_model gpt2 \
    --lora_weights outputs/checkpoints/final \
    --output_dir outputs/merged_models/gpt2-lora
```

## 🐛 문제 해결

### OOM (Out of Memory) 에러
```yaml
# training_config.yaml 수정
batch_size: 1
gradient_accumulation_steps: 8  # 16에서 감소
max_seq_length: 256  # 512에서 감소
```

### 학습 불안정
```yaml
# training_config.yaml 수정
learning_rate: 1.0e-4  # 2.0e-4에서 감소
warmup_steps: 200  # 100에서 증가
```

## 📚 참고 자료

- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [GPT-2 모델](https://huggingface.co/gpt2)

## 📄 라이선스

MIT License

## 🤝 기여

이슈 및 PR 환영합니다!

## 📧 문의

프로젝트 관련 문의사항은 이슈로 남겨주세요.
