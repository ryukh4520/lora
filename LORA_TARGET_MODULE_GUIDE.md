# LoRA Target Module 선정 완전 가이드

## 🎯 목차
1. [모델 구조 이해하기](#모델-구조-이해하기)
2. [Target Module 찾기](#target-module-찾기)
3. [선정 기준과 전략](#선정-기준과-전략)
4. [실전 선정 프로세스](#실전-선정-프로세스)
5. [모델별 권장 설정](#모델별-권장-설정)

---

## 🏗️ 1. 모델 구조 이해하기

### 1.1 Transformer 기본 구조

```
┌─────────────────────────────────────────────────────────┐
│                   Transformer Layer                      │
└─────────────────────────────────────────────────────────┘

Input Embedding
    ↓
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Query Projection (Q)          │ │ ← LoRA 적용 가능
│  │ W_q: (d_model, d_model)       │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Key Projection (K)            │ │ ← LoRA 적용 가능
│  │ W_k: (d_model, d_model)       │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Value Projection (V)          │ │ ← LoRA 적용 가능
│  │ W_v: (d_model, d_model)       │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Attention Computation         │ │
│  │ Softmax(QK^T/√d)V            │ │ (LoRA 적용 안함)
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Output Projection (O)         │ │ ← LoRA 적용 가능
│  │ W_o: (d_model, d_model)       │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Feed-Forward Network (FFN)         │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Up Projection                 │ │ ← LoRA 적용 가능
│  │ W_up: (d_model, d_ff)         │ │
│  │ (예: 768 → 3072)              │ │
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Activation (GELU, ReLU)       │ │ (LoRA 적용 안함)
│  └───────────────────────────────┘ │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Down Projection               │ │ ← LoRA 적용 가능
│  │ W_down: (d_ff, d_model)       │ │
│  │ (예: 3072 → 768)              │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓
Output
```

**LoRA 적용 가능 레이어**:
```
✅ Linear layers (nn.Linear)
❌ Activation functions
❌ LayerNorm
❌ Dropout
❌ Embedding layers (선택적)
```

---

### 1.2 왜 Linear Layer만 적용하는가?

```python
# Linear layer의 특징

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = Parameter(torch.randn(out_features, in_features))
        self.bias = Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias
        #      ↑
        # 행렬 곱 → LoRA 적용 가능!

# LoRA 적용
output = x @ W.T + (x @ A.T) @ B.T * (alpha/r)
         ↑        ↑
      원래 가중치  LoRA 추가
```

**이유**:
```
1. 행렬 곱 연산
   → Low-rank decomposition 가능

2. 대부분의 파라미터
   → Transformer의 90%+ 파라미터가 Linear

3. 학습 효과
   → Linear layer가 표현력의 핵심
```

---

## 🔍 2. Target Module 찾기

### 2.1 모델 구조 탐색

#### **Step 1: 모델 로드**

```python
from transformers import AutoModelForCausalLM

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("gpt2")

print(model)
```

**출력 예시 (GPT-2)**:
```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,))
        (attn): GPT2Attention(
          (c_attn): Conv1D()      ← 이것!
          (c_proj): Conv1D()      ← 이것!
          (attn_dropout): Dropout(p=0.1)
          (resid_dropout): Dropout(p=0.1)
        )
        (ln_2): LayerNorm((768,))
        (mlp): GPT2MLP(
          (c_fc): Conv1D()        ← 이것!
          (c_proj): Conv1D()      ← 이것!
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1)
        )
      )
    )
    (ln_f): LayerNorm((768,))
  )
  (lm_head): Linear(in_features=768, out_features=50257)
)
```

---

#### **Step 2: Linear Layer 찾기**

```python
def find_linear_layers(model):
    """모델의 모든 Linear 레이어 찾기"""
    linear_layers = {}
    
    for name, module in model.named_modules():
        # Linear 또는 Conv1D (GPT-2의 경우)
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1D)):
            # 크기 정보
            if hasattr(module, 'weight'):
                shape = module.weight.shape
                linear_layers[name] = {
                    'type': module.__class__.__name__,
                    'shape': shape,
                    'params': shape[0] * shape[1]
                }
    
    return linear_layers

# 사용
layers = find_linear_layers(model)
for name, info in layers.items():
    print(f"{name}: {info['shape']} ({info['params']:,} params)")
```

**출력 예시 (GPT-2)**:
```
transformer.h.0.attn.c_attn: (2304, 768) (1,769,472 params)
transformer.h.0.attn.c_proj: (768, 768) (589,824 params)
transformer.h.0.mlp.c_fc: (3072, 768) (2,359,296 params)
transformer.h.0.mlp.c_proj: (768, 3072) (2,359,296 params)
...
(12 layers 반복)
lm_head: (50257, 768) (38,597,376 params)
```

---

#### **Step 3: 모듈 이름 패턴 파악**

```python
def analyze_module_patterns(model):
    """모듈 이름 패턴 분석"""
    patterns = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1D)):
            # 마지막 부분 추출 (예: "c_attn")
            module_name = name.split('.')[-1]
            
            if module_name not in patterns:
                patterns[module_name] = []
            patterns[module_name].append(name)
    
    return patterns

# 사용
patterns = analyze_module_patterns(model)
for pattern, occurrences in patterns.items():
    print(f"\n{pattern}: {len(occurrences)}개")
    print(f"  예시: {occurrences[0]}")
```

**출력 예시 (GPT-2)**:
```
c_attn: 12개
  예시: transformer.h.0.attn.c_attn

c_proj: 24개 (attn 12개 + mlp 12개)
  예시: transformer.h.0.attn.c_proj
  예시: transformer.h.0.mlp.c_proj

c_fc: 12개
  예시: transformer.h.0.mlp.c_fc

lm_head: 1개
  예시: lm_head
```

---

### 2.2 모델별 모듈 이름

#### **GPT-2**

```python
# Attention
"c_attn"   # QKV projection (통합)
"c_proj"   # Output projection

# MLP
"c_fc"     # Up projection
"c_proj"   # Down projection (이름 중복!)

# 주의: c_proj가 attn과 mlp에 모두 있음!
```

---

#### **LLaMA / Mistral**

```python
# Attention
"q_proj"   # Query projection
"k_proj"   # Key projection
"v_proj"   # Value projection
"o_proj"   # Output projection

# MLP
"gate_proj"  # Gate projection
"up_proj"    # Up projection
"down_proj"  # Down projection
```

---

#### **BERT**

```python
# Attention
"query"    # Query projection
"key"      # Key projection
"value"    # Value projection

# Output
"dense"    # Output projection (여러 곳에 있음)

# MLP
"intermediate.dense"  # Up projection
"output.dense"        # Down projection
```

---

## 📋 3. 선정 기준과 전략

### 3.1 선정 기준

#### **기준 1: 파라미터 수**

```python
# 파라미터가 많은 레이어 우선

GPT-2 예시:
c_attn:  1,769,472 params  ← 가장 큼
mlp.c_fc: 2,359,296 params ← 가장 큼
mlp.c_proj: 2,359,296 params
attn.c_proj: 589,824 params

→ 큰 레이어에 LoRA 적용 시 효과적
```

---

#### **기준 2: 태스크 관련성**

```python
# 태스크에 중요한 레이어 우선

QA, 분류:
→ Attention layers (Q, K, V, O)
→ 문맥 이해가 중요

생성, 요약:
→ Attention + MLP
→ 표현력이 중요

번역:
→ Attention layers
→ 정렬(alignment)이 중요
```

---

#### **기준 3: 메모리 제약**

```python
# 메모리에 따라 조절

4GB VRAM:
→ Attention만 (Q, V만 또는 Q, K, V, O)

8GB VRAM:
→ Attention 전체 (Q, K, V, O)

16GB+ VRAM:
→ Attention + MLP
```

---

### 3.2 선정 전략

#### **전략 1: Attention Only (기본, 권장)**

```python
# GPT-2
target_modules = ["c_attn", "c_proj"]

# LLaMA
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# BERT
target_modules = ["query", "key", "value"]

장점:
✅ 파라미터 효율적
✅ 대부분의 경우 충분
✅ 빠른 학습
✅ 메모리 절약

사용:
- 간단한 QA
- 분류
- 요약
- 우리 프로젝트 ✅
```

---

#### **전략 2: Attention + MLP (높은 성능)**

```python
# GPT-2
target_modules = ["c_attn", "c_proj", "c_fc"]

# LLaMA
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

장점:
✅ 높은 표현력
✅ 복잡한 태스크 대응

단점:
⚠️ 파라미터 2-3배 증가
⚠️ 메모리 사용 증가
⚠️ 학습 시간 증가

사용:
- 복잡한 생성
- 창의적 글쓰기
- 전문 도메인
```

---

#### **전략 3: Query + Value Only (효율적)**

```python
# LLaMA
target_modules = ["q_proj", "v_proj"]

장점:
✅ 파라미터 절약 (50%)
✅ 여전히 효과적

이론:
- Query: "무엇을 찾을까?"
- Value: "무엇을 반환할까?"
- Key는 상대적으로 덜 중요

사용:
- 메모리 제약
- 빠른 실험
```

---

## 🛠️ 4. 실전 선정 프로세스

### 4.1 단계별 프로세스

#### **Step 1: 모델 구조 파악**

```python
# 1. 모델 로드
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Linear 레이어 찾기
def print_linear_layers(model, max_display=20):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1D)):
            if hasattr(module, 'weight'):
                shape = module.weight.shape
                params = shape[0] * shape[1]
                print(f"{name}")
                print(f"  Shape: {shape}")
                print(f"  Params: {params:,}")
                print()
                count += 1
                if count >= max_display:
                    print(f"... (총 더 많은 레이어 있음)")
                    break

print_linear_layers(model)
```

---

#### **Step 2: 패턴 분석**

```python
# 3. 모듈 이름 패턴 추출
def extract_module_patterns(model):
    patterns = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1D)):
            # 마지막 부분만 추출
            module_name = name.split('.')[-1]
            patterns.add(module_name)
    return sorted(patterns)

patterns = extract_module_patterns(model)
print("사용 가능한 모듈 이름:")
for p in patterns:
    print(f"  - {p}")
```

**출력 (GPT-2)**:
```
사용 가능한 모듈 이름:
  - c_attn
  - c_fc
  - c_proj
  - lm_head
```

---

#### **Step 3: Attention 레이어 식별**

```python
# 4. Attention 관련 레이어 찾기
def find_attention_modules(model):
    attn_modules = set()
    for name, module in model.named_modules():
        if 'attn' in name.lower():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1D)):
                module_name = name.split('.')[-1]
                attn_modules.add(module_name)
    return sorted(attn_modules)

attn_modules = find_attention_modules(model)
print("Attention 모듈:")
for m in attn_modules:
    print(f"  - {m}")
```

**출력 (GPT-2)**:
```
Attention 모듈:
  - c_attn
  - c_proj
```

---

#### **Step 4: 초기 설정 (Attention만)**

```python
# 5. 기본 설정으로 시작
target_modules = ["c_attn", "c_proj"]  # GPT-2

# 또는
# target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLaMA
```

---

#### **Step 5: 테스트 학습**

```python
# 6. 짧은 학습으로 테스트
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 파라미터 확인
model.print_trainable_parameters()
```

**출력**:
```
trainable params: 811,008 || all params: 125,250,816 || trainable%: 0.6475
```

---

#### **Step 6: 성능 평가**

```python
# 7. 짧은 학습 (1-2 epochs)
# 성능 측정

if performance < target:
    # MLP 추가 고려
    target_modules = ["c_attn", "c_proj", "c_fc"]
    # 재학습 및 비교
```

---

### 4.2 검증 방법

```python
def verify_target_modules(model, target_modules):
    """Target modules가 실제로 존재하는지 확인"""
    
    # 모든 모듈 이름 수집
    all_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1D)):
            module_name = name.split('.')[-1]
            all_modules.add(module_name)
    
    # 검증
    print("검증 결과:")
    for target in target_modules:
        if target in all_modules:
            print(f"  ✅ {target}: 존재함")
        else:
            print(f"  ❌ {target}: 존재하지 않음!")
            print(f"     사용 가능: {all_modules}")
    
    # 적용될 레이어 수 계산
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1D)):
            module_name = name.split('.')[-1]
            if module_name in target_modules:
                count += 1
    
    print(f"\n총 {count}개 레이어에 LoRA 적용됨")

# 사용
verify_target_modules(model, ["c_attn", "c_proj"])
```

---

## 📚 5. 모델별 권장 설정

### 5.1 GPT-2 (우리 프로젝트)

```python
# 기본 (권장)
target_modules = ["c_attn", "c_proj"]

# 설명
c_attn:  QKV projection (768 → 2304)
c_proj:  Output projection (768 → 768)

# 파라미터
r=8 기준:
- c_attn: 24,576 params × 12 layers = 294,912
- c_proj: 12,288 params × 12 layers = 147,456
- 합계: 442,368 params

# 확장 (높은 성능)
target_modules = ["c_attn", "c_proj", "c_fc"]

c_fc: MLP up projection (768 → 3072)
추가 파라미터: ~300K
```

---

### 5.2 LLaMA / Mistral

```python
# 기본 (권장)
target_modules = [
    "q_proj",   # Query
    "k_proj",   # Key
    "v_proj",   # Value
    "o_proj"    # Output
]

# 효율적
target_modules = ["q_proj", "v_proj"]

# 확장 (높은 성능)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

---

### 5.3 BERT

```python
# 기본 (권장)
target_modules = ["query", "key", "value"]

# 확장
target_modules = ["query", "key", "value", "dense"]

# 주의: "dense"는 여러 곳에 있음
# 더 정확한 지정:
target_modules = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value"
]
```

---

## 🎯 우리 프로젝트 선정 과정

### 실제 수행한 단계

```python
# Step 1: 모델 확인
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: 구조 파악
# GPT-2는 c_attn, c_proj 사용 확인

# Step 3: 기본 설정 선택
target_modules = ["c_attn", "c_proj"]

# 이유:
# ✅ Attention 레이어만 (효율적)
# ✅ 간단한 QA 태스크
# ✅ 1,000 샘플 (작은 데이터)
# ✅ 8GB VRAM (충분)

# Step 4: 적용
lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["c_attn", "c_proj"]
}

# Step 5: 결과
# 파라미터: 811,008 (0.65%)
# Perplexity: 9.08 → 1.05
# 성공! ✅
```

---

## 💡 핵심 요약

### **선정 프로세스**

```
1. 모델 구조 파악
   → Linear 레이어 찾기

2. 모듈 이름 패턴 분석
   → 사용 가능한 이름 확인

3. Attention 레이어 식별
   → 기본 target 선정

4. 초기 설정 (Attention만)
   → 테스트 학습

5. 성능 평가
   → 필요시 MLP 추가

6. 최종 선택
   → 성능/비용 균형
```

---

### **권장 설정**

```
기본 (대부분의 경우):
→ Attention 레이어만
→ GPT-2: ["c_attn", "c_proj"]
→ LLaMA: ["q_proj", "k_proj", "v_proj", "o_proj"]

확장 (높은 성능 필요):
→ Attention + MLP
→ 파라미터 2-3배 증가

효율적 (메모리 제약):
→ Query + Value만
→ 파라미터 50% 절약
```

---

이제 **Target Module 선정 과정**을 완전히 이해하셨을 것입니다! 🚀
