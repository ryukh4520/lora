"""
LoRA Target Module 탐색 유틸리티

이 스크립트는 모델의 구조를 분석하고 LoRA 적용 가능한 모듈을 찾습니다.
"""

import torch
from transformers import AutoModelForCausalLM, AutoModel
from transformers.pytorch_utils import Conv1D
from collections import defaultdict


def find_all_linear_layers(model, verbose=True):
    """
    모델의 모든 Linear 레이어 찾기
    
    Args:
        model: Hugging Face 모델
        verbose: 상세 출력 여부
    
    Returns:
        dict: {module_name: [layer_names]}
    """
    linear_layers = defaultdict(list)
    
    for name, module in model.named_modules():
        # Linear 또는 Conv1D (GPT-2의 경우)
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            # 모듈 이름 추출 (마지막 부분)
            module_name = name.split('.')[-1]
            linear_layers[module_name].append({
                'full_name': name,
                'shape': tuple(module.weight.shape),
                'params': module.weight.numel()
            })
    
    if verbose:
        print("=" * 80)
        print("📊 Linear Layer 분석 결과")
        print("=" * 80)
        
        for module_name, layers in sorted(linear_layers.items()):
            print(f"\n🔹 {module_name}: {len(layers)}개")
            if layers:
                first_layer = layers[0]
                print(f"   Shape: {first_layer['shape']}")
                print(f"   Params: {first_layer['params']:,}")
                print(f"   예시: {first_layer['full_name']}")
    
    return dict(linear_layers)


def analyze_attention_modules(model, verbose=True):
    """
    Attention 관련 모듈 분석
    
    Args:
        model: Hugging Face 모델
        verbose: 상세 출력 여부
    
    Returns:
        list: Attention 모듈 이름 리스트
    """
    attention_modules = set()
    
    for name, module in model.named_modules():
        # 'attn' 또는 'attention'이 이름에 포함된 경우
        if 'attn' in name.lower() or 'attention' in name.lower():
            if isinstance(module, (torch.nn.Linear, Conv1D)):
                module_name = name.split('.')[-1]
                attention_modules.add(module_name)
    
    if verbose:
        print("\n" + "=" * 80)
        print("🎯 Attention 모듈")
        print("=" * 80)
        for module in sorted(attention_modules):
            print(f"  ✅ {module}")
    
    return sorted(attention_modules)


def analyze_mlp_modules(model, verbose=True):
    """
    MLP/FFN 관련 모듈 분석
    
    Args:
        model: Hugging Face 모델
        verbose: 상세 출력 여부
    
    Returns:
        list: MLP 모듈 이름 리스트
    """
    mlp_modules = set()
    
    for name, module in model.named_modules():
        # 'mlp' 또는 'ffn'이 이름에 포함된 경우
        if 'mlp' in name.lower() or 'ffn' in name.lower() or 'feed_forward' in name.lower():
            if isinstance(module, (torch.nn.Linear, Conv1D)):
                module_name = name.split('.')[-1]
                mlp_modules.add(module_name)
    
    if verbose:
        print("\n" + "=" * 80)
        print("🔧 MLP/FFN 모듈")
        print("=" * 80)
        for module in sorted(mlp_modules):
            print(f"  ✅ {module}")
    
    return sorted(mlp_modules)


def suggest_target_modules(model, strategy='attention_only'):
    """
    모델에 적합한 target_modules 제안
    
    Args:
        model: Hugging Face 모델
        strategy: 'attention_only', 'attention_mlp', 'efficient'
    
    Returns:
        list: 권장 target_modules
    """
    all_layers = find_all_linear_layers(model, verbose=False)
    attn_modules = analyze_attention_modules(model, verbose=False)
    mlp_modules = analyze_mlp_modules(model, verbose=False)
    
    print("\n" + "=" * 80)
    print("💡 Target Modules 제안")
    print("=" * 80)
    
    if strategy == 'attention_only':
        print("\n전략: Attention Only (기본, 권장)")
        print("장점: 효율적, 대부분의 경우 충분")
        target = attn_modules
        
    elif strategy == 'attention_mlp':
        print("\n전략: Attention + MLP (높은 성능)")
        print("장점: 높은 표현력, 복잡한 태스크 대응")
        print("단점: 파라미터 2-3배 증가")
        target = attn_modules + mlp_modules
        
    elif strategy == 'efficient':
        print("\n전략: Efficient (메모리 제약)")
        print("장점: 파라미터 절약")
        # Query, Value만 (일반적으로 q_proj, v_proj 또는 query, value)
        target = [m for m in attn_modules if 'q' in m.lower() or 'v' in m.lower() or 'query' in m.lower() or 'value' in m.lower()]
        if not target:
            target = attn_modules[:2]  # 처음 2개만
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"\n권장 target_modules:")
    for module in target:
        print(f"  - {module}")
    
    return target


def estimate_lora_params(model, target_modules, r=8):
    """
    LoRA 적용 시 파라미터 수 추정
    
    Args:
        model: Hugging Face 모델
        target_modules: Target module 리스트
        r: LoRA rank
    
    Returns:
        dict: 파라미터 통계
    """
    total_lora_params = 0
    layer_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            module_name = name.split('.')[-1]
            if module_name in target_modules:
                # LoRA 파라미터: r * (in_features + out_features)
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]
                lora_params = r * (in_features + out_features)
                total_lora_params += lora_params
                layer_count += 1
    
    # 전체 모델 파라미터
    total_params = sum(p.numel() for p in model.parameters())
    
    stats = {
        'total_params': total_params,
        'lora_params': total_lora_params,
        'trainable_ratio': 100 * total_lora_params / total_params,
        'layer_count': layer_count
    }
    
    print("\n" + "=" * 80)
    print("📈 LoRA 파라미터 추정")
    print("=" * 80)
    print(f"전체 파라미터:     {stats['total_params']:,}")
    print(f"LoRA 파라미터:     {stats['lora_params']:,}")
    print(f"학습 비율:         {stats['trainable_ratio']:.4f}%")
    print(f"적용 레이어 수:    {stats['layer_count']}")
    
    return stats


def verify_target_modules(model, target_modules):
    """
    Target modules가 실제로 존재하는지 확인
    
    Args:
        model: Hugging Face 모델
        target_modules: 확인할 module 리스트
    
    Returns:
        dict: 검증 결과
    """
    all_modules = set()
    matched_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            module_name = name.split('.')[-1]
            all_modules.add(module_name)
            if module_name in target_modules:
                matched_layers.append(name)
    
    print("\n" + "=" * 80)
    print("✅ Target Modules 검증")
    print("=" * 80)
    
    for target in target_modules:
        if target in all_modules:
            count = sum(1 for name in matched_layers if name.endswith(target))
            print(f"  ✅ {target}: 존재함 ({count}개 레이어)")
        else:
            print(f"  ❌ {target}: 존재하지 않음!")
    
    if not all(t in all_modules for t in target_modules):
        print(f"\n사용 가능한 모듈:")
        for module in sorted(all_modules):
            print(f"  - {module}")
    
    return {
        'valid': all(t in all_modules for t in target_modules),
        'matched_count': len(matched_layers),
        'available_modules': sorted(all_modules)
    }


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LoRA Target Module 탐색')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='모델 이름 (예: gpt2, meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--strategy', type=str, default='attention_only',
                       choices=['attention_only', 'attention_mlp', 'efficient'],
                       help='Target module 선정 전략')
    parser.add_argument('--r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--verify', type=str, nargs='+',
                       help='검증할 target modules (예: --verify c_attn c_proj)')
    
    args = parser.parse_args()
    
    print(f"\n🔍 모델 분석 중: {args.model}")
    print("=" * 80)
    
    # 모델 로드
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True
        )
    except:
        try:
            model = AutoModel.from_pretrained(
                args.model,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return
    
    # 분석
    find_all_linear_layers(model)
    analyze_attention_modules(model)
    analyze_mlp_modules(model)
    
    # 제안
    target_modules = suggest_target_modules(model, args.strategy)
    
    # 파라미터 추정
    estimate_lora_params(model, target_modules, args.r)
    
    # 검증 (사용자 지정 시)
    if args.verify:
        verify_target_modules(model, args.verify)
    
    # 설정 출력
    print("\n" + "=" * 80)
    print("📝 LoRA 설정 예시")
    print("=" * 80)
    print(f"""
lora_config = {{
    "r": {args.r},
    "lora_alpha": {args.r * 2},
    "target_modules": {target_modules},
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}}
""")


if __name__ == "__main__":
    main()
