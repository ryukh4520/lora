"""
Data preparation script for LoRA fine-tuning.

This script downloads and prepares datasets for training.
Supports:
- KoAlpaca dataset (Korean instruction-following)
- Custom JSON datasets
- Sample/demo datasets
"""

import os
import json
import argparse
import random
from typing import List, Dict
from datasets import load_dataset


def create_sample_dataset(num_samples: int = 100) -> List[Dict]:
    """
    Create a sample dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of instruction-output pairs
    """
    print(f"📝 Creating sample dataset with {num_samples} samples...")
    
    # Sample Korean instruction-following data
    samples = [
        {
            "instruction": "한국의 수도는 어디인가요?",
            "input": "",
            "output": "한국의 수도는 서울입니다. 서울은 대한민국의 정치, 경제, 문화의 중심지이며, 약 1천만 명의 인구가 거주하고 있습니다."
        },
        {
            "instruction": "김치의 효능에 대해 설명해주세요.",
            "input": "",
            "output": "김치는 한국의 대표적인 발효 음식으로, 유산균이 풍부하여 장 건강에 좋습니다. 또한 비타민 C와 식이섬유가 풍부하며, 항산화 효과도 있어 면역력 향상에 도움이 됩니다."
        },
        {
            "instruction": "다음 문장을 영어로 번역하세요.",
            "input": "오늘 날씨가 정말 좋네요.",
            "output": "The weather is really nice today."
        },
        {
            "instruction": "Python에서 리스트와 튜플의 차이점은 무엇인가요?",
            "input": "",
            "output": "리스트(list)는 변경 가능한(mutable) 자료구조로, 요소를 추가, 삭제, 수정할 수 있습니다. 반면 튜플(tuple)은 변경 불가능한(immutable) 자료구조로, 한 번 생성되면 내용을 변경할 수 없습니다. 리스트는 대괄호 []로, 튜플은 소괄호 ()로 표현합니다."
        },
        {
            "instruction": "서울의 주요 관광지를 추천해주세요.",
            "input": "",
            "output": "서울의 주요 관광지로는 경복궁, 남산타워, 명동, 인사동, 홍대, 강남 등이 있습니다. 경복궁은 조선시대의 대표적인 궁궐이며, 남산타워에서는 서울 전경을 감상할 수 있습니다. 명동과 홍대는 쇼핑과 먹거리로 유명하고, 인사동은 전통 문화를 체험할 수 있는 곳입니다."
        },
        {
            "instruction": "인공지능이란 무엇인가요?",
            "input": "",
            "output": "인공지능(AI, Artificial Intelligence)은 인간의 학습능력, 추론능력, 지각능력 등을 컴퓨터 프로그램으로 구현한 기술입니다. 머신러닝, 딥러닝 등의 기술을 통해 데이터로부터 패턴을 학습하고, 이를 바탕으로 예측이나 의사결정을 수행합니다."
        },
        {
            "instruction": "다음 숫자들의 평균을 계산하세요.",
            "input": "10, 20, 30, 40, 50",
            "output": "주어진 숫자들의 평균은 30입니다. 계산 과정: (10 + 20 + 30 + 40 + 50) / 5 = 150 / 5 = 30"
        },
        {
            "instruction": "건강한 아침 식사 메뉴를 추천해주세요.",
            "input": "",
            "output": "건강한 아침 식사로는 통곡물 빵, 계란, 샐러드, 과일, 우유 또는 요거트를 추천합니다. 통곡물은 식이섬유가 풍부하고, 계란은 양질의 단백질을 제공합니다. 과일과 채소는 비타민과 미네랄을 공급하며, 유제품은 칼슘을 보충해줍니다."
        },
        {
            "instruction": "환경 보호를 위해 개인이 할 수 있는 일은 무엇인가요?",
            "input": "",
            "output": "개인이 환경 보호를 위해 할 수 있는 일로는 일회용품 사용 줄이기, 분리수거 철저히 하기, 대중교통 이용하기, 에너지 절약하기, 친환경 제품 사용하기 등이 있습니다. 작은 실천들이 모여 큰 변화를 만들 수 있습니다."
        },
        {
            "instruction": "스트레스 해소 방법을 알려주세요.",
            "input": "",
            "output": "스트레스 해소 방법으로는 규칙적인 운동, 충분한 수면, 명상이나 요가, 취미 활동, 친구나 가족과의 대화 등이 있습니다. 또한 깊은 호흡이나 산책도 즉각적인 스트레스 완화에 도움이 됩니다."
        }
    ]
    
    # Repeat samples to reach desired number
    dataset = []
    while len(dataset) < num_samples:
        dataset.extend(samples)
    
    # Shuffle and trim to exact number
    random.shuffle(dataset)
    dataset = dataset[:num_samples]
    
    print(f"✅ Created {len(dataset)} sample entries")
    return dataset


def download_koalpaca_dataset(num_samples: int = None) -> List[Dict]:
    """
    Download KoAlpaca dataset from Hugging Face.
    
    Args:
        num_samples: Number of samples to use (None for all)
        
    Returns:
        List of instruction-output pairs
    """
    print("📥 Downloading KoAlpaca dataset from Hugging Face...")
    
    try:
        # Load KoAlpaca dataset
        dataset = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
        
        # Convert to our format
        data = []
        for item in dataset:
            data.append({
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", "")
            })
        
        # Limit samples if specified
        if num_samples and num_samples < len(data):
            random.shuffle(data)
            data = data[:num_samples]
        
        print(f"✅ Downloaded {len(data)} samples from KoAlpaca")
        return data
    
    except Exception as e:
        print(f"❌ Failed to download KoAlpaca: {e}")
        print("💡 Falling back to sample dataset...")
        return create_sample_dataset(num_samples or 1000)


def load_custom_dataset(file_path: str) -> List[Dict]:
    """
    Load custom dataset from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of instruction-output pairs
    """
    print(f"📂 Loading custom dataset from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} samples")
    return data


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True
) -> tuple:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data: Full dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        shuffle: Whether to shuffle before splitting
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    if shuffle:
        random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train:      {len(train_data):5d} samples ({len(train_data)/total*100:.1f}%)")
    print(f"   Validation: {len(val_data):5d} samples ({len(val_data)/total*100:.1f}%)")
    print(f"   Test:       {len(test_data):5d} samples ({len(test_data)/total*100:.1f}%)")
    print(f"   Total:      {total:5d} samples")
    
    return train_data, val_data, test_data


def save_dataset(data: List[Dict], output_path: str):
    """
    Save dataset to JSON file.
    
    Args:
        data: Dataset to save
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA fine-tuning")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample",
        choices=["sample", "koalpaca", "custom"],
        help="Dataset type to use"
    )
    parser.add_argument(
        "--custom_file",
        type=str,
        help="Path to custom dataset file (required if dataset=custom)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use (for sample/koalpaca)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("\n" + "="*60)
    print("🚀 Data Preparation Script")
    print("="*60)
    
    # Load dataset
    if args.dataset == "sample":
        data = create_sample_dataset(args.num_samples)
    elif args.dataset == "koalpaca":
        data = download_koalpaca_dataset(args.num_samples)
    elif args.dataset == "custom":
        if not args.custom_file:
            raise ValueError("--custom_file is required when dataset=custom")
        data = load_custom_dataset(args.custom_file)
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Save datasets
    save_dataset(train_data, os.path.join(args.output_dir, "train.json"))
    save_dataset(val_data, os.path.join(args.output_dir, "validation.json"))
    save_dataset(test_data, os.path.join(args.output_dir, "test.json"))
    
    # Print sample
    print("\n" + "="*60)
    print("📝 Sample Data:")
    print("="*60)
    sample = train_data[0]
    print(f"Instruction: {sample['instruction']}")
    if sample['input']:
        print(f"Input: {sample['input']}")
    print(f"Output: {sample['output'][:100]}...")
    print("="*60)
    
    print("\n✅ Data preparation complete!")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
