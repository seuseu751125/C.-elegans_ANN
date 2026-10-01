"""
테스트용 C. elegans 커넥톤 데이터 생성

실제 302개 뉴런의 구조를 모방한 테스트 데이터를 생성합니다.
파이프라인을 검증할 때 사용됩니다.
"""

import json
import random
import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def generate_test_connectome(num_neurons: int = 100, num_synapses: int = 500) -> dict:
    """
    테스트용 커넥톰 데이터 생성

    Args:
        num_neurons: 생성할 뉴런 개수 (기본: 100, 원본: 302)
        num_synapses: 생성할 시냅스 개수 (기본: 500)

    Returns:
        커넥톰 데이터 딕셔너리
    """
    random.seed(42)

    # 신경 타입 정의
    neuron_types = ['sensory', 'motor', 'interneuron', 'polymodal']

    # 1. 뉴런 생성
    logger.info(f"뉴런 생성 중: {num_neurons}개")
    neurons = []

    for i in range(num_neurons):
        neuron_id = f"NEURON_{i:03d}"

        # 위치는 3D 공간에 무작위 배치
        position = [
            random.uniform(0, 400),  # x 좌표
            random.uniform(0, 400),  # y 좌표
            random.uniform(0, 100),  # z 좌표
        ]

        # 뉴런 타입 (확률적으로 할당)
        # 원본 C. elegans 비율: 감각(~30), 운동(~95), 중간(~177)
        if i < 30:
            ntype = 'sensory'
        elif i < 125:
            ntype = 'motor'
        else:
            ntype = 'interneuron'

        neuron = {
            'id': neuron_id,
            'name': neuron_id,
            'type': ntype,
            'position': position,
        }
        neurons.append(neuron)

    # 2. 시냅스 생성
    logger.info(f"시냅스 생성 중: {num_synapses}개")
    synapses = []
    created_pairs = set()

    while len(synapses) < num_synapses:
        # 무작위로 출발/도착 뉴런 선택
        source_idx = random.randint(0, num_neurons - 1)
        target_idx = random.randint(0, num_neurons - 1)

        # 자기 자신으로의 연결은 제외
        if source_idx == target_idx:
            continue

        # 중복 확인
        pair = (source_idx, target_idx)
        if pair in created_pairs:
            continue

        source_id = neurons[source_idx]['id']
        target_id = neurons[target_idx]['id']

        # 강도는 1~20 사이의 값
        weight = random.randint(1, 20)

        synapse = {
            'source': source_id,
            'target': target_id,
            'weight': weight,
            'type': 'chemical',  # 또는 'gap junction'
        }
        synapses.append(synapse)
        created_pairs.add(pair)

    # 3. 커넥톰 데이터 구성
    connectome = {
        'neurons': neurons,
        'synapses': synapses,
        'metadata': {
            'source': 'generated_test_data',
            'description': 'Test data for pipeline validation',
            'num_neurons': num_neurons,
            'num_synapses': len(synapses),
            'neuron_types': {
                'sensory': sum(1 for n in neurons if n['type'] == 'sensory'),
                'motor': sum(1 for n in neurons if n['type'] == 'motor'),
                'interneuron': sum(1 for n in neurons if n['type'] == 'interneuron'),
                'polymodal': sum(1 for n in neurons if n['type'] == 'polymodal'),
            }
        }
    }

    logger.info(f"테스트 커넥톰 생성 완료:")
    logger.info(f"  뉴런: {num_neurons}개")
    logger.info(f"  시냅스: {len(synapses)}개")
    logger.info(f"  뉴런 타입 분포: {connectome['metadata']['neuron_types']}")

    return connectome


def save_connectome(connectome: dict, output_file: str) -> None:
    """
    커넥톰 데이터를 JSON 파일로 저장

    Args:
        connectome: 커넥톤 데이터
        output_file: 저장 파일 경로
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(connectome, f, indent=2)

    logger.info(f"커넥톰 데이터 저장 완료: {output_path}")


def main():
    """메인 실행 함수"""
    config = get_config()
    output_file = config.get('data.connectome_file')

    # 테스트 데이터 생성 (크기: 100 뉴런, 500 시냅스)
    # 실제 C. elegans: 302 뉴런, ~7,000 시냅스
    connectome = generate_test_connectome(num_neurons=100, num_synapses=500)

    # 저장
    save_connectome(connectome, output_file)

    logger.info(f"✓ 테스트 데이터 생성 및 저장 완료")
    logger.info(f"  경로: {output_file}")
    logger.info(f"\n다음 단계:")
    logger.info(f"  1. 데이터 검증: python -c \"from src.data.loader import load_connectome; c = load_connectome('{output_file}'); print(c.get_statistics())\"")
    logger.info(f"  2. 데이터 탐색: notebooks/01_data_exploration.ipynb")

    return 0


if __name__ == "__main__":
    sys.exit(main())
