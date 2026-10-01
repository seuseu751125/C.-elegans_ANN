"""
C. elegans 커넥톰 데이터 로더

WormAtlas 또는 GitHub에서 받은 C. elegans 신경망 데이터를 로드합니다.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class Connectome:
    """
    C. elegans 커넥톰(신경연결 지도) 데이터를 관리하는 클래스.

    커넥톰은 다음 정보를 포함합니다:
    - neurons: 뉴런 목록 (ID, 위치, 타입 등)
    - synapses: 시냅스 목록 (출발 뉴런, 도착 뉴런, 강도 등)
    """

    def __init__(self, neurons: List[Dict], synapses: List[Dict]):
        """
        커넥톰 초기화

        Args:
            neurons: 뉴런 정보 리스트
            synapses: 시냅스 정보 리스트
        """
        self.neurons = neurons
        self.synapses = synapses

        # 뉴런 ID로 빠르게 접근하기 위한 딕셔너리 생성
        self.neuron_dict = {n['id']: n for n in neurons}

        # 기본 통계
        self.num_neurons = len(neurons)
        self.num_synapses = len(synapses)

        logger.info(f"커넥톰 로드 완료: {self.num_neurons}개 뉴런, {self.num_synapses}개 시냅스")

    def get_neuron_info(self, neuron_id: str) -> Optional[Dict]:
        """
        특정 뉴런의 정보 조회

        Args:
            neuron_id: 뉴런 ID (예: 'AVAL')

        Returns:
            뉴런 정보 딕셔너리 또는 None
        """
        return self.neuron_dict.get(neuron_id)

    def get_neuron_position(self, neuron_id: str) -> Optional[Tuple[float, float, float]]:
        """뉴런의 3D 위치 좌표 반환"""
        neuron = self.get_neuron_info(neuron_id)
        if neuron and 'position' in neuron:
            pos = neuron['position']
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                return tuple(pos)
        return None

    def get_neuron_type(self, neuron_id: str) -> Optional[str]:
        """뉴런의 타입 반환 (sensory, motor, interneuron 등)"""
        neuron = self.get_neuron_info(neuron_id)
        if neuron:
            return neuron.get('type', 'unknown')
        return None

    def get_all_neuron_ids(self) -> List[str]:
        """모든 뉴런 ID 반환"""
        return [n['id'] for n in self.neurons]

    def get_incoming_synapses(self, target_neuron_id: str) -> List[Dict]:
        """
        특정 뉴런으로 들어오는 모든 시냅스 반환

        Args:
            target_neuron_id: 도착 뉴런 ID

        Returns:
            해당 뉴런으로 들어오는 시냅스 리스트
        """
        return [s for s in self.synapses if s.get('target') == target_neuron_id]

    def get_outgoing_synapses(self, source_neuron_id: str) -> List[Dict]:
        """
        특정 뉴런에서 나가는 모든 시냅스 반환

        Args:
            source_neuron_id: 출발 뉴런 ID

        Returns:
            해당 뉴런에서 나가는 시냅스 리스트
        """
        return [s for s in self.synapses if s.get('source') == source_neuron_id]

    def is_connected(self, source_id: str, target_id: str) -> bool:
        """
        두 뉴런 사이에 시냅스가 존재하는지 확인

        Args:
            source_id: 출발 뉴런 ID
            target_id: 도착 뉴런 ID

        Returns:
            True if 연결 exists, False otherwise
        """
        for synapse in self.synapses:
            if synapse.get('source') == source_id and synapse.get('target') == target_id:
                return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """커넥톰의 통계 정보 반환"""
        neuron_types = {}
        for neuron in self.neurons:
            ntype = neuron.get('type', 'unknown')
            neuron_types[ntype] = neuron_types.get(ntype, 0) + 1

        synapse_types = {}
        for synapse in self.synapses:
            stype = synapse.get('type', 'chemical')
            synapse_types[stype] = synapse_types.get(stype, 0) + 1

        return {
            'num_neurons': self.num_neurons,
            'num_synapses': self.num_synapses,
            'neuron_types': neuron_types,
            'synapse_types': synapse_types,
            'avg_synapses_per_neuron': self.num_synapses / self.num_neurons if self.num_neurons > 0 else 0,
        }


def load_connectome(connectome_file: str) -> Connectome:
    """
    JSON 파일에서 커넥톰 데이터 로드

    Args:
        connectome_file: 커넥톰 JSON 파일 경로

    Returns:
        Connectome 객체

    Raises:
        FileNotFoundError: 파일이 없을 때
        json.JSONDecodeError: JSON 형식이 잘못되었을 때
    """
    file_path = Path(connectome_file)

    if not file_path.exists():
        raise FileNotFoundError(f"커넥톰 파일을 찾을 수 없습니다: {connectome_file}")

    logger.info(f"커넥톤 파일 로드 중: {connectome_file}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        raise

    # 데이터 검증
    if 'neurons' not in data or 'synapses' not in data:
        raise ValueError("커넥톰 파일에 'neurons'와 'synapses' 필드가 필요합니다")

    connectome = Connectome(data['neurons'], data['synapses'])
    return connectome


def extract_neuron_pairs(connectome: Connectome) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    커넥톰에서 모든 신경 쌍과 시냅스 여부 추출

    이 함수는 다음을 반환합니다:
    - 모든 가능한 신경 쌍 (N x (N-1) 개, N = 뉴런 수)
    - 각 쌍에 대한 시냅스 존재 여부 (1 = 있음, 0 = 없음)

    Args:
        connectome: Connectome 객체

    Returns:
        (신경 쌍 리스트, 시냅스 레이블 리스트)
    """
    neuron_ids = connectome.get_all_neuron_ids()
    pairs = []
    labels = []

    # 모든 신경 쌍 생성 (source != target)
    for source_id in neuron_ids:
        for target_id in neuron_ids:
            if source_id != target_id:
                pairs.append((source_id, target_id))
                # 시냅스 존재 여부 확인
                label = 1 if connectome.is_connected(source_id, target_id) else 0
                labels.append(label)

    logger.info(f"추출된 신경 쌍: {len(pairs)}개, 양성 샘플(시냅스 있음): {sum(labels)}개")

    return pairs, labels


def generate_neuron_pair_data(
    connectome: Connectome,
    pairs: List[Tuple[str, str]],
    labels: List[int]
) -> Tuple[List[Dict], List[int]]:
    """
    신경 쌍에 대한 메타데이터 생성

    각 신경 쌍에 대해 다음 정보를 포함하는 딕셔너리를 생성합니다:
    - source_neuron_id: 출발 뉴런 ID
    - target_neuron_id: 도착 뉴런 ID
    - source_position: 출발 뉴런의 3D 위치
    - target_position: 도착 뉴런의 3D 위치
    - source_type: 출발 뉴런의 타입
    - target_type: 도착 뉴런의 타입

    Args:
        connectome: Connectome 객체
        pairs: 신경 쌍 리스트
        labels: 시냅스 존재 여부 리스트

    Returns:
        (메타데이터 리스트, 레이블 리스트)
    """
    data = []

    for (source_id, target_id), label in zip(pairs, labels):
        source_pos = connectome.get_neuron_position(source_id)
        target_pos = connectome.get_neuron_position(target_id)
        source_type = connectome.get_neuron_type(source_id)
        target_type = connectome.get_neuron_type(target_id)

        pair_data = {
            'source_neuron_id': source_id,
            'target_neuron_id': target_id,
            'source_position': source_pos,
            'target_position': target_pos,
            'source_type': source_type,
            'target_type': target_type,
            'synapse_exists': label,
        }
        data.append(pair_data)

    return data, labels


if __name__ == "__main__":
    # 테스트 코드 (선택사항)
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        connectome_file = sys.argv[1]
        try:
            connectome = load_connectome(connectome_file)
            stats = connectome.get_statistics()
            print("\n커넥톰 통계:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            pairs, labels = extract_neuron_pairs(connectome)
            print(f"\n신경 쌍 추출: {len(pairs)}개")
            print(f"양성 샘플: {sum(labels)}개")
            print(f"음성 샘플: {len(labels) - sum(labels)}개")
        except Exception as e:
            print(f"오류: {e}")
