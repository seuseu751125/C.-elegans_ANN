"""
C. elegans 커넥톰 데이터 다운로드 스크립트

GitHub 저장소에서 C. elegans 신경망 연결 데이터를 다운로드합니다.
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ConnectomeDownloader:
    """
    C. elegans 커넥톰 데이터를 다운로드하고 변환하는 클래스
    """

    # 데이터 소스
    GITHUB_BASE = "https://raw.githubusercontent.com/altundag/celegans-connectome/main"

    def __init__(self, output_dir: str):
        """
        다운로더 초기화

        Args:
            output_dir: 데이터 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_connectome_csv(self) -> Optional[str]:
        """
        GitHub에서 CSV 형식의 커넥톰 데이터 다운로드

        Returns:
            다운로드한 파일 경로 또는 None
        """
        url = f"{self.GITHUB_BASE}/connectome_dataprocessed.csv"
        output_file = self.output_dir / "connectome_raw.csv"

        logger.info(f"데이터 다운로드 중: {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            logger.info(f"다운로드 완료: {output_file}")
            return str(output_file)

        except requests.exceptions.RequestException as e:
            logger.error(f"다운로드 실패: {e}")
            return None

    def csv_to_json(self, csv_file: str) -> Optional[str]:
        """
        CSV 파일을 JSON 형식으로 변환

        CSV 파일 형식:
        Neuron1,Neuron2,Weight,Neurotransmitter
        AVAL,AVAR,7,Acetylcholine
        ...

        Args:
            csv_file: CSV 파일 경로

        Returns:
            변환된 JSON 파일 경로 또는 None
        """
        try:
            import csv

            neurons_set = set()
            synapses = []

            logger.info(f"CSV 파일 파싱 중: {csv_file}")

            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if not row.get('Neuron1') or not row.get('Neuron2'):
                        continue

                    source = row['Neuron1'].strip()
                    target = row['Neuron2'].strip()
                    weight = int(row.get('Weight', 1))

                    neurons_set.add(source)
                    neurons_set.add(target)

                    # 가중치가 0이 아닌 경우만 시냅스로 간주
                    if weight > 0:
                        synapse = {
                            'source': source,
                            'target': target,
                            'weight': weight,
                            'type': 'chemical'  # 기본값
                        }
                        synapses.append(synapse)

            # 뉴런 정보 생성 (위치 정보는 없으므로 기본값 사용)
            neurons = [
                {
                    'id': nid,
                    'name': nid,
                    'type': self._infer_neuron_type(nid),
                    'position': self._get_default_position(nid),
                }
                for nid in sorted(neurons_set)
            ]

            # JSON 형식으로 생성
            connectome_data = {
                'neurons': neurons,
                'synapses': synapses,
                'metadata': {
                    'source': 'altundag/celegans-connectome',
                    'num_neurons': len(neurons),
                    'num_synapses': len(synapses),
                }
            }

            # JSON 파일로 저장
            json_file = self.output_dir / 'connectome.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(connectome_data, f, indent=2)

            logger.info(f"JSON 변환 완료: {json_file}")
            logger.info(f"  뉴런: {len(neurons)}개")
            logger.info(f"  시냅스: {len(synapses)}개")

            return str(json_file)

        except Exception as e:
            logger.error(f"CSV 파싱 오류: {e}")
            return None

    @staticmethod
    def _infer_neuron_type(neuron_id: str) -> str:
        """
        뉴런 ID로부터 뉴런의 대략적인 타입 추론

        C. elegans 뉴런 명명 규칙:
        - 'R'로 시작: 감각 뉴런 (sensory)
        - 'M'으로 끝남: 운동 뉴런 (motor)
        - 나머지: 중간 뉴런 (interneuron)

        Args:
            neuron_id: 뉴런 ID

        Returns:
            뉴런 타입 추론값
        """
        if neuron_id.startswith(('ADA', 'ADC', 'ADL', 'AFD', 'AIA', 'AIB', 'AIN',
                                  'AIY', 'AUA', 'AWA', 'AWB', 'AWC', 'ASE', 'ASG',
                                  'ASH', 'ASI', 'ASJ', 'ASK', 'OLQ', 'OLL', 'PHA',
                                  'PHB', 'PHC', 'PLM', 'PLN', 'PVD', 'PVM', 'PVN',
                                  'PVP', 'PVR', 'PVT', 'URX', 'URY')):
            return 'sensory'
        elif neuron_id.endswith(('M', 'VA', 'VB', 'VC', 'VD')):
            return 'motor'
        else:
            return 'interneuron'

    @staticmethod
    def _get_default_position(neuron_id: str) -> List[float]:
        """
        뉴런의 기본 위치 설정 (실제 좌표 대신 임의값 사용)

        실제 위치 정보는 WormAtlas에서 얻을 수 있지만,
        여기서는 임시로 난수를 사용합니다.

        Args:
            neuron_id: 뉴런 ID

        Returns:
            3D 좌표 [x, y, z]
        """
        import hashlib

        # 뉴런 ID 기반으로 재현 가능한 위치 생성
        hash_obj = hashlib.md5(neuron_id.encode())
        hash_bytes = hash_obj.digest()

        x = (hash_bytes[0] / 255.0) * 400
        y = (hash_bytes[1] / 255.0) * 400
        z = (hash_bytes[2] / 255.0) * 400

        return [x, y, z]

    def download_and_convert(self) -> Optional[str]:
        """
        데이터 다운로드 및 변환 실행

        Returns:
            최종 JSON 파일 경로 또는 None
        """
        logger.info("C. elegans 커넥톰 데이터 다운로드 및 변환 시작")

        # 1단계: CSV 다운로드
        csv_file = self.download_connectome_csv()
        if not csv_file:
            logger.error("CSV 파일 다운로드 실패")
            return None

        # 2단계: CSV to JSON 변환
        json_file = self.csv_to_json(csv_file)
        if not json_file:
            logger.error("JSON 변환 실패")
            return None

        logger.info(f"완료! 데이터 경로: {json_file}")
        return json_file


def main():
    """메인 실행 함수"""
    config = get_config()
    data_dir = config.get('data.raw_dir')

    logger.info(f"저장 디렉토리: {data_dir}")

    downloader = ConnectomeDownloader(data_dir)
    result = downloader.download_and_convert()

    if result:
        logger.info(f"✓ 데이터 다운로드 및 변환 성공")
        logger.info(f"  저장 위치: {result}")
        return 0
    else:
        logger.error("✗ 데이터 다운로드 및 변환 실패")
        return 1


if __name__ == "__main__":
    sys.exit(main())
