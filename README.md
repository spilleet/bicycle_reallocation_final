# 서울시 따릉이 재배치 최적화 시스템

## 📋 프로젝트 소개
서울시 공공자전거 '따릉이'의 효율적인 재배치를 위한 웹 기반 최적화 시스템입니다. 실시간 데이터를 기반으로 각 구별 자전거 불균형을 분석하고, 최적의 재배치 경로를 계산하여 시각화합니다.

## 🚀 주요 기능

### 1. 실시간 데이터 수집
- 서울시 열린데이터광장 API를 통한 실시간 따릉이 대여소 정보 수집
- 구별 자동 분류 및 재배치 필요도 분석
- 긴급도 점수 기반 우선순위 산정

### 2. 최적화 알고리즘
- **OR-Tools 기반 최적화**: Google OR-Tools를 활용한 차량 경로 문제(VRP) 해결
- **K-means 클러스터링**: 대규모 문제를 작은 단위로 분할하여 처리
- **휴리스틱 알고리즘**: 최단 근접 이웃 기반 빠른 경로 생성
- **적응형 알고리즘 선택**: 문제 규모에 따라 자동으로 최적 알고리즘 선택

### 3. 대시보드 기능
- **인터랙티브 지도**: Leaflet 기반 실시간 경로 시각화
- **트럭별 경로 표시**: 색상 구분된 경로 및 정류소 마커
- **실시간 통계**: 구별 불균형 현황 및 재배치 필요 대수
- **경로 상세 정보**: 각 트럭의 수거/배송 계획 및 이동 거리

## 🛠 기술 스택

### Backend
- **Python 3.13+**
- **Flask**: 웹 프레임워크
- **OR-Tools**: 최적화 엔진
- **scikit-learn**: K-means 클러스터링
- **NumPy**: 수치 연산

### Frontend
- **Alpine.js**: 반응형 UI 프레임워크
- **Tailwind CSS**: 유틸리티 기반 스타일링
- **Leaflet.js**: 인터랙티브 지도
- **Chart.js**: 데이터 시각화

## 📦 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/spilleet/bicycle_reallocation_final
cd bicycle_reallocation_final
```

### 2. 가상환경 설정 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정 (선택사항)
```bash
export SEOUL_API_KEY="your_api_key_here"  # 기본값 제공됨
```

## 🚀 실행 방법

### 웹 애플리케이션 실행
```bash
python app_new.py
```
브라우저에서 http://localhost:8080 접속

### CLI 모드 실행
```bash
python reallocation_final.py
```

## 📱 사용 방법

### 1. 데이터 수집
- 대시보드 상단의 "데이터 수집" 버튼 클릭
- 서울시 25개 구의 실시간 따릉이 현황 자동 수집

### 2. 구 선택
- 좌측 패널에서 재배치가 필요한 구 선택
- 긴급도 점수가 높은 구가 상단에 표시됨

### 3. 최적화 실행
- 트럭 대수 및 용량 설정
- "최적화 실행" 버튼 클릭
- 또는 "자동 최적화"로 AI가 최적 파라미터 자동 설정

### 4. 결과 확인
- 지도에서 트럭별 경로 확인
- 각 경로의 수거/배송 정류소 확인
- CSV 형식으로 경로 내보내기 가능

## 📊 주요 알고리즘

### 1. 구별 긴급도 산정
```python
urgency_score = (
    len(pickup_needed) * 2 +  # 수거 필요 정류소 (가중치 2)
    len(delivery_needed) * 1.5 +  # 배송 필요 정류소 (가중치 1.5)
    total_imbalance * 0.1  # 총 불균형 대수
)
```

### 2. 클러스터링 기준
- 30개 이상 정류소: K-means 클러스터링 적용
- 30개 미만: 직접 OR-Tools 적용
- 실패 시: 휴리스틱 알고리즘으로 폴백

### 3. 제약 조건
- 트럭 용량: 기본 20대
- 시간 제약: 없음 (거리 최소화만 고려)
- 수거/배송 균형: 각 정류소의 요구사항 충족

## 🗂 프로젝트 구조
```
bicycle_reallocation_final/
├── app_new.py              # Flask 웹 애플리케이션
├── reallocation_final.py   # 핵심 최적화 로직
├── templates/
│   ├── dashboard.html      # 메인 대시보드
│   └── index_new.html      # 클래식 버전
├── requirements.txt        # Python 패키지 목록
└── README.md              # 프로젝트 문서
```

## 🔧 주요 설정

### 트럭 설정
- **기본 트럭 대수**: 2대
- **트럭 용량**: 20대
- **차고지 위치**: 각 구청 위치 기준

### API 설정
- **데이터 갱신 주기**: 30분 (백그라운드)
- **API 제한**: 시간당 1000건
- **캐시 유효시간**: 5분

## 📈 성능 최적화

### 1. 캐싱 전략
- Flask-Caching으로 API 응답 캐싱
- 5분간 동일 데이터 재사용

### 2. 병렬 처리
- 클러스터별 독립적 최적화
- 백그라운드 데이터 업데이트

### 3. 적응형 알고리즘
- 문제 규모에 따른 알고리즘 자동 선택
- 실패 시 자동 폴백 메커니즘

## 🎯 시각화 특징

### 1. 트럭별 경로 시각화
- 각 트럭마다 고유한 색상으로 경로 표시
- 수거/배송 정류소를 동일한 색상의 마커로 표시
- 출발/도착 지점 특별 표시

### 2. 인터랙티브 기능
- 경로 클릭 시 상세 정보 팝업
- "보기" 버튼으로 특정 트럭 경로 포커스
- 지도 줌/패닝 기능

### 3. 토스트 알림
- 작업 완료/오류 알림
- 3초 후 자동 사라짐
- 수동 닫기 버튼 제공

## 📊 API 엔드포인트

### POST `/api/data/collect`
- 실시간 데이터 수집 및 분석

### GET `/api/districts`
- 구별 통계 정보 조회

### GET `/api/district/<district_id>`
- 특정 구 상세 정보

### POST `/api/optimize`
- 경로 최적화 실행
```json
{
    "district_id": "강남구",
    "num_vehicles": 2,
    "vehicle_capacity": 20,
    "use_clustering": true
}
```

### GET `/api/statistics`
- 전체 통계 정보

### GET `/api/export/<format>`
- 데이터 내보내기 (json/csv)
