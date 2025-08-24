# 서울시 따릉이 재배치 시스템 웹사이트

## 📋 개요

이 프로젝트는 서울시 공공데이터 API를 활용하여 실시간 따릉이 대여소 정보를 수집하고, 클러스터링 기반 최적화 알고리즘을 통해 효율적인 자전거 재배치 계획을 수립하는 웹 애플리케이션입니다.

## ✨ 주요 기능

### 1. 실시간 데이터 수집
- 서울시 공공데이터 API를 통한 실시간 따릉이 대여소 정보 수집
- 구별 자동 분류 및 분석
- 대여소별 수거/배송 필요도 자동 계산

### 2. 클러스터링 기반 최적화
- K-means 클러스터링을 통한 대여소 그룹화
- 수거/배송 작업량 균형 고려
- 지리적 근접성 기반 클러스터 병합

### 3. 경로 최적화
- OR-Tools를 활용한 Vehicle Routing Problem (VRP) 해결
- 휴리스틱 알고리즘을 통한 대규모 문제 처리
- 트럭 용량 및 거리 제약 조건 고려

### 4. 시각화 및 결과 분석
- Leaflet 지도를 통한 대여소 위치 표시
- 최적화된 경로의 시각적 표현
- 상세한 경로 정보 및 통계 제공

## 🚀 설치 및 실행

### 1. 환경 요구사항
- Python 3.8 이상
- pip (Python 패키지 관리자)

### 2. 설치 단계

```bash
# 1. 프로젝트 클론 또는 다운로드
git clone <repository-url>
cd bicycle_reallocation_final2

# 2. 가상환경 생성 (권장)
python -m venv venv

# 3. 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 4. 필요한 패키지 설치
pip install -r requirements.txt
```

### 3. 실행

```bash
# Flask 애플리케이션 실행
python app.py
```

웹 브라우저에서 `http://localhost:5000`으로 접속하여 시스템을 사용할 수 있습니다.

## 📖 사용법

### 1. 데이터 수집
- 웹페이지에서 "데이터 수집 시작" 버튼 클릭
- 서울시 공공데이터 API를 통해 실시간 따릉이 정보 수집
- 구별 분석 및 긴급도 계산

### 2. 구별 분석
- 수집된 데이터를 바탕으로 구별 재배치 필요도 분석
- 긴급도 순으로 구 목록 표시
- 각 구의 상세 정보 확인 (수거/배송 필요 대여소 수, 총 불균형 등)

### 3. 재배치 계획 수립
- 재배치가 필요한 구 선택
- 트럭 수 및 용량 설정
- 클러스터링 사용 여부 선택
- "최적화 실행" 버튼으로 경로 최적화 수행

### 4. 결과 확인
- 지도상에 최적화된 경로 표시
- 각 트럭별 상세 경로 정보 확인
- 총 이동 거리 및 작업량 통계 확인

## 🏗️ 시스템 아키텍처

### 백엔드 (Flask)
- **데이터 수집**: 서울시 공공데이터 API 연동
- **구 분류**: GeoJSON 기반 좌표 매핑
- **클러스터링**: scikit-learn K-means 알고리즘
- **최적화**: OR-Tools VRP 솔버
- **API 엔드포인트**: RESTful API 제공

### 프론트엔드 (HTML/CSS/JavaScript)
- **Bootstrap**: 반응형 UI 프레임워크
- **Leaflet**: 인터랙티브 지도 라이브러리
- **Chart.js**: 데이터 시각화
- **AJAX**: 비동기 데이터 통신

## 🔧 기술 스택

### 백엔드
- **Flask**: Python 웹 프레임워크
- **OR-Tools**: Google 최적화 라이브러리
- **scikit-learn**: 머신러닝 라이브러리
- **NumPy**: 수치 계산 라이브러리

### 프론트엔드
- **Bootstrap 5**: CSS 프레임워크
- **Leaflet**: 오픈소스 지도 라이브러리
- **Font Awesome**: 아이콘 라이브러리
- **Vanilla JavaScript**: ES6+ 모던 JavaScript

## 📊 알고리즘 상세

### 1. 클러스터링 알고리즘
```python
# K-means 클러스터링
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(features_scaled)
```

### 2. 경로 최적화
```python
# OR-Tools VRP 모델
manager = pywrapcp.RoutingIndexManager(len(nodes), num_vehicles, 0)
routing = pywrapcp.RoutingModel(manager)
```

### 3. 거리 계산
```python
# Haversine 공식을 통한 지구상 두 점 간 거리 계산
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    # ... 거리 계산 로직
```

## 🎯 최적화 목표

1. **효율성**: 총 이동 거리 최소화
2. **균형**: 수거/배송 작업량 균등 분배
3. **실용성**: 트럭 용량 및 시간 제약 고려
4. **확장성**: 대규모 문제 처리 가능

## 📈 성능 지표

- **처리 시간**: 일반적인 구별 문제 (30초 이내)
- **정확도**: OR-Tools 최적해 vs 휴리스틱 해 비교
- **확장성**: 최대 1000개 대여소, 10대 트럭까지 지원

## 🔍 API 엔드포인트

### GET `/api/districts`
- 구별 분석 데이터 반환
- 실시간 따릉이 정보 수집 및 분석

### POST `/api/solve`
- 특정 구의 재배치 문제 해결
- 클러스터링 및 경로 최적화 수행

### GET `/api/stations/<district>`
- 특정 구의 대여소 정보 반환
- 수거/배송 필요 대여소 목록

## 🚨 주의사항

1. **API 키**: 서울시 공공데이터 API 키가 필요합니다
2. **데이터 제한**: API 호출 횟수 제한이 있을 수 있습니다
3. **성능**: 대규모 문제의 경우 처리 시간이 오래 걸릴 수 있습니다

## 🔮 향후 개선 계획

1. **실시간 모니터링**: 대시보드 및 알림 시스템
2. **예측 모델링**: 수요 예측을 통한 사전 재배치 계획
3. **모바일 앱**: iOS/Android 네이티브 앱 개발
4. **AI 최적화**: 딥러닝 기반 경로 최적화

## 📞 문의 및 지원

프로젝트 관련 문의사항이나 버그 리포트는 이슈 트래커를 통해 제출해 주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**개발자**: AI Assistant  
**최종 업데이트**: 2024년 12월  
**버전**: 1.0.0
