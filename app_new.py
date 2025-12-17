"""
서울시 따릉이 재배치 시스템 
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_caching import Cache
import json
import os
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Optional, Tuple
import io
import csv

# 재배치 로직 임포트
from reallocation_final import (
    SeoulDistrictClassifier,
    BikeStationClusterer,
    get_bike_station_data_by_district,
    analyze_district_redistribution_needs,
    solve_district_with_clustering,
    solve_single_cluster_with_ortools
)

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'seoul-bike-reallocation-2024'
app.config['JSON_AS_ASCII'] = False

# CORS 설정 (API 접근 허용)
CORS(app)

# 캐싱 설정 (성능 향상)
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5분 캐시
})

# 전역 데이터 저장소
class DataStore:
    """애플리케이션 데이터 저장소"""
    def __init__(self):
        self.district_data = {}
        self.district_analysis = {}
        self.last_update = None
        self.update_lock = threading.Lock()
        self.optimization_results = {}
        
    def update_data(self, district_data, district_analysis):
        """데이터 업데이트"""
        with self.update_lock:
            self.district_data = district_data
            self.district_analysis = district_analysis
            self.last_update = datetime.now()
            
    def get_data(self):
        """데이터 조회"""
        with self.update_lock:
            return self.district_data, self.district_analysis, self.last_update
            
    def save_optimization(self, key: str, result: dict):
        """최적화 결과 저장"""
        self.optimization_results[key] = {
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_optimization_history(self):
        """최적화 이력 조회"""
        return self.optimization_results

# 데이터 저장소 인스턴스
data_store = DataStore()

# API 키 (환경변수에서 가져오거나 기본값 사용)
API_KEY = os.environ.get('SEOUL_API_KEY', '6464716442737069363863566b466c')

# ===========================================================================
# 백그라운드 작업
# ===========================================================================

def background_data_update():
    """백그라운드에서 주기적으로 데이터 업데이트"""
    while True:
        try:
            # 30분마다 데이터 업데이트
            district_data = get_bike_station_data_by_district(API_KEY)
            district_analysis = analyze_district_redistribution_needs(district_data)
            data_store.update_data(district_data, district_analysis)
            print(f"[{datetime.now()}] 데이터 업데이트 완료")
        except Exception as e:
            print(f"[{datetime.now()}] 데이터 업데이트 실패: {e}")
        
        time.sleep(1800)  # 30분 대기

# ===========================================================================
# 라우트
# ===========================================================================

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('dashboard.html')

@app.route('/classic')
def classic():
    """기존 버전 페이지"""
    return render_template('index_new.html')

@app.route('/api/status')
def api_status():
    """API 상태 확인"""
    _, _, last_update = data_store.get_data()
    return jsonify({
        'status': 'online',
        'last_update': last_update.isoformat() if last_update else None,
        'version': '2.0.0'
    })

@app.route('/api/data/collect', methods=['POST'])
def collect_data():
    """데이터 수집 및 분석"""
    try:
        # 즉시 데이터 수집
        district_data = get_bike_station_data_by_district(API_KEY)
        district_analysis = analyze_district_redistribution_needs(district_data)
        data_store.update_data(district_data, district_analysis)
        
        # 응답 데이터 준비
        districts_summary = []
        for district_name, analysis in district_analysis.items():
            districts_summary.append({
                'id': district_name,
                'name': district_name,
                'total_stations': analysis['total_stations'],
                'pickup_needed': len(analysis['pickup_needed']),
                'delivery_needed': len(analysis['delivery_needed']),
                'total_imbalance': analysis['total_imbalance'],
                'urgency_score': analysis['urgency_score'],
                'pickup_stations': analysis['pickup_needed'],
                'delivery_stations': analysis['delivery_needed']
            })
        
        # 긴급도 순 정렬
        districts_summary.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': {
                'districts': districts_summary,
                'total_districts': len(districts_summary),
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_stations': sum(d['total_stations'] for d in districts_summary),
                    'total_pickup_needed': sum(d['pickup_needed'] for d in districts_summary),
                    'total_delivery_needed': sum(d['delivery_needed'] for d in districts_summary)
                }
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/districts')
@cache.cached(timeout=60)  # 1분 캐시
def get_districts():
    """구별 데이터 조회"""
    _, district_analysis, last_update = data_store.get_data()
    
    if not district_analysis:
        return jsonify({
            'success': False,
            'error': '데이터가 없습니다. 먼저 데이터를 수집해주세요.'
        }), 404
    
    districts_summary = []
    for district_name, analysis in district_analysis.items():
        districts_summary.append({
            'id': district_name,
            'name': district_name,
            'total_stations': analysis['total_stations'],
            'pickup_needed': len(analysis['pickup_needed']),
            'delivery_needed': len(analysis['delivery_needed']),
            'total_imbalance': analysis['total_imbalance'],
            'urgency_score': analysis['urgency_score']
        })
    
    districts_summary.sort(key=lambda x: x['urgency_score'], reverse=True)
    
    return jsonify({
        'success': True,
        'data': districts_summary,
        'last_update': last_update.isoformat() if last_update else None
    })

@app.route('/api/district/<district_id>')
def get_district_detail(district_id):
    """특정 구 상세 정보"""
    _, district_analysis, _ = data_store.get_data()
    
    if district_id not in district_analysis:
        return jsonify({
            'success': False,
            'error': '해당 구를 찾을 수 없습니다.'
        }), 404
    
    analysis = district_analysis[district_id]
    
    # 구 중심점 가져오기
    classifier = SeoulDistrictClassifier()
    center = classifier.district_centers.get(district_id, (37.5665, 126.9780))
    
    return jsonify({
        'success': True,
        'data': {
            'id': district_id,
            'name': district_id,
            'center': {'lat': center[0], 'lon': center[1]},
            'total_stations': analysis['total_stations'],
            'pickup_needed': len(analysis['pickup_needed']),
            'delivery_needed': len(analysis['delivery_needed']),
            'total_imbalance': analysis['total_imbalance'],
            'urgency_score': analysis['urgency_score'],
            'pickup_stations': analysis['pickup_needed'],
            'delivery_stations': analysis['delivery_needed']
        }
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_route():
    """경로 최적화"""
    try:
        data = request.get_json()
        district_id = data.get('district_id')
        num_vehicles = int(data.get('num_vehicles', 2))
        vehicle_capacity = int(data.get('vehicle_capacity', 20))
        use_clustering = data.get('use_clustering', True)
        
        _, district_analysis, _ = data_store.get_data()
        
        if district_id not in district_analysis:
            return jsonify({
                'success': False,
                'error': '해당 구를 찾을 수 없습니다.'
            }), 404
        
        analysis = district_analysis[district_id]
        
        print(f"\n[DEBUG] Optimizing district: {district_id}")
        print(f"[DEBUG] Pickup needed: {len(analysis['pickup_needed'])} stations")
        print(f"[DEBUG] Delivery needed: {len(analysis['delivery_needed'])} stations")
        if analysis['pickup_needed']:
            print(f"[DEBUG] First pickup station: {analysis['pickup_needed'][0]}")
        
        # 최적화 실행
        if use_clustering and len(analysis['pickup_needed'] + analysis['delivery_needed']) > 30:
            solution = solve_district_with_clustering(
                district_id, analysis, num_vehicles, vehicle_capacity
            )
        else:
            problem_stations = analysis['pickup_needed'] + analysis['delivery_needed']
            solution = solve_single_cluster_with_ortools(
                district_id, problem_stations, num_vehicles, vehicle_capacity
            )
        
        if not solution:
            return jsonify({
                'success': False,
                'error': '최적화에 실패했습니다.'
            }), 500
        
        print(f"\n[DEBUG] Solution received:")
        print(f"[DEBUG] Routes count: {len(solution.get('routes', []))}")
        if solution.get('routes'):
            for i, route in enumerate(solution['routes']):
                print(f"[DEBUG] Route {i}: {len(route.get('path', []))} stops")
                if route.get('path'):
                    print(f"[DEBUG] First stop: {route['path'][0]}")
        
        # Ensure all route paths have lat/lon coordinates
        if solution.get('routes'):
            for route_idx, route in enumerate(solution['routes']):
                if route.get('path'):
                    print(f"\n[DEBUG] Processing route {route_idx} with {len(route['path'])} stops")
                    for stop_idx, stop in enumerate(route['path']):
                        # Make sure each stop has lat and lon
                        if 'lat' not in stop or stop['lat'] is None:
                            print(f"[DEBUG] Missing lat for stop {stop_idx}: {stop.get('name')}")
                            stop['lat'] = 37.5665  # Default Seoul center
                        if 'lon' not in stop or stop['lon'] is None:
                            print(f"[DEBUG] Missing lon for stop {stop_idx}: {stop.get('name')}")
                            stop['lon'] = 126.9780  # Default Seoul center
                        # Ensure numeric values
                        stop['lat'] = float(stop['lat']) if stop['lat'] else 37.5665
                        stop['lon'] = float(stop['lon']) if stop['lon'] else 126.9780
                        print(f"[DEBUG] Stop {stop_idx}: {stop['name']} at ({stop['lat']}, {stop['lon']})")
                else:
                    print(f"[DEBUG] Route {route_idx} has no path!")
        
        # 결과 저장
        optimization_key = f"{district_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data_store.save_optimization(optimization_key, solution)
        
        # 응답 준비
        response_data = {
            'optimization_id': optimization_key,
            'district_id': district_id,
            'total_distance': solution.get('total_distance', 0),
            'routes': solution.get('routes', []),
            'method': solution.get('method', 'Unknown'),
            'clustering_used': solution.get('clustering_used', False),
            'num_clusters': solution.get('num_clusters', 0),
            'parameters': {
                'num_vehicles': num_vehicles,
                'vehicle_capacity': vehicle_capacity,
                'use_clustering': use_clustering
            }
        }
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        import traceback
        print(f"Optimization error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/<format>')
def export_data(format):
    """데이터 내보내기"""
    _, district_analysis, last_update = data_store.get_data()
    
    if not district_analysis:
        return jsonify({
            'success': False,
            'error': '내보낼 데이터가 없습니다.'
        }), 404
    
    if format == 'json':
        # JSON 형식으로 내보내기
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'districts': district_analysis
        }
        
        return jsonify(export_data)
    
    elif format == 'csv':
        # CSV 형식으로 내보내기
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 헤더
        writer.writerow([
            '구', '전체대여소', '수거필요', '배송필요', 
            '총불균형', '긴급도점수'
        ])
        
        # 데이터
        for district, analysis in district_analysis.items():
            writer.writerow([
                district,
                analysis['total_stations'],
                len(analysis['pickup_needed']),
                len(analysis['delivery_needed']),
                analysis['total_imbalance'],
                analysis['urgency_score']
            ])
        
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'bike_reallocation_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    
    else:
        return jsonify({
            'success': False,
            'error': '지원하지 않는 형식입니다.'
        }), 400

@app.route('/api/history')
def get_history():
    """최적화 이력 조회"""
    history = data_store.get_optimization_history()
    
    return jsonify({
        'success': True,
        'data': history
    })

@app.route('/api/statistics')
@cache.cached(timeout=300)  # 5분 캐시
def get_statistics():
    """통계 정보"""
    _, district_analysis, last_update = data_store.get_data()
    
    if not district_analysis:
        return jsonify({
            'success': False,
            'error': '데이터가 없습니다.'
        }), 404
    
    # 통계 계산
    total_stations = sum(a['total_stations'] for a in district_analysis.values())
    total_pickup = sum(len(a['pickup_needed']) for a in district_analysis.values())
    total_delivery = sum(len(a['delivery_needed']) for a in district_analysis.values())
    total_imbalance = sum(a['total_imbalance'] for a in district_analysis.values())
    
    # 상위 5개 긴급 구
    urgent_districts = sorted(
        district_analysis.items(),
        key=lambda x: x[1]['urgency_score'],
        reverse=True
    )[:5]
    
    return jsonify({
        'success': True,
        'data': {
            'total_districts': len(district_analysis),
            'total_stations': total_stations,
            'total_pickup_needed': total_pickup,
            'total_delivery_needed': total_delivery,
            'total_imbalance': total_imbalance,
            'urgent_districts': [
                {
                    'name': d[0],
                    'urgency_score': d[1]['urgency_score']
                } for d in urgent_districts
            ],
            'last_update': last_update.isoformat() if last_update else None
        }
    })

# ===========================================================================
# 에러 핸들러
# ===========================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': '요청한 리소스를 찾을 수 없습니다.'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': '서버 내부 오류가 발생했습니다.'
    }), 500

# ===========================================================================
# 앱 실행
# ===========================================================================

if __name__ == '__main__':
    # 백그라운드 데이터 업데이트 스레드 시작
    update_thread = threading.Thread(target=background_data_update, daemon=True)
    update_thread.start()
    
    # Flask 앱 실행
    app.run(debug=True, host='0.0.0.0', port=8080)
