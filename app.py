from flask import Flask, render_template, jsonify, request
import requests
import json
import folium
from folium import plugins
from collections import defaultdict
import math
from datetime import datetime
import os

app = Flask(__name__)

# API 키 설정
API_KEY = "6464716842737069363863566b466c"

class SeoulDistrictClassifier:
    """서울시 구 분류기 (간소화 버전)"""
    
    def __init__(self):
        self.district_bounds = {
            '강남구': {'min_lat': 37.4687, 'max_lat': 37.5687, 'min_lon': 127.0164, 'max_lon': 127.0964},
            '강동구': {'min_lat': 37.5201, 'max_lat': 37.5501, 'min_lon': 127.1138, 'max_lon': 127.1438},
            '강북구': {'min_lat': 37.6296, 'max_lat': 37.6596, 'min_lon': 127.0157, 'max_lon': 127.0457},
            '강서구': {'min_lat': 37.5409, 'max_lat': 37.5709, 'min_lon': 126.8295, 'max_lon': 126.8795},
            '관악구': {'min_lat': 37.4684, 'max_lat': 37.4984, 'min_lon': 126.9416, 'max_lon': 126.9716},
            '광진구': {'min_lat': 37.5284, 'max_lat': 37.5584, 'min_lon': 127.0722, 'max_lon': 127.1022},
            '구로구': {'min_lat': 37.4854, 'max_lat': 37.5154, 'min_lon': 126.8774, 'max_lon': 126.9074},
            '금천구': {'min_lat': 37.4467, 'max_lat': 37.4767, 'min_lon': 126.8854, 'max_lon': 126.9154},
            '노원구': {'min_lat': 37.6442, 'max_lat': 37.6742, 'min_lon': 127.0468, 'max_lon': 127.0768},
            '도봉구': {'min_lat': 37.6587, 'max_lat': 37.6887, 'min_lon': 127.0371, 'max_lon': 127.0671},
            '동대문구': {'min_lat': 37.5644, 'max_lat': 37.5944, 'min_lon': 127.0299, 'max_lon': 127.0599},
            '동작구': {'min_lat': 37.5024, 'max_lat': 37.5324, 'min_lon': 126.9293, 'max_lon': 126.9593},
            '마포구': {'min_lat': 37.5537, 'max_lat': 37.5837, 'min_lon': 126.8987, 'max_lon': 126.9287},
            '서대문구': {'min_lat': 37.5691, 'max_lat': 37.5991, 'min_lon': 126.9268, 'max_lon': 126.9568},
            '서초구': {'min_lat': 37.4737, 'max_lat': 37.5037, 'min_lon': 127.0225, 'max_lon': 127.0525},
            '성동구': {'min_lat': 37.5533, 'max_lat': 37.5833, 'min_lon': 127.0269, 'max_lon': 127.0569},
            '성북구': {'min_lat': 37.5794, 'max_lat': 37.6094, 'min_lon': 127.0067, 'max_lon': 127.0367},
            '송파구': {'min_lat': 37.5046, 'max_lat': 37.5346, 'min_lon': 127.0950, 'max_lon': 127.1250},
            '양천구': {'min_lat': 37.5067, 'max_lat': 37.5367, 'min_lon': 126.8565, 'max_lon': 126.8865},
            '영등포구': {'min_lat': 37.5164, 'max_lat': 37.5464, 'min_lon': 126.8863, 'max_lon': 126.9163},
            '용산구': {'min_lat': 37.5284, 'max_lat': 37.5584, 'min_lon': 126.9554, 'max_lon': 126.9854},
            '은평구': {'min_lat': 37.6076, 'max_lat': 37.6376, 'min_lon': 126.9127, 'max_lon': 126.9427},
            '종로구': {'min_lat': 37.5635, 'max_lat': 37.5935, 'min_lon': 126.9691, 'max_lon': 126.9991},
            '중구': {'min_lat': 37.5540, 'max_lat': 37.5840, 'min_lon': 126.9879, 'max_lon': 127.0179},
            '중랑구': {'min_lat': 37.5963, 'max_lat': 37.6263, 'min_lon': 127.0825, 'max_lon': 127.1125}
        }
        
        self.district_centers = {
            '강남구': (37.5172, 127.0473), '강동구': (37.5301, 127.1238),
            '강북구': (37.6396, 127.0257), '강서구': (37.5509, 126.8495),
            '관악구': (37.4784, 126.9516), '광진구': (37.5384, 127.0822),
            '구로구': (37.4954, 126.8874), '금천구': (37.4567, 126.8954),
            '노원구': (37.6542, 127.0568), '도봉구': (37.6687, 127.0471),
            '동대문구': (37.5744, 127.0399), '동작구': (37.5124, 126.9393),
            '마포구': (37.5637, 126.9087), '서대문구': (37.5791, 126.9368),
            '서초구': (37.4837, 127.0325), '성동구': (37.5633, 127.0369),
            '성북구': (37.5894, 127.0167), '송파구': (37.5146, 127.1050),
            '양천구': (37.5167, 126.8665), '영등포구': (37.5264, 126.8963),
            '용산구': (37.5384, 126.9654), '은평구': (37.6176, 126.9227),
            '종로구': (37.5735, 126.9791), '중구': (37.5640, 126.9979),
            '중랑구': (37.6063, 127.0925)
        }
    
    def find_district(self, lat, lon):
        """좌표가 속한 구를 찾습니다"""
        for district, bounds in self.district_bounds.items():
            if (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
                bounds['min_lon'] <= lon <= bounds['max_lon']):
                return district
        
        min_distance = float('inf')
        closest_district = None
        
        for district, (center_lat, center_lon) in self.district_centers.items():
            distance = math.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            if distance < min_distance:
                min_distance = distance
                closest_district = district
        
        return closest_district

def get_bike_data():
    """따릉이 데이터 수집"""
    all_stations = []
    for start_index in [1, 1001, 2001]:
        end_index = start_index + 999
        url = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/bikeList/{start_index}/{end_index}/"
        try:
            response = requests.get(url)
            data = response.json()
            if 'rentBikeStatus' in data and 'row' in data['rentBikeStatus']:
                all_stations.extend(data['rentBikeStatus']['row'])
        except:
            continue
    return all_stations

def analyze_stations(stations):
    """대여소 분석"""
    classifier = SeoulDistrictClassifier()
    district_stations = defaultdict(list)
    
    for station in stations:
        try:
            lat = float(station['stationLatitude'])
            lon = float(station['stationLongitude'])
            district = classifier.find_district(lat, lon)
            
            if district:
                racks = int(station['rackTotCnt'])
                bikes = int(station['parkingBikeTotCnt'])
                
                if racks > 0:
                    occupancy = (bikes / racks) * 100
                    
                    station_info = {
                        'id': station['stationId'],
                        'name': station['stationName'],
                        'lat': lat,
                        'lon': lon,
                        'bikes': bikes,
                        'racks': racks,
                        'occupancy': occupancy,
                        'district': district,
                        'status': 'normal'
                    }
                    
                    if occupancy >= 80:
                        station_info['status'] = 'full'
                    elif occupancy <= 20:
                        station_info['status'] = 'empty'
                    
                    district_stations[district].append(station_info)
        except:
            continue
    
    return district_stations

def create_map(district_stations):
    """지도 생성"""
    # 서울 중심 좌표
    seoul_map = folium.Map(
        location=[37.5665, 126.9780],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # 상태별 색상
    colors = {
        'full': 'red',
        'empty': 'blue',
        'normal': 'green'
    }
    
    # 각 구별로 마커 그룹 생성
    for district, stations in district_stations.items():
        feature_group = folium.FeatureGroup(name=district)
        
        for station in stations:
            color = colors[station['status']]
            icon_name = 'bicycle'
            
            popup_text = f"""
            <b>{station['name']}</b><br>
            구: {station['district']}<br>
            자전거: {station['bikes']}대<br>
            거치대: {station['racks']}개<br>
            점유율: {station['occupancy']:.1f}%<br>
            상태: {station['status']}
            """
            
            folium.Marker(
                location=[station['lat'], station['lon']],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color=color, icon=icon_name, prefix='fa'),
                tooltip=f"{station['name']} ({station['bikes']}/{station['racks']})"
            ).add_to(feature_group)
        
        feature_group.add_to(seoul_map)
    
    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(seoul_map)
    
    # 히트맵 추가
    heat_data = []
    for district, stations in district_stations.items():
        for station in stations:
            if station['status'] in ['full', 'empty']:
                heat_data.append([station['lat'], station['lon'], station['occupancy']/100])
    
    if heat_data:
        plugins.HeatMap(heat_data, name='재배치 필요도').add_to(seoul_map)
    
    # 범례 추가
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 90px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px;
                ">
    <p style="margin: 10px;"><b>대여소 상태</b></p>
    <p style="margin: 10px;"><i class="fa fa-bicycle" style="color:red"></i> 포화 (80% 이상)</p>
    <p style="margin: 10px;"><i class="fa fa-bicycle" style="color:blue"></i> 부족 (20% 이하)</p>
    <p style="margin: 10px;"><i class="fa fa-bicycle" style="color:green"></i> 정상</p>
    </div>
    '''
    seoul_map.get_root().html.add_child(folium.Element(legend_html))
    
    return seoul_map

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """데이터 API"""
    stations = get_bike_data()
    district_stations = analyze_stations(stations)
    
    # 통계 계산
    stats = {}
    for district, stations in district_stations.items():
        full_count = sum(1 for s in stations if s['status'] == 'full')
        empty_count = sum(1 for s in stations if s['status'] == 'empty')
        
        stats[district] = {
            'total': len(stations),
            'full': full_count,
            'empty': empty_count,
            'normal': len(stations) - full_count - empty_count,
            'urgency': full_count + empty_count
        }
    
    return jsonify({
        'stations': dict(district_stations),
        'stats': stats,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/map')
def show_map():
    """지도 페이지"""
    stations = get_bike_data()
    district_stations = analyze_stations(stations)
    
    # 지도 생성
    seoul_map = create_map(district_stations)
    
    # HTML로 저장
    map_html = seoul_map._repr_html_()
    
    return render_template('map.html', map_html=map_html)

@app.route('/districts')
def districts():
    """구별 상세 페이지"""
    stations = get_bike_data()
    district_stations = analyze_stations(stations)
    
    # 구별 통계
    district_stats = []
    for district, stations in district_stations.items():
        full_count = sum(1 for s in stations if s['status'] == 'full')
        empty_count = sum(1 for s in stations if s['status'] == 'empty')
        
        total_bikes = sum(s['bikes'] for s in stations)
        total_racks = sum(s['racks'] for s in stations)
        avg_occupancy = (total_bikes / total_racks * 100) if total_racks > 0 else 0
        
        district_stats.append({
            'name': district,
            'total_stations': len(stations),
            'full_stations': full_count,
            'empty_stations': empty_count,
            'total_bikes': total_bikes,
            'total_racks': total_racks,
            'avg_occupancy': avg_occupancy,
            'urgency_score': full_count + empty_count
        })
    
    # 긴급도 순으로 정렬
    district_stats.sort(key=lambda x: x['urgency_score'], reverse=True)
    
    return render_template('districts.html', districts=district_stats)

if __name__ == '__main__':
    # templates 폴더가 없으면 생성
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("서버 시작: http://localhost:5001")
    app.run(debug=True, port=5001)