from flask import Flask, render_template, request, jsonify, session
import requests
import json
import math
import time
from collections import defaultdict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# 세션 영구화 설정

# ---------------------------------------------------------------------------
# 1. GeoJSON 기반 구 분류기
# ---------------------------------------------------------------------------
class SeoulDistrictClassifier:
    """서울시 구 분류기 (GeoJSON 활용)"""
    def __init__(self):
        self.district_polygons = {}
        self.load_geojson()
    
    def load_geojson(self):
        """GeoJSON 데이터 로드 (온라인 또는 로컬)"""
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

# ---------------------------------------------------------------------------
# 2. 클러스터링 모듈
# ---------------------------------------------------------------------------
class BikeStationClusterer:
    """따릉이 대여소 클러스터링"""
    
    def __init__(self, num_vehicles=3):
        self.num_vehicles = num_vehicles
        self.clusters = {}
        
    def create_balanced_clusters(self, stations):
        """작업량 균형을 고려한 클러스터 생성"""
        
        # 수거/배송 필요 대여소 분리
        pickup_stations = []
        delivery_stations = []
        
        for station in stations:
            if station.get('pickup', 0) > 0:
                pickup_stations.append(station)
            elif station.get('delivery', 0) > 0:
                delivery_stations.append(station)
        
        # 1차 클러스터링: 수거/배송 별도
        pickup_clusters = self._kmeans_clustering(
            pickup_stations, 
            n_clusters=min(self.num_vehicles, len(pickup_stations))
        ) if pickup_stations else {}
        
        delivery_clusters = self._kmeans_clustering(
            delivery_stations,
            n_clusters=min(self.num_vehicles, len(delivery_stations))
        ) if delivery_stations else {}
        
        # 2차 클러스터링: 인접 클러스터 병합
        final_clusters = self._merge_clusters(pickup_clusters, delivery_clusters)
        
        return final_clusters
    
    def _kmeans_clustering(self, stations, n_clusters):
        """K-means 클러스터링 수행"""
        if not stations or n_clusters == 0:
            return {}
            
        # 좌표와 작업량을 특징으로 사용
        features = []
        for station in stations:
            features.append([
                station['lat'],
                station['lon'],
                station.get('pickup', 0) + station.get('delivery', 0)  # 작업량
            ])
        
        # 정규화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # 클러스터별로 스테이션 그룹화
        clusters = defaultdict(list)
        for station, label in zip(stations, labels):
            clusters[label].append(station)
        
        return dict(clusters)
    
    def _merge_clusters(self, pickup_clusters, delivery_clusters):
        """수거/배송 클러스터를 지리적 근접성 기반으로 병합"""
        final_clusters = []
        
        # 각 클러스터의 중심점 계산
        pickup_centers = {}
        for idx, stations in pickup_clusters.items():
            if stations:
                avg_lat = np.mean([s['lat'] for s in stations])
                avg_lon = np.mean([s['lon'] for s in stations])
                pickup_centers[f'pickup_{idx}'] = {
                    'center': (avg_lat, avg_lon),
                    'stations': stations
                }
        
        delivery_centers = {}
        for idx, stations in delivery_clusters.items():
            if stations:
                avg_lat = np.mean([s['lat'] for s in stations])
                avg_lon = np.mean([s['lon'] for s in stations])
                delivery_centers[f'delivery_{idx}'] = {
                    'center': (avg_lat, avg_lon),
                    'stations': stations
                }
        
        # 가장 가까운 수거/배송 클러스터 페어링
        used_delivery = set()
        
        for pickup_key, pickup_data in pickup_centers.items():
            cluster_stations = pickup_data['stations'].copy()
            
            # 가장 가까운 배송 클러스터 찾기
            min_dist = float('inf')
            closest_delivery = None
            
            for delivery_key, delivery_data in delivery_centers.items():
                if delivery_key not in used_delivery:
                    dist = self._calculate_distance(
                        pickup_data['center'][0], pickup_data['center'][1],
                        delivery_data['center'][0], delivery_data['center'][1]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_delivery = delivery_key
            
            if closest_delivery:
                cluster_stations.extend(delivery_centers[closest_delivery]['stations'])
                used_delivery.add(closest_delivery)
            
            final_clusters.append(cluster_stations)
        
        # 페어링되지 않은 배송 클러스터 추가
        for delivery_key, delivery_data in delivery_centers.items():
            if delivery_key not in used_delivery:
                final_clusters.append(delivery_data['stations'])
        
        # 빈 클러스터가 있으면 트럭 수에 맞게 조정
        while len(final_clusters) < self.num_vehicles:
            final_clusters.append([])
        
        return final_clusters
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """두 지점 간 거리 계산 (km)"""
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

# ---------------------------------------------------------------------------
# 3. 데이터 수집 및 구별 분류
# ---------------------------------------------------------------------------
def get_bike_station_data_by_district(api_key):
    """따릉이 데이터를 수집하고 구별로 분류합니다"""
    
    all_stations = []
    for start_index in [1, 1001, 2001]:
        end_index = start_index + 999
        url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{start_index}/{end_index}/"
        try:
            response = requests.get(url)
            data = response.json()
            if 'rentBikeStatus' in data and 'row' in data['rentBikeStatus']:
                all_stations.extend(data['rentBikeStatus']['row'])
        except:
            continue
    
    classifier = SeoulDistrictClassifier()
    district_stations = defaultdict(list)
    
    for station in all_stations:
        try:
            lat = float(station['stationLatitude'])
            lon = float(station['stationLongitude'])
            district = classifier.find_district(lat, lon)
            
            if district:
                station['district'] = district
                district_stations[district].append(station)
        except:
            continue
    
    return district_stations

# ---------------------------------------------------------------------------
# 4. 구별 재배치 분석
# ---------------------------------------------------------------------------
def analyze_district_redistribution_needs(district_stations):
    """구별 재배치 필요도를 분석합니다"""
    
    district_analysis = {}
    
    for district, stations in district_stations.items():
        pickup_needed = []
        delivery_needed = []
        total_imbalance = 0
        
        for station in stations:
            try:
                racks = int(station['rackTotCnt'])
                bikes = int(station['parkingBikeTotCnt'])
                
                if racks == 0:
                    continue
                
                occupancy = int(station['shared'])
                target = racks * 0.7
                imbalance = bikes - target
                
                station_info = {
                    'id': station['stationId'],
                    'name': station['stationName'],
                    'lat': float(station['stationLatitude']),
                    'lon': float(station['stationLongitude']),
                    'bikes': bikes,
                    'racks': racks,
                    'occupancy': occupancy,
                    'imbalance': abs(imbalance)
                }
                
                if occupancy >= 130:
                    station_info['pickup'] = min(int(imbalance), 10)
                    station_info['delivery'] = 0
                    pickup_needed.append(station_info)
                    total_imbalance += station_info['pickup']
                elif occupancy <= 30:
                    station_info['pickup'] = 0
                    station_info['delivery'] = min(int(abs(imbalance)), 10)
                    delivery_needed.append(station_info)
                    total_imbalance += station_info['delivery']
                    
            except:
                continue
        
        district_analysis[district] = {
            'total_stations': len(stations),
            'pickup_needed': pickup_needed,
            'delivery_needed': delivery_needed,
            'total_imbalance': total_imbalance,
            'urgency_score': len(pickup_needed) + len(delivery_needed)
        }
    
    return district_analysis

# ---------------------------------------------------------------------------
# 5. 클러스터 기반 OR-Tools 최적화
# ---------------------------------------------------------------------------
def solve_district_with_clustering(district_name, analysis, num_vehicles=2, vehicle_capacity=20):
    """클러스터링 기반 구별 재배치 최적화"""
    
    problem_stations = analysis['pickup_needed'] + analysis['delivery_needed']
    
    if len(problem_stations) == 0:
        return None
    
    # 노드 수가 적으면 클러스터링 없이 직접 처리
    if len(problem_stations) <= 30:
        return solve_single_cluster_with_ortools(
            district_name, problem_stations, num_vehicles, vehicle_capacity
        )
    
    # 클러스터링 수행
    clusterer = BikeStationClusterer(num_vehicles)
    clusters = clusterer.create_balanced_clusters(problem_stations)
    
    # 각 클러스터별로 OR-Tools 적용
    all_routes = []
    total_distance = 0
    solution_methods = []
    
    for i, cluster_stations in enumerate(clusters):
        if not cluster_stations:
            continue
        
        # 단일 트럭으로 클러스터 해결
        solution = solve_single_cluster_with_ortools(
            district_name,
            cluster_stations,
            num_vehicles=1,
            vehicle_capacity=vehicle_capacity,
            cluster_id=i+1
        )
        
        if solution and solution['routes']:
            # 트럭 ID 조정
            for route in solution['routes']:
                route['vehicle_id'] = i
                route['cluster_id'] = i+1
            all_routes.extend(solution['routes'])
            total_distance += solution.get('total_distance', 0)
            solution_methods.append(solution.get('method', 'Unknown'))
    
    return {
        'routes': all_routes,
        'total_distance': total_distance,
        'clustering_used': True,
        'num_clusters': len(clusters),
        'solution_methods': solution_methods
    }

def solve_single_cluster_with_ortools(district_name, stations, num_vehicles=1, vehicle_capacity=20, cluster_id=None):
    """단일 클러스터에 대한 OR-Tools 최적화"""
    
    if not stations:
        return None
    
    # 구별 고정 차고지 사용
    classifier = SeoulDistrictClassifier()
    actual_district = district_name.split('_')[0] if '_' in district_name else district_name
    
    if actual_district in classifier.district_centers:
        depot_lat, depot_lon = classifier.district_centers[actual_district]
    else:
        depot_lat = np.mean([s['lat'] for s in stations])
        depot_lon = np.mean([s['lon'] for s in stations])
    
    depot = {
        'id': f'DEPOT_{actual_district}',
        'name': f'{actual_district} 차고지',
        'lat': depot_lat,
        'lon': depot_lon,
        'pickup': 0,
        'delivery': 0
    }
    
    nodes = [depot] + stations
    pickups = [0] + [s.get('pickup', 0) for s in stations]
    deliveries = [0] + [s.get('delivery', 0) for s in stations]
    
    # 거리 행렬 계산
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return int(R * c * 1000)
    
    distance_matrix = []
    for from_node in nodes:
        row = []
        for to_node in nodes:
            dist = haversine(from_node['lat'], from_node['lon'],
                           to_node['lat'], to_node['lon'])
            row.append(dist)
        distance_matrix.append(row)
    
    # OR-Tools 모델 생성
    manager = pywrapcp.RoutingIndexManager(len(nodes), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # 거리 콜백
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 용량 제약
    initial_load = min(vehicle_capacity // 2, sum(deliveries))
    
    def capacity_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return pickups[from_node] - deliveries[from_node]
    
    capacity_callback_index = routing.RegisterUnaryTransitCallback(capacity_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        capacity_callback_index,
        0,
        [vehicle_capacity] * num_vehicles,
        False,
        "Capacity"
    )
    
    # 각 차량의 초기 적재량 설정
    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        capacity_dimension.SetCumulVarSoftLowerBound(index, initial_load, 1000)
        capacity_dimension.SetCumulVarSoftUpperBound(index, initial_load, 1000)
    
    # 거리 균등화
    routing.AddDimension(
        transit_callback_index,
        0,
        50000,
        True,
        "Distance"
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # 탐색 파라미터
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.time_limit.FromSeconds(30)
    
    # 문제 해결
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        result = extract_solution(manager, routing, solution, nodes, pickups, deliveries, num_vehicles)
        result['method'] = 'OR-Tools'
        return result
    else:
        result = solve_with_heuristic(district_name, stations, num_vehicles, vehicle_capacity, depot)
        result['method'] = 'Heuristic'
        return result

# ---------------------------------------------------------------------------
# 6. 솔루션 추출
# ---------------------------------------------------------------------------
def extract_solution(manager, routing, solution, nodes, pickups, deliveries, num_vehicles):
    """OR-Tools 솔루션에서 경로 정보를 추출합니다"""
    
    routes = []
    total_distance = 0
    
    for vehicle_id in range(num_vehicles):
        route = {
            'vehicle_id': vehicle_id,
            'path': [],
            'distance': 0,
            'pickups': 0,
            'deliveries': 0,
            'load_changes': []
        }
        
        index = routing.Start(vehicle_id)
        current_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            node = nodes[node_index]
            pickup = pickups[node_index]
            delivery = deliveries[node_index]
            current_load += pickup - delivery
            
            route['path'].append({
                'name': node['name'],
                'pickup': pickup,
                'delivery': delivery,
                'current_load': current_load,
                'lat': node['lat'],
                'lon': node['lon']
            })
            
            route['pickups'] += pickup
            route['deliveries'] += delivery
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route['distance'] += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        node_index = manager.IndexToNode(index)
        route['path'].append({
            'name': nodes[node_index]['name'],
            'pickup': 0,
            'delivery': 0,
            'current_load': current_load,
            'lat': nodes[node_index]['lat'],
            'lon': nodes[node_index]['lon']
        })
        
        if len(route['path']) > 2:
            routes.append(route)
            total_distance += route['distance']
    
    return {
        'routes': routes,
        'total_distance': total_distance
    }

# ---------------------------------------------------------------------------
# 7. 휴리스틱 솔버
# ---------------------------------------------------------------------------
def solve_with_heuristic(district_name, stations, num_vehicles, vehicle_capacity, depot=None):
    """대규모 문제를 위한 휴리스틱 해법"""
    
    if depot is None:
        classifier = SeoulDistrictClassifier()
        actual_district = district_name.split('_')[0] if '_' in district_name else district_name
        if actual_district in classifier.district_centers:
            depot_lat, depot_lon = classifier.district_centers[actual_district]
        else:
            depot_lat = np.mean([s['lat'] for s in stations]) if stations else 37.5665
            depot_lon = np.mean([s['lon'] for s in stations]) if stations else 126.9780
        
        depot = {
            'name': f'{actual_district} 차고지',
            'lat': depot_lat,
            'lon': depot_lon
        }
    
    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c * 1000
    
    pickup_stations = [s for s in stations if s.get('pickup', 0) > 0]
    delivery_stations = [s for s in stations if s.get('delivery', 0) > 0]
    
    pickup_stations.sort(key=lambda x: x.get('pickup', 0), reverse=True)
    delivery_stations.sort(key=lambda x: x.get('delivery', 0), reverse=True)
    
    routes = []
    total_distance = 0
    
    for vehicle_id in range(num_vehicles):
        route = {
            'vehicle_id': vehicle_id,
            'path': [{'name': depot['name'], 'pickup': 0, 'delivery': 0, 'current_load': 0, 'lat': depot['lat'], 'lon': depot['lon']}],
            'distance': 0,
            'pickups': 0,
            'deliveries': 0
        }
        
        current_load = 0
        current_lat, current_lon = depot['lat'], depot['lon']
        
        # 수거 작업
        for station in pickup_stations[vehicle_id::num_vehicles]:
            if current_load + station['pickup'] <= vehicle_capacity:
                dist = calculate_distance(current_lat, current_lon, station['lat'], station['lon'])
                route['distance'] += dist
                
                current_load += station['pickup']
                current_lat, current_lon = station['lat'], station['lon']
                
                route['path'].append({
                    'name': station['name'],
                    'pickup': station['pickup'],
                    'delivery': 0,
                    'current_load': current_load,
                    'lat': station['lat'],
                    'lon': station['lon']
                })
                route['pickups'] += station['pickup']
        
        # 배송 작업
        for station in delivery_stations[vehicle_id::num_vehicles]:
            if current_load >= station['delivery']:
                dist = calculate_distance(current_lat, current_lon, station['lat'], station['lon'])
                route['distance'] += dist
                
                current_load -= station['delivery']
                current_lat, current_lon = station['lat'], station['lon']
                
                route['path'].append({
                    'name': station['name'],
                    'pickup': 0,
                    'delivery': station['delivery'],
                    'current_load': current_load,
                    'lat': station['lat'],
                    'lon': station['lon']
                })
                route['deliveries'] += station['delivery']
        
        # 차고지 복귀 거리 추가
        if len(route['path']) > 1:
            dist = calculate_distance(current_lat, current_lon, depot['lat'], depot['lon'])
            route['distance'] += dist
        
        route['path'].append({
            'name': depot['name'],
            'pickup': 0,
            'delivery': 0,
            'current_load': current_load,
            'lat': depot['lat'],
            'lon': depot['lon']
        })
        
        if len(route['path']) > 2:
            routes.append(route)
            total_distance += route['distance']
    
    return {
        'routes': routes, 
        'total_distance': total_distance,
        'method': 'Heuristic'
    }

# ---------------------------------------------------------------------------
# Flask 라우트
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/districts')
def get_districts():
    """구별 분석 데이터를 반환합니다"""
    try:
        API_KEY = "6464716442737069363863566b466c"
        district_stations = get_bike_station_data_by_district(API_KEY)
        district_analysis = analyze_district_redistribution_needs(district_stations)
        
        # 세션에 저장 (영구 세션으로 설정)
        session.permanent = True
        session['district_analysis'] = district_analysis
        
        print(f"DEBUG: 세션에 {len(district_analysis)}개 구 데이터 저장됨")
        print(f"DEBUG: 세션 키: {list(session.keys())}")
        
        # 프론트엔드용으로 데이터 정리
        districts_data = []
        for district, analysis in district_analysis.items():
            districts_data.append({
                'name': district,
                'total_stations': analysis['total_stations'],
                'pickup_needed': len(analysis['pickup_needed']),
                'delivery_needed': len(analysis['delivery_needed']),
                'total_imbalance': analysis['total_imbalance'],
                'urgency_score': analysis['urgency_score']
            })
        
        # 긴급도 순으로 정렬
        districts_data.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'districts': districts_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"ERROR: 데이터 수집 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/solve', methods=['POST'])
def solve_district():
    """특정 구의 재배치 문제를 해결합니다"""
    try:
        data = request.get_json()
        district_name = data.get('district')
        num_vehicles = int(data.get('num_vehicles', 2))
        vehicle_capacity = int(data.get('vehicle_capacity', 20))
        use_clustering = data.get('use_clustering', True)
        
        print(f"DEBUG: 최적화 요청 - 구: {district_name}, 트럭: {num_vehicles}, 용량: {vehicle_capacity}")
        print(f"DEBUG: 현재 세션 키: {list(session.keys())}")
        
        # 세션에서 분석 데이터 가져오기
        district_analysis = session.get('district_analysis')
        print(f"DEBUG: 세션에서 가져온 데이터: {type(district_analysis)}")
        
        if district_analysis:
            print(f"DEBUG: 저장된 구 목록: {list(district_analysis.keys())}")
        
        if not district_analysis or district_name not in district_analysis:
            return jsonify({
                'success': False,
                'error': '구별 분석 데이터를 찾을 수 없습니다. 먼저 데이터를 수집해주세요.'
            })
        
        analysis = district_analysis[district_name]
        print(f"DEBUG: {district_name} 분석 데이터 로드 완료")
        
        if use_clustering:
            solution = solve_district_with_clustering(
                district_name, analysis, num_vehicles, vehicle_capacity
            )
        else:
            problem_stations = analysis['pickup_needed'] + analysis['delivery_needed']
            solution = solve_single_cluster_with_ortools(
                district_name, problem_stations, num_vehicles, vehicle_capacity
            )
        
        if solution:
            return jsonify({
                'success': True,
                'solution': solution,
                'district': district_name,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': '해당 구에는 재배치가 필요하지 않습니다.'
            })
            
    except Exception as e:
        print(f"ERROR: 최적화 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/debug/session')
def debug_session():
    """세션 상태를 디버깅합니다"""
    try:
        session_data = {
            'keys': list(session.keys()),
            'district_analysis_keys': list(session.get('district_analysis', {}).keys()) if session.get('district_analysis') else [],
            'session_permanent': session.permanent,
            'session_modified': session.modified
        }
        return jsonify({
            'success': True,
            'session_info': session_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/stations/<district>')
def get_district_stations(district):
    """특정 구의 대여소 정보를 반환합니다"""
    try:
        district_analysis = session.get('district_analysis')
        if not district_analysis or district not in district_analysis:
            return jsonify({
                'success': False,
                'error': '구별 분석 데이터를 찾을 수 없습니다.'
            })
        
        analysis = district_analysis[district]
        
        # 모든 대여소 정보 수집
        all_stations = []
        
        # 수거 필요 대여소
        for station in analysis['pickup_needed']:
            station['type'] = 'pickup'
            all_stations.append(station)
        
        # 배송 필요 대여소
        for station in analysis['delivery_needed']:
            station['type'] = 'delivery'
            all_stations.append(station)
        
        return jsonify({
            'success': True,
            'stations': all_stations,
            'district': district
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
