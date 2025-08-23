import requests
import json
import math
import time
from collections import defaultdict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from shapely.geometry import Point, shape
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. GeoJSON 기반 구 분류기
# ---------------------------------------------------------------------------
class SeoulDistrictClassifier:
    """서울시 구 분류기 (GeoJSON 활용)"""
    #생성자/객체의 초기 상태 설정 
    def __init__(self):
        self.district_polygons = {}
        self.load_geojson()
    
    def load_geojson(self):
        """GeoJSON 데이터 로드 (온라인 또는 로컬)"""
        # 실제 서비스에서는 정확한 GeoJSON 파일 사용
        # 여기서는 간소화된 경계 박스 사용
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
        
        # 구 중심점 (가장 가까운 구 찾기용)
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
        # 경계 박스 체크
        for district, bounds in self.district_bounds.items():
            if (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
                bounds['min_lon'] <= lon <= bounds['max_lon']):
                return district
        
        # 가장 가까운 구 찾기
        min_distance = float('inf')
        closest_district = None
        
        for district, (center_lat, center_lon) in self.district_centers.items():
            distance = math.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            if distance < min_distance:
                min_distance = distance
                closest_district = district
        
        return closest_district

# ---------------------------------------------------------------------------
# 2. 데이터 수집 및 구별 분류
# ---------------------------------------------------------------------------
def get_bike_station_data_by_district(api_key):
    """따릉이 데이터를 수집하고 구별로 분류합니다"""
    
    print("="*70)
    print("STEP 1: 데이터 수집 및 구별 분류")
    print("="*70)
    
    # 데이터 수집
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
    
    print(f"✓ 총 {len(all_stations)}개 대여소 데이터 수집 완료")
    
    # 구 분류기 초기화
    classifier = SeoulDistrictClassifier()
    
    # 구별 분류
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
    
    print(f"✓ {len(district_stations)}개 구로 분류 완료")
    
    # 구별 통계
    for district, stations in sorted(district_stations.items()):
        print(f"  - {district}: {len(stations)}개 대여소")
    
    return district_stations

# ---------------------------------------------------------------------------
# 3. 구별 재배치 분석
# ---------------------------------------------------------------------------
def analyze_district_redistribution_needs(district_stations):
    """구별 재배치 필요도를 분석합니다"""
    
    print("\n" + "="*70)
    print("STEP 2: 구별 재배치 필요도 분석")
    print("="*70)
    
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
                target = racks * 0.7  # 목표: 70%
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
                # 수거/배송 필요량 계산
                if occupancy >= 130:  # 수거 필요
                    station_info['pickup'] = min(int(imbalance), 10) # 최대 10대 수거
                    station_info['delivery'] = 0
                    pickup_needed.append(station_info)
                    total_imbalance += station_info['pickup']
                elif occupancy <= 30:  # 배송 필요
                    station_info['pickup'] = 0
                    station_info['delivery'] = min(int(abs(imbalance)), 10) # 최대 10대 배송
                    delivery_needed.append(station_info)
                    total_imbalance += station_info['delivery']
                    
            except:
                continue
        
        district_analysis[district] = {
            'total_stations': len(stations),
            'pickup_needed': pickup_needed,
            'delivery_needed': delivery_needed,
            'total_imbalance': total_imbalance,
            'urgency_score': len(pickup_needed) + len(delivery_needed) #수거가 필요한 대여소의 총개수와 배송이 필요한 대여소의 총개수를 합산하여 만든다 
        }
    
    # 우선순위 출력
    sorted_districts = sorted(district_analysis.items(), 
                            key=lambda x: x[1]['urgency_score'], 
                            reverse=True)
    
    print(f"\n{'구':<10} {'대여소':<8} {'수거필요':<10} {'배송필요':<10} {'긴급도':<8}")
    print("-"*50)
    
    for district, analysis in sorted_districts[:10]:
        print(f"{district:<10} {analysis['total_stations']:<8} "
              f"{len(analysis['pickup_needed']):<10} "
              f"{len(analysis['delivery_needed']):<10} "
              f"{analysis['urgency_score']:<8}")
    
    return district_analysis

# ---------------------------------------------------------------------------
# 4. OR-Tools를 사용한 구별 최적 경로 계산
# ---------------------------------------------------------------------------
def solve_district_with_ortools(district_name, analysis, num_vehicles=2, vehicle_capacity=20):
    """특정 구의 재배치 경로를 OR-Tools로 최적화합니다"""
    
    # 재배치 필요 대여소 추출
    problem_stations = analysis['pickup_needed'] + analysis['delivery_needed']
    
    if len(problem_stations) == 0:
        return None
    
    print(f"\n{district_name} 최적화 중...")
    print(f"  - 문제 크기: {len(problem_stations)}개 대여소")
    
    # 차고지 설정 (구 중심부)
    classifier = SeoulDistrictClassifier()
    center_lat, center_lon = classifier.district_centers.get(district_name, (37.5665, 126.9780))
    depot = {
        'id': f'DEPOT_{district_name}',
        'name': f'{district_name} 차고지',
        'lat': center_lat,
        'lon': center_lon,
        'pickup': 0,
        'delivery': 0
    }
    
    # 노드 리스트 생성
    nodes = [depot] + problem_stations
    
    # 수거/배송량 리스트
    pickups = [0] + [s.get('pickup', 0) for s in problem_stations]
    deliveries = [0] + [s.get('delivery', 0) for s in problem_stations]
    
    # 거리 행렬 계산/지구상 두 지점 사이의 거리를 계산하는 함수
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return int(R * c * 1000)
    #이중 for문으로 모든 노드 간의 거리를 계산하고 2차원 리스트로 만드는 과정 
    distance_matrix = []
    for from_node in nodes:
        row = []
        for to_node in nodes:
            dist = haversine(from_node['lat'], from_node['lon'], 
                           to_node['lat'], to_node['lon'])
            row.append(dist)
        distance_matrix.append(row)
    
    # 문제 크기에 따라 처리 방법 결정
    if len(nodes) > 300:
        # 큰 문제는 휴리스틱 사용
        return solve_with_heuristic(district_name, problem_stations, num_vehicles, vehicle_capacity)
    
    # OR-Tools 모델 생성
    manager = pywrapcp.RoutingIndexManager(len(nodes), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # 거리 콜백/OR-Tools의 임시 작업 번호(내부 인덱스)를 우리가 아는 고유한 주소(실제 인덱스)로 변환해서, 정확한 두 지점 사이의 거리를 계산하는 함수
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    # RegisterTransitCallback은 두 노드 사이의 거리를 계산하는 콜백 함수로, OR-Tools가 경로를 최적화할 때 사용한다.(계산기 등록)이동시 호출할 함수를 등록
    # 이 함수는 distance_callback 함수 자체를 돌려주는 것이 아니라 엔진 내부에 등록된 고유한 ID번호(transit_callback_index)를 반환한다.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    #계산기 지정: 모든 차량의 경로 비용을 계산하는 공식 평가 도구로 지정한다고 최종적으로 설정하는 과정 
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    #등록과 지정을 분리함으로써 다양한 평가 기준을 만들어 놓고 필요에 따라 선택적으로 사용할 수 있다.(거리뿐 아니라 이동시간을 계산 하는 time_ccallback함수도 만들 수 있다)



    # 용량 제약/ 차량의 적재 용량을 고려하여 수거/배송량 제약을 추가한다.
    def capacity_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return pickups[from_node] - deliveries[from_node]
    #계산기 등록: 차량의 적재 용량을 계산하는 콜백 함수를 등록한다.
    capacity_callback_index = routing.RegisterUnaryTransitCallback(capacity_callback)
    #규칙 설정 및 제약 추가 
    routing.AddDimensionWithVehicleCapacity(
        capacity_callback_index, #사용할 계산기 ID
        0, #대기 시간 등 여유 용량(여기선 0으로 설정)
        [vehicle_capacity] * num_vehicles, #각 차량이 가질 수 있는 최대 적재 용량을 리스트 형태로 지정
        True, #모든 차량은 차고지에서 출발할 때 적재량이 0인 상태에서 시작해야한다 
        "Capacity" #차량의 적재 용량을 관리하는 차원 이름 지정
    )

    # [개선] 트럭 간 이동 거리 균등화를 위한 제약 조건 추가
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        50000,  # vehicle maximum travel distance (50km) 
        True,  # start cumul to zero
        "Distance"
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    # GlobalSpanCost: 모든 차량의 경로 중 가장 긴 경로와 가장 짧은 경로의 차이에 페널티를 부과하여 경로 길이를 비슷하게 만듭니다.
    # 계수(100)가 클수록 균등화 효과가 커집니다.
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # 탐색 파라미터
    search_parameters = pywrapcp.DefaultRoutingSearchParameters() #표준 탐색 전략 설정
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC #PATH_CHEAPEST_ARC 전략은 Greedy방식이다 현재 위치에서 다음 노드로 이동할 때 가장 가까운 노드를 선택하는 방식이다.
    )
    # 첫번째 해답을 찾은 후 그 해답을 어떻게 개선해 나갈 것인지에 대한 고급 전략/현재 찾은 경로를 조금씩 변경 
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    #최적의 해답을 찾기 위해 최대 30초까지만 시간을 사용하겠다는 제한 시간 설정 
    search_parameters.time_limit.FromSeconds(120)
    
    # 문제 해결/ 위에 세운 제약 조건과 탐색 전략을 바탕으로 문제를 해결한다.
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution: #성공 케이스: extract_solution 함수를 호출하여 qhrwkqgks solution객체에서 경로,거리,수거/배송량 등의 정보를 추출하여 반환 
        return extract_solution(manager, routing, solution, nodes, pickups, deliveries, num_vehicles)
    else:
        print(f"  ⚠ OR-Tools 해결 실패, 휴리스틱 사용")
        return solve_with_heuristic(district_name, problem_stations, num_vehicles, vehicle_capacity)

# ---------------------------------------------------------------------------
# 5. 솔루션 추출 및 포맷팅
# ---------------------------------------------------------------------------
def extract_solution(manager, routing, solution, nodes, pickups, deliveries, num_vehicles):
    """OR-Tools 솔루션에서 경로 정보를 추출합니다"""
    
    #트럭의 최종 운행 계획을 담을 리스트를 만든다/ 트럭이이동한 총 거리를 합산하기 위한 변수를 0으로 초기화한다
    routes = []
    total_distance = 0
    #각 차량에 대해 경로를 추출한다/각 차량의 경로는 vehicle_id, 경로, 거리, 수거량, 배송량, 적재량 변화 등을 포함한다.
    for vehicle_id in range(num_vehicles):
        # 각 차량의 경로를 초기화한다/차량의 ID, 빈 경로, 거리, 수거량, 배송량, 적재량 변화 등을 초기화한다.
        route = {
            'vehicle_id': vehicle_id,
            'path': [],
            'distance': 0,
            'pickups': 0,
            'deliveries': 0,
            'load_changes': []
        }
        
        index = routing.Start(vehicle_id) #트럭의 출발점(차고지) 위치를 가져옴
        current_load = 0 #트럭의 현재 적재량을 0으로 초기화
        
        while not routing.IsEnd(index): #트럭이 최종 목적지에 도착할 때까지 계속해서 다음 목적지를 찾아 이동한다. routing.IsEnd(index)는 현재 인덱스가 종점인지 확인하는 함수이다.
            
            # 현재 노드 인덱스를 가져오고, 해당 노드의 정보(이름,수거/배송량)를 가져온다.
            node_index = manager.IndexToNode(index)
            node = nodes[node_index]
            pickup = pickups[node_index]
            delivery = deliveries[node_index]
            #트럭의 현재 적재량 업데이트
            current_load += pickup - delivery
            #현재 방문지의 정보를 경로에 기록 
            route['path'].append({
                'name': node['name'],
                'pickup': pickup,
                'delivery': delivery,
                'current_load': current_load
            })
            
            route['pickups'] += pickup
            route['deliveries'] += delivery
            #다음 목적지로 이동 준비 
            previous_index = index #현재 위치를 이전 위치로 저장
            index = solution.Value(routing.NextVar(index)) # 다음 목적지 인덱스를 가져옴
            route['distance'] += routing.GetArcCostForVehicle(previous_index, index, vehicle_id) #이전 위치와 다음 위치 사이의 거리를 계산하여 경로 거리 업데이트
        
        # 마지막 depot 도착
        node_index = manager.IndexToNode(index)
        route['path'].append({
            'name': nodes[node_index]['name'],
            'pickup': 0,
            'delivery': 0,
            'current_load': current_load
        })
        #실제 운행한 트럭의 경로만 최종 결과에 추가 
        if len(route['path']) > 2:  # depot 외 방문지가 있는 경우만
            routes.append(route)
            total_distance += route['distance']
    #트럭의 모든 운행 계획이 담긴 routes 리스트와 총 거리를 반환한다.
    return {
        'routes': routes,
        'total_distance': total_distance
    }

# ---------------------------------------------------------------------------
# 6. 휴리스틱 솔버 (대규모 문제용)
# ---------------------------------------------------------------------------

def solve_with_heuristic(district_name, stations, num_vehicles, vehicle_capacity):
    """대규모 문제를 위한 휴리스틱 해법"""
    
    pickup_stations = [s for s in stations if s.get('pickup', 0) > 0]
    delivery_stations = [s for s in stations if s.get('delivery', 0) > 0]
    
    # 우선순위 정렬
    pickup_stations.sort(key=lambda x: x.get('pickup', 0), reverse=True)
    delivery_stations.sort(key=lambda x: x.get('delivery', 0), reverse=True)
    
    routes = []
    for vehicle_id in range(num_vehicles):
        route = {
            'vehicle_id': vehicle_id,
            'path': [{'name': f'{district_name} 차고지', 'pickup': 0, 'delivery': 0, 'current_load': 0}],
            'distance': 0,
            'pickups': 0,
            'deliveries': 0
        }
        
        current_load = 0
        
        # 수거 작업
        for station in pickup_stations[vehicle_id::num_vehicles]:
            if current_load + station['pickup'] <= vehicle_capacity:
                current_load += station['pickup']
                route['path'].append({
                    'name': station['name'],
                    'pickup': station['pickup'],
                    'delivery': 0,
                    'current_load': current_load
                })
                route['pickups'] += station['pickup']
        
        # 배송 작업
        for station in delivery_stations[vehicle_id::num_vehicles]:
            if current_load >= station['delivery']:
                current_load -= station['delivery']
                route['path'].append({
                    'name': station['name'],
                    'pickup': 0,
                    'delivery': station['delivery'],
                    'current_load': current_load
                })
                route['deliveries'] += station['delivery']
        
        # 차고지 복귀
        route['path'].append({
            'name': f'{district_name} 차고지',
            'pickup': 0,
            'delivery': 0,
            'current_load': current_load
        })
        
        if len(route['path']) > 2:
            routes.append(route)
    
    return {'routes': routes, 'total_distance': 0}

# ---------------------------------------------------------------------------
# 7. 결과 출력
# ---------------------------------------------------------------------------
def print_district_solution(district_name, solution):
    """구별 솔루션을 보기 좋게 출력합니다"""
    
    print(f"\n{'='*70}")
    print(f"{district_name} 재배치 계획")
    print(f"{'='*70}")
    
    if not solution or not solution['routes']:
        print("재배치가 필요없거나 경로 생성 실패")
        return
    
    for route in solution['routes']:
        print(f"\n🚚 트럭 {route['vehicle_id'] + 1}번")
        print(f"총 거리: {route['distance']/1000:.2f}km")
        print(f"수거: {route['pickups']}대, 배송: {route['deliveries']}대")
        print("\n경로:")
        
        for i, stop in enumerate(route['path']):
            if stop['pickup'] > 0:
                print(f"  {i}. ↗️ {stop['name']}: +{stop['pickup']}대 (적재: {stop['current_load']})")
            elif stop['delivery'] > 0:
                print(f"  {i}. ↘️ {stop['name']}: -{stop['delivery']}대 (적재: {stop['current_load']})")
            else:
                print(f"  {i}. 📍 {stop['name']}")

# ---------------------------------------------------------------------------
# 8. 메인 실행 함수
# ---------------------------------------------------------------------------
def main():
    """통합 재배치 시스템 메인 함수"""
    
    print("\n" + "="*70)
    print(" "*20 + "서울시 따릉이 통합 재배치 시스템")
    print(" "*25 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    # 설정
    API_KEY = "6464716442737069363863566b466c"
    
    # 1. 데이터 수집 및 구별 분류
    district_stations = get_bike_station_data_by_district(API_KEY)
    
    # 2. 구별 재배치 필요도 분석
    district_analysis = analyze_district_redistribution_needs(district_stations)
    
    # 3. 우선순위 높은 구 선택
    sorted_districts = sorted(district_analysis.items(), 
                            key=lambda x: x[1]['urgency_score'], 
                            reverse=True)
    
    print("\n" + "="*70)
    print("STEP 3: 재배치 계획 수립")
    print("="*70)
    
    # 처리 방식 선택
    print("\n처리 방식을 선택하세요:")
    print("1. 전체 구 자동 처리 (우선순위 상위 5개)")
    print("2. 특정 구 선택 처리")
    print("3. 구별 상세 분석만 보기")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == '1':
        # 상위 5개 구 자동 처리
        print("\n우선순위 상위 5개 구 처리 중...")
        
        for district, analysis in sorted_districts[:5]:
            if analysis['urgency_score'] > 0:
                # 구별 차량 할당 (문제 크기에 따라)
                num_vehicles = min(3, max(1, analysis['urgency_score'] // 10))
                
                solution = solve_district_with_ortools(
                    district, analysis, num_vehicles, 20
                )
                
                if solution:
                    print_district_solution(district, solution)
                    
    elif choice == '2':
        # 특정 구 선택
        print("\n사용 가능한 구:")
        for i, (district, _) in enumerate(sorted_districts[:10], 1):
            print(f"{i}. {district}")
        
        try:
            idx = int(input("\n처리할 구 번호 선택: ")) - 1
            if 0 <= idx < len(sorted_districts):
                district, analysis = sorted_districts[idx]
                
                # 상세 설정
                num_vehicles = int(input(f"투입할 트럭 수 (권장: {max(1, analysis['urgency_score']//10)}): ") or 2)
                capacity = int(input("트럭 용량 (기본: 20): ") or 20)
                
                solution = solve_district_with_ortools(
                    district, analysis, num_vehicles, capacity
                )
                
                if solution:
                    print_district_solution(district, solution)
                    
                    # 결과 저장 옵션
                    save = input("\n결과를 JSON으로 저장하시겠습니까? (y/n): ")
                    if save.lower() == 'y':
                        filename = f"{district}_redistribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(solution, f, ensure_ascii=False, indent=2)
                        print(f"✓ {filename}에 저장되었습니다.")
        except:
            print("잘못된 입력입니다.")
            
    elif choice == '3':
        # 구별 상세 분석
        for district, analysis in sorted_districts[:25]:
            print(f"\n{district}:")
            print(f"  - 전체 대여소: {analysis['total_stations']}개")
            print(f"  - 수거 필요: {len(analysis['pickup_needed'])}개")
            print(f"  - 배송 필요: {len(analysis['delivery_needed'])}개")
            print(f"  - 총 불균형: {analysis['total_imbalance']}대")
    
    print("\n" + "="*70)
    print("프로그램 종료")
    print("="*70)

# ---------------------------------------------------------------------------
# 9. 추가 유틸리티 함수
# ---------------------------------------------------------------------------
def export_all_districts_solution(district_stations, district_analysis):
    """전체 구의 솔루션을 한번에 계산하고 저장"""
    
    all_solutions = {}
    
    for district in district_stations.keys():
        if district in district_analysis:
            analysis = district_analysis[district]
            if analysis['urgency_score'] > 0:
                num_vehicles = min(3, max(1, analysis['urgency_score'] // 10))
                solution = solve_district_with_ortools(
                    district, analysis, num_vehicles, 20
                )
                if solution:
                    all_solutions[district] = solution
    
    # JSON으로 저장
    filename = f"seoul_bike_redistribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_solutions, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 전체 구 솔루션이 {filename}에 저장되었습니다.")
    return all_solutions

if __name__ == "__main__":
    main()