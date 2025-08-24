#!/usr/bin/env python3
"""Test script to trigger optimization and see debug output"""

import requests
import json
import time

BASE_URL = "http://localhost:8080"

def test_optimization():
    # Step 1: Collect data
    print("Step 1: Collecting data...")
    response = requests.post(f"{BASE_URL}/api/data/collect")
    if response.status_code != 200:
        print(f"Error collecting data: {response.text}")
        return
    
    data = response.json()
    if not data['success']:
        print(f"Failed to collect data: {data}")
        return
    
    print(f"Data collected successfully. Found {len(data['data']['districts'])} districts")
    
    # Find a district with problems
    districts = data['data']['districts']
    urgent_district = None
    for district in districts:
        if district['pickup_needed'] > 0 or district['delivery_needed'] > 0:
            urgent_district = district
            break
    
    if not urgent_district:
        print("No districts need optimization")
        return
    
    print(f"\nStep 2: Optimizing district {urgent_district['name']}...")
    print(f"  - Pickup needed: {urgent_district['pickup_needed']} stations")
    print(f"  - Delivery needed: {urgent_district['delivery_needed']} stations")
    
    # Step 2: Optimize
    optimization_data = {
        'district_id': urgent_district['id'],
        'num_vehicles': 2,
        'vehicle_capacity': 20,
        'use_clustering': False  # Start without clustering for simpler debugging
    }
    
    response = requests.post(
        f"{BASE_URL}/api/optimize", 
        json=optimization_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code != 200:
        print(f"Error optimizing: {response.text}")
        return
    
    result = response.json()
    if not result['success']:
        print(f"Optimization failed: {result}")
        return
    
    print("\nOptimization Result:")
    print(json.dumps(result['data'], indent=2))
    
    # Check for routes and path data
    if 'routes' in result['data']:
        for i, route in enumerate(result['data']['routes']):
            print(f"\nRoute {i}:")
            print(f"  - Path length: {len(route.get('path', []))}")
            if route.get('path'):
                print(f"  - First stop: {route['path'][0]}")
                print(f"  - Last stop: {route['path'][-1]}")

if __name__ == "__main__":
    test_optimization()