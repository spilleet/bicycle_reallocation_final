#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
서울시 따릉이 재배치 시스템 실행 스크립트
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Python 버전 확인"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        return False
    print(f"✅ Python 버전 확인 완료: {sys.version}")
    return True

def check_dependencies():
    """필요한 패키지 확인 및 설치"""
    required_packages = [
        'flask', 'requests', 'numpy', 'sklearn', 'ortools'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 패키지 확인 완료")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 패키지가 설치되지 않았습니다")
    
    if missing_packages:
        print(f"\n📦 누락된 패키지 설치 중: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ 모든 패키지 설치 완료")
        except subprocess.CalledProcessError:
            print("❌ 패키지 설치에 실패했습니다")
            return False
    
    return True

def create_templates_directory():
    """templates 디렉토리 생성"""
    templates_dir = Path('templates')
    if not templates_dir.exists():
        print("📁 templates 디렉토리 생성 중...")
        templates_dir.mkdir()
        print("✅ templates 디렉토리 생성 완료")

def start_flask_app():
    """Flask 애플리케이션 시작"""
    print("\n🚀 Flask 애플리케이션 시작 중...")
    
    try:
        # Flask 앱 실행
        from app import app
        
        print("✅ Flask 앱 로드 완료")
        print("🌐 웹 브라우저에서 http://localhost:5000 으로 접속하세요")
        print("⏹️  종료하려면 Ctrl+C를 누르세요")
        
        # 자동으로 브라우저 열기 (선택사항)
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
            print("🌐 브라우저가 자동으로 열렸습니다")
        except:
            print("⚠️  브라우저를 수동으로 열어주세요")
        
        # Flask 앱 실행
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Flask 앱 로드 실패: {e}")
        print("💡 requirements.txt의 패키지들이 제대로 설치되었는지 확인해주세요")
        return False
    except Exception as e:
        print(f"❌ Flask 앱 실행 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🚲 서울시 따릉이 재배치 시스템")
    print("=" * 60)
    
    # 1. Python 버전 확인
    if not check_python_version():
        sys.exit(1)
    
    # 2. 의존성 확인 및 설치
    if not check_dependencies():
        sys.exit(1)
    
    # 3. templates 디렉토리 생성
    create_templates_directory()
    
    # 4. Flask 앱 시작
    start_flask_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 프로그램이 종료되었습니다")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)
