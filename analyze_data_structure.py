#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 구조 분석 도구 (텍스트 기반)
train_diffusion_manip_seq_joints24.p 파일의 구조와 내용을 분석
"""

import joblib
import numpy as np
from collections import defaultdict, Counter

def print_header(title):
    """헤더 출력"""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}")

def print_section(title):
    """섹션 출력"""
    print(f"\n📊 {title}")
    print(f"{'-'*60}")

def analyze_data_structure(data_path):
    """데이터 구조 분석"""
    print_header("데이터 구조 분석 시작")
    
    print("🔄 데이터 로딩 중...")
    data = joblib.load(data_path)
    
    print(f"✅ 로딩 완료!")
    print(f"   📁 파일: {data_path}")
    print(f"   📊 데이터 타입: {type(data)}")
    print(f"   📈 전체 시퀀스 개수: {len(data)}")
    
    return data

def analyze_field_shapes(data):
    """각 필드의 shape 정보 분석"""
    print_section("필드별 Shape 정보")
    
    sample = data[0]
    print(f"🔬 샘플 시퀀스: {sample['seq_name']}")
    print(f"📝 총 필드 개수: {len(sample)}")
    
    print(f"\n{'필드명':<20} {'Shape':<20} {'dtype':<15} {'크기':<10} {'타입'}")
    print(f"{'-'*80}")
    
    field_info = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            field_info[key] = {
                'shape': value.shape,
                'dtype': str(value.dtype),
                'size': value.size,
                'ndim': value.ndim
            }
            print(f"{key:<20} {str(value.shape):<20} {str(value.dtype):<15} {value.size:<10} numpy.ndarray")
        else:
            field_info[key] = {
                'type': type(value).__name__,
                'value': str(value)
            }
            print(f"{key:<20} {'-':<20} {'-':<15} {'-':<10} {type(value).__name__}")
    
    return field_info

def analyze_sequence_statistics(data):
    """시퀀스 통계 분석"""
    print_section("시퀀스 통계 분석")
    
    subjects = set()
    objects = set()
    seq_lengths = []
    genders = []
    
    print("📊 데이터 수집 중...")
    for seq_data in data.values():
        seq_name = seq_data['seq_name']
        parts = seq_name.split('_')
        
        if len(parts) >= 2:
            subjects.add(parts[0])
            objects.add(parts[1])
        
        seq_lengths.append(seq_data['trans'].shape[0])
        gender_val = seq_data['gender']
        if isinstance(gender_val, np.ndarray):
            gender_val = gender_val.item()  # numpy scalar을 Python 값으로 변환
        genders.append(gender_val)
    
    seq_lengths = np.array(seq_lengths)
    gender_counts = Counter(genders)
    
    print(f"✅ 기본 통계:")
    print(f"   👥 피험자 수: {len(subjects)}")
    print(f"   📦 객체 종류 수: {len(objects)}")
    print(f"   ⏱️  평균 시퀀스 길이: {np.mean(seq_lengths):.1f} frames")
    print(f"   ⏱️  최소 길이: {np.min(seq_lengths)} frames")
    print(f"   ⏱️  최대 길이: {np.max(seq_lengths)} frames")
    print(f"   ⏱️  중간값: {np.median(seq_lengths):.1f} frames")
    print(f"   ⏱️  표준편차: {np.std(seq_lengths):.1f} frames")
    
    print(f"\n👥 성별 분포:")
    for gender, count in gender_counts.items():
        percentage = (count / len(genders)) * 100
        print(f"   {gender}: {count}개 ({percentage:.1f}%)")
    
    return {
        'subjects': sorted(subjects),
        'objects': sorted(objects),
        'seq_lengths': seq_lengths,
        'genders': genders,
        'gender_counts': gender_counts
    }

def analyze_object_distribution(data):
    """객체별 분포 분석"""
    print_section("객체별 시퀀스 분포")
    
    obj_counts = defaultdict(int)
    obj_subjects = defaultdict(set)
    
    for seq_data in data.values():
        seq_name = seq_data['seq_name']
        parts = seq_name.split('_')
        if len(parts) >= 2:
            obj_name = parts[1]
            subj_name = parts[0]
            obj_counts[obj_name] += 1
            obj_subjects[obj_name].add(subj_name)
    
    print(f"{'객체명':<15} {'시퀀스 수':<10} {'피험자 수':<10} {'평균/피험자':<12}")
    print(f"{'-'*50}")
    
    total_sequences = sum(obj_counts.values())
    for obj_name in sorted(obj_counts.keys()):
        count = obj_counts[obj_name]
        subj_count = len(obj_subjects[obj_name])
        avg_per_subj = count / subj_count if subj_count > 0 else 0
        percentage = (count / total_sequences) * 100
        print(f"{obj_name:<15} {count:<10} {subj_count:<10} {avg_per_subj:<12.1f} ({percentage:.1f}%)")

def analyze_subject_distribution(data):
    """피험자별 분포 분석"""
    print_section("피험자별 시퀀스 분포")
    
    subj_counts = defaultdict(int)
    subj_objects = defaultdict(set)
    
    for seq_data in data.values():
        seq_name = seq_data['seq_name']
        parts = seq_name.split('_')
        if len(parts) >= 2:
            subj_name = parts[0]
            obj_name = parts[1]
            subj_counts[subj_name] += 1
            subj_objects[subj_name].add(obj_name)
    
    print(f"{'피험자':<10} {'시퀀스 수':<10} {'객체 수':<10} {'평균/객체':<12}")
    print(f"{'-'*45}")
    
    # 피험자 번호 순으로 정렬
    sorted_subjects = sorted(subj_counts.keys(), key=lambda x: int(x.replace('sub', '')))
    
    for subj_name in sorted_subjects:
        count = subj_counts[subj_name]
        obj_count = len(subj_objects[subj_name])
        avg_per_obj = count / obj_count if obj_count > 0 else 0
        print(f"{subj_name:<10} {count:<10} {obj_count:<10} {avg_per_obj:<12.1f}")

def show_sample_data(data):
    """샘플 데이터 미리보기"""
    print_section("샘플 데이터 미리보기")
    
    sample = data[0]
    print(f"🔍 시퀀스: {sample['seq_name']}")
    print(f"👤 성별: {sample['gender']}")
    print(f"⏱️  프레임 수: {sample['trans'].shape[0]}")
    
    print(f"\n🧍 첫 프레임 인간 데이터:")
    print(f"   루트 위치 (trans): {sample['trans'][0]}")
    print(f"   루트 회전 (root_orient): {sample['root_orient'][0]}")
    print(f"   첫 관절 포즈: {sample['pose_body'][0][:3]}")
    
    print(f"\n📦 첫 프레임 객체 데이터:")
    print(f"   객체 스케일: {sample['obj_scale'][0]:.6f}")
    print(f"   객체 위치: {sample['obj_trans'][0].flatten()}")
    print(f"   객체 중심: {sample['obj_com_pos'][0]}")
    
    print(f"\n🔢 SMPL 체형 파라미터 (betas):")
    print(f"   전체 차원: {sample['betas'].shape}")
    print(f"   처음 8개 값: {sample['betas'][0][:8]}")

def create_structure_diagram():
    """텍스트 기반 구조 다이어그램"""
    print_section("데이터 구조 다이어그램")
    
    diagram = """
📁 train_diffusion_manip_seq_joints24.p
│
├── 📊 전체 구조: Dict[int, Dict] (5280개 시퀀스)
│   ├── Key: 0, 1, 2, ..., 5279 (시퀀스 인덱스)
│   └── Value: 각 시퀀스 데이터 딕셔너리
│
└── 🔍 각 시퀀스 구조:
    │
    ├── 📝 메타데이터
    │   ├── seq_name: str           # "sub10_clothesstand_000"
    │   ├── gender: str             # "male" or "female"  
    │   └── betas: (1, 16)          # SMPL 체형 파라미터
    │
    ├── 🔧 변환 정보
    │   ├── trans2joint: (3,)       # 루트→관절 변환 벡터
    │   └── rest_offsets: (24, 3)   # 관절 오프셋 (휴식 포즈)
    │
    ├── 🚶 인간 모션 데이터 (T 프레임)
    │   ├── trans: (T, 3)           # 루트 관절 위치 (전역좌표)
    │   ├── root_orient: (T, 3)     # 루트 회전 (axis-angle)
    │   └── pose_body: (T, 63)      # 몸체 포즈 (21관절 × 3)
    │
    └── 📦 객체 모션 데이터 (T 프레임)
        ├── obj_scale: (T,)         # 객체 크기 스케일
        ├── obj_trans: (T, 3, 1)    # 객체 위치 (전역좌표)
        ├── obj_rot: (T, 3, 3)      # 객체 회전 행렬
        └── obj_com_pos: (T, 3)     # 객체 중심점 위치

💡 주요 특징:
   • T는 각 시퀀스마다 다름 (29~652 프레임)
   • 모든 좌표는 전역 좌표계 기준
   • SMPL 모델 기반 인간 표현 (24개 관절)
   • 15종류 객체 × 15명 피험자 조합
    """
    
    print(diagram)

def main():
    """메인 함수"""
    data_path = "processed_data/train_diffusion_manip_seq_joints24.p"
    
    print("🚀 데이터 구조 분석 도구")
    print(f"📁 대상 파일: {data_path}")
    
    try:
        # 1. 데이터 로드 및 기본 구조
        data = analyze_data_structure(data_path)
        
        # 2. 필드 shape 정보
        field_info = analyze_field_shapes(data)
        
        # 3. 시퀀스 통계
        stats = analyze_sequence_statistics(data)
        
        # 4. 객체별 분포
        analyze_object_distribution(data)
        
        # 5. 피험자별 분포  
        analyze_subject_distribution(data)
        
        # 6. 샘플 데이터
        show_sample_data(data)
        
        # 7. 구조 다이어그램
        create_structure_diagram()
        
        print_header("분석 완료")
        print(f"✅ 총 {len(data)}개 시퀀스 분석 완료")
        print(f"📊 {len(stats['subjects'])}명 피험자, {len(stats['objects'])}종류 객체")
        print(f"⏱️  평균 {np.mean(stats['seq_lengths']):.1f} 프레임")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
