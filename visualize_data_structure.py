#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 구조 시각화 도구
train_diffusion_manip_seq_joints24.p 파일의 구조와 내용을 분석하고 시각화
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd

def analyze_data_structure(data_path):
    """데이터 구조 분석"""
    print("🔍 데이터 로딩 중...")
    data = joblib.load(data_path)
    
    print(f"📊 전체 데이터 타입: {type(data)}")
    print(f"📊 전체 시퀀스 개수: {len(data)}")
    
    # 첫 번째 시퀀스 구조 분석
    first_seq = data[0]
    print(f"\n🔬 첫 번째 시퀀스 구조:")
    print(f"   시퀀스 이름: {first_seq['seq_name']}")
    print(f"   키 개수: {len(first_seq)}")
    
    return data

def visualize_field_shapes(data):
    """각 필드의 shape 정보 시각화"""
    print("\n📐 각 필드의 Shape 정보:")
    print("=" * 60)
    
    sample = data[0]
    field_info = {}
    
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            field_info[key] = {
                'shape': value.shape,
                'dtype': value.dtype,
                'size': value.size,
                'ndim': value.ndim
            }
            print(f"🔹 {key:15} | Shape: {str(value.shape):15} | dtype: {str(value.dtype):10} | Size: {value.size:6}")
        else:
            field_info[key] = {
                'type': type(value).__name__,
                'value': str(value)[:50]
            }
            print(f"🔸 {key:15} | Type: {type(value).__name__:16} | Value: {str(value)[:30]}")
    
    return field_info

def analyze_sequence_statistics(data):
    """시퀀스 통계 분석"""
    print("\n📈 시퀀스 통계 분석:")
    print("=" * 60)
    
    # 기본 통계
    subjects = set()
    objects = set()
    seq_lengths = []
    genders = []
    
    for seq_data in data.values():
        seq_name = seq_data['seq_name']
        parts = seq_name.split('_')
        
        if len(parts) >= 2:
            subjects.add(parts[0])
            objects.add(parts[1])
        
        seq_lengths.append(seq_data['trans'].shape[0])
        genders.append(seq_data['gender'])
    
    print(f"👥 피험자 수: {len(subjects)}")
    print(f"📦 객체 종류: {len(objects)}")
    print(f"⏱️  평균 시퀀스 길이: {np.mean(seq_lengths):.1f} frames")
    print(f"⏱️  최소/최대 길이: {np.min(seq_lengths)} / {np.max(seq_lengths)} frames")
    
    return {
        'subjects': sorted(subjects),
        'objects': sorted(objects),
        'seq_lengths': seq_lengths,
        'genders': genders
    }

def create_visualizations(data, stats):
    """다양한 시각화 생성"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Train Diffusion Manip Seq Joints24 데이터 분석', fontsize=16, fontweight='bold')
    
    # 1. 시퀀스 길이 분포
    axes[0, 0].hist(stats['seq_lengths'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('시퀀스 길이 분포')
    axes[0, 0].set_xlabel('프레임 수')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 객체별 시퀀스 개수
    obj_counts = defaultdict(int)
    for seq_data in data.values():
        obj_name = seq_data['seq_name'].split('_')[1]
        obj_counts[obj_name] += 1
    
    obj_names = list(obj_counts.keys())
    obj_values = list(obj_counts.values())
    
    axes[0, 1].bar(range(len(obj_names)), obj_values, color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('객체별 시퀀스 개수')
    axes[0, 1].set_xlabel('객체')
    axes[0, 1].set_ylabel('시퀀스 개수')
    axes[0, 1].set_xticks(range(len(obj_names)))
    axes[0, 1].set_xticklabels(obj_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 성별 분포
    gender_counts = Counter(stats['genders'])
    axes[0, 2].pie(gender_counts.values(), labels=gender_counts.keys(), autopct='%1.1f%%', 
                   colors=['lightblue', 'lightpink'])
    axes[0, 2].set_title('성별 분포')
    
    # 4. 피험자별 시퀀스 개수
    subj_counts = defaultdict(int)
    for seq_data in data.values():
        subj_name = seq_data['seq_name'].split('_')[0]
        subj_counts[subj_name] += 1
    
    subj_names = sorted(subj_counts.keys(), key=lambda x: int(x.replace('sub', '')))
    subj_values = [subj_counts[name] for name in subj_names]
    
    axes[1, 0].bar(range(len(subj_names)), subj_values, color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('피험자별 시퀀스 개수')
    axes[1, 0].set_xlabel('피험자')
    axes[1, 0].set_ylabel('시퀀스 개수')
    axes[1, 0].set_xticks(range(len(subj_names)))
    axes[1, 0].set_xticklabels(subj_names, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. SMPL 체형 파라미터 분포 (첫 5개 차원)
    betas_data = []
    for seq_data in data.values():
        betas_data.append(seq_data['betas'][0][:5])  # 처음 5개 차원만
    
    betas_array = np.array(betas_data)
    im = axes[1, 1].imshow(betas_array[:100].T, aspect='auto', cmap='RdBu', 
                           interpolation='nearest')
    axes[1, 1].set_title('SMPL 체형 파라미터 (처음 100개 시퀀스)')
    axes[1, 1].set_xlabel('시퀀스 인덱스')
    axes[1, 1].set_ylabel('Beta 차원')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 6. 데이터 필드 크기 비교
    sample = data[0]
    field_sizes = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            field_sizes[key] = value.size
    
    field_names = list(field_sizes.keys())
    field_values = list(field_sizes.values())
    
    axes[1, 2].barh(range(len(field_names)), field_values, color='orange', alpha=0.7)
    axes[1, 2].set_title('필드별 데이터 크기')
    axes[1, 2].set_xlabel('원소 개수')
    axes[1, 2].set_yticks(range(len(field_names)))
    axes[1, 2].set_yticklabels(field_names)
    axes[1, 2].set_xscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_data_structure_diagram():
    """데이터 구조 다이어그램 생성"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 구조 정보
    structure_info = """
    📁 train_diffusion_manip_seq_joints24.p
    ├── Dict[int, Dict] (5280개 시퀀스)
    │
    └── 각 시퀀스 (key: 0~5279)
        ├── seq_name: str           # "sub10_clothesstand_000"
        ├── betas: (1, 16)          # SMPL 체형 파라미터
        ├── gender: scalar          # 성별 정보
        ├── trans2joint: (3,)       # 루트-관절 변환
        ├── rest_offsets: (24, 3)   # 관절 오프셋
        │
        ├── 🚶 인간 모션 (T 프레임)
        │   ├── trans: (T, 3)       # 루트 위치
        │   ├── root_orient: (T, 3) # 루트 회전 (axis-angle)
        │   └── pose_body: (T, 63)  # 몸체 포즈 (21관절×3)
        │
        └── 📦 객체 모션 (T 프레임)
            ├── obj_scale: (T,)      # 객체 스케일
            ├── obj_trans: (T, 3, 1) # 객체 위치
            ├── obj_rot: (T, 3, 3)   # 객체 회전 행렬
            └── obj_com_pos: (T, 3)  # 객체 중심 위치
    """
    
    ax.text(0.05, 0.95, structure_info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('데이터 구조 다이어그램', fontsize=16, fontweight='bold', pad=20)
    
    return fig

def generate_sample_data_preview(data):
    """샘플 데이터 미리보기"""
    print("\n🔍 샘플 데이터 미리보기:")
    print("=" * 80)
    
    sample = data[0]
    print(f"📝 시퀀스: {sample['seq_name']}")
    print(f"👤 성별: {sample['gender']}")
    print(f"⏱️  프레임 수: {sample['trans'].shape[0]}")
    
    print(f"\n🧍 첫 프레임 인간 데이터:")
    print(f"   루트 위치: {sample['trans'][0]}")
    print(f"   루트 회전: {sample['root_orient'][0]}")
    print(f"   첫 관절 포즈: {sample['pose_body'][0][:3]}")
    
    print(f"\n📦 첫 프레임 객체 데이터:")
    print(f"   객체 스케일: {sample['obj_scale'][0]:.4f}")
    print(f"   객체 위치: {sample['obj_trans'][0].flatten()}")
    print(f"   객체 중심: {sample['obj_com_pos'][0]}")
    
    print(f"\n🔢 체형 파라미터 (betas) 처음 5개:")
    print(f"   {sample['betas'][0][:5]}")

def main():
    """메인 함수"""
    data_path = "processed_data/train_diffusion_manip_seq_joints24.p"
    
    print("🚀 데이터 구조 시각화 시작")
    print("=" * 80)
    
    # 1. 데이터 구조 분석
    data = analyze_data_structure(data_path)
    
    # 2. 필드 shape 정보
    field_info = visualize_field_shapes(data)
    
    # 3. 통계 분석
    stats = analyze_sequence_statistics(data)
    
    # 4. 샘플 데이터 미리보기
    generate_sample_data_preview(data)
    
    # 5. 시각화 생성
    print(f"\n📊 시각화 생성 중...")
    
    # 통계 차트
    fig1 = create_visualizations(data, stats)
    fig1.savefig('data_statistics.png', dpi=300, bbox_inches='tight')
    print("✅ 통계 차트 저장: data_statistics.png")
    
    # 구조 다이어그램
    fig2 = create_data_structure_diagram()
    fig2.savefig('data_structure_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ 구조 다이어그램 저장: data_structure_diagram.png")
    
    plt.show()
    
    print(f"\n🎉 분석 완료!")
    print(f"   총 {len(data)}개 시퀀스")
    print(f"   {len(stats['subjects'])}명 피험자")
    print(f"   {len(stats['objects'])}종류 객체")

if __name__ == "__main__":
    main()
