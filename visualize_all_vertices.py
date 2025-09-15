#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample.npz에서 특정 프레임의 모든 버텍스와 관절을 3D로 시각화하는 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def load_frame_data(npz_path, frame_t):
    """
    NPZ 파일에서 특정 프레임의 데이터 로드
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        seq_name = str(data['seq_name'])
        seq_len = int(data['seq_len'])
        
        print(f"🎬 시퀀스: {seq_name}")
        print(f"📏 총 프레임: {seq_len}")
        print(f"🎯 요청 프레임: T={frame_t}")
        
        if frame_t >= seq_len:
            print(f"❌ 오류: 프레임 {frame_t}이 시퀀스 길이 {seq_len}을 초과합니다.")
            return None
        
        # 프레임 데이터 추출
        mesh_jnts = data['mesh_jnts'][0, frame_t]  # (24, 3)
        mesh_verts = data['mesh_verts'][0, frame_t]  # (10475, 3)
        mesh_faces = data['mesh_faces']  # (20908, 3)
        
        print(f"✅ 데이터 로드 완료:")
        print(f"   관절 개수: {mesh_jnts.shape[0]}")
        print(f"   버텍스 개수: {mesh_verts.shape[0]}")
        print(f"   페이스 개수: {mesh_faces.shape[0]}")
        
        return {
            'seq_name': seq_name,
            'frame_t': frame_t,
            'seq_len': seq_len,
            'mesh_jnts': mesh_jnts,
            'mesh_verts': mesh_verts,
            'mesh_faces': mesh_faces
        }
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None

def analyze_vertex_distribution(mesh_verts):
    """
    버텍스 분포 분석
    """
    print(f"\n📊 버텍스 분포 분석:")
    print("=" * 50)
    
    # 축별 통계
    for i, axis in enumerate(['X', 'Y', 'Z']):
        coords = mesh_verts[:, i]
        print(f"{axis}축: 최소={coords.min():.4f}, 최대={coords.max():.4f}, "
              f"평균={coords.mean():.4f}, 표준편차={coords.std():.4f}")
    
    # Z축 높이별 버텍스 개수
    z_coords = mesh_verts[:, 2]
    z_ranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0)]
    
    print(f"\n높이별 버텍스 분포:")
    for z_min, z_max in z_ranges:
        count = np.sum((z_coords >= z_min) & (z_coords < z_max))
        percentage = count / len(z_coords) * 100
        print(f"  {z_min:.1f}m ~ {z_max:.1f}m: {count:5d}개 ({percentage:5.1f}%)")
    
    # 가장 낮은/높은 버텍스들
    min_z_idx = np.argmin(z_coords)
    max_z_idx = np.argmax(z_coords)
    
    print(f"\n극값 버텍스:")
    print(f"  최저점 (idx {min_z_idx}): [{mesh_verts[min_z_idx, 0]:.4f}, "
          f"{mesh_verts[min_z_idx, 1]:.4f}, {mesh_verts[min_z_idx, 2]:.4f}]")
    print(f"  최고점 (idx {max_z_idx}): [{mesh_verts[max_z_idx, 0]:.4f}, "
          f"{mesh_verts[max_z_idx, 1]:.4f}, {mesh_verts[max_z_idx, 2]:.4f}]")

def find_foot_vertices(mesh_jnts, mesh_verts, xy_radius=0.02):
    """
    발 관절 주변 버텍스들 찾기
    """
    foot_joints = {
        'L_ANKLE': 7,   # 왼쪽 발목
        'R_ANKLE': 8,   # 오른쪽 발목
        'L_TOE': 10,    # 왼쪽 발가락
        'R_TOE': 11     # 오른쪽 발가락
    }
    
    foot_vertex_indices = {}
    
    for name, joint_idx in foot_joints.items():
        joint_pos = mesh_jnts[joint_idx]
        joint_xy = joint_pos[:2]
        
        # XY 평면에서 거리 계산
        verts_xy = mesh_verts[:, :2]
        distances = np.linalg.norm(verts_xy - joint_xy, axis=1)
        near_mask = distances <= xy_radius
        
        foot_vertex_indices[name] = np.where(near_mask)[0]
        
        print(f"{name} 주변 {xy_radius*100:.1f}cm 내 버텍스: {np.sum(near_mask)}개")
    
    return foot_vertex_indices

def visualize_all_vertices_3d(frame_data, show_mesh=False, show_foot_regions=True):
    """
    모든 버텍스를 3D로 시각화
    
    Args:
        frame_data: 프레임 데이터
        show_mesh: 메쉬 면 표시 여부 (느려질 수 있음)
        show_foot_regions: 발 부위 하이라이트 여부
    """
    mesh_jnts = frame_data['mesh_jnts']
    mesh_verts = frame_data['mesh_verts']
    mesh_faces = frame_data['mesh_faces']
    
    print(f"\n🎨 3D 시각화 생성 중...")
    
    # 한글 폰트 문제 해결
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 여러 뷰로 나누어 표시
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 전체 뷰 - 모든 버텍스
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_all_vertices_view(ax1, mesh_jnts, mesh_verts, "All Vertices + Joints")
    
    # 2. 높이별 색상 뷰
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_vertices_by_height(ax2, mesh_jnts, mesh_verts, "Vertices by Height")
    
    # 3. 발 부위 하이라이트
    if show_foot_regions:
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        plot_foot_regions(ax3, mesh_jnts, mesh_verts, "Foot Regions")
    
    # 4. 메쉬 와이어프레임 (샘플링)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    plot_mesh_wireframe(ax4, mesh_verts, mesh_faces, "Mesh Wireframe (Sampled)")
    
    # 5. 상위뷰 (XY 평면)
    ax5 = fig.add_subplot(2, 3, 5)
    plot_top_view(ax5, mesh_jnts, mesh_verts, "Top View (XY)")
    
    # 6. 측면뷰 (XZ 평면)
    ax6 = fig.add_subplot(2, 3, 6)
    plot_side_view(ax6, mesh_jnts, mesh_verts, "Side View (XZ)")
    
    plt.suptitle(f'Complete Vertex Visualization - {frame_data["seq_name"]} Frame {frame_data["frame_t"]}', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    
    # 저장
    save_path = f"all_vertices_frame_{frame_data['frame_t']}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 전체 버텍스 시각화가 {save_path}에 저장되었습니다.")
    
    return save_path

def plot_all_vertices_view(ax, mesh_jnts, mesh_verts, title):
    """
    모든 버텍스와 관절 표시
    """
    # 모든 버텍스 (작은 점으로)
    ax.scatter(mesh_verts[:, 0], mesh_verts[:, 1], mesh_verts[:, 2], 
              c='lightblue', s=0.5, alpha=0.3, label=f'Vertices ({len(mesh_verts)})')
    
    # 모든 관절
    ax.scatter(mesh_jnts[:, 0], mesh_jnts[:, 1], mesh_jnts[:, 2], 
              c='red', s=50, alpha=0.8, label='Joints (24)', edgecolors='black')
    
    # 발 관절 하이라이트
    foot_joints = [7, 8, 10, 11]  # L_ANKLE, R_ANKLE, L_TOE, R_TOE
    foot_colors = ['blue', 'red', 'cyan', 'magenta']
    for i, joint_idx in enumerate(foot_joints):
        ax.scatter(*mesh_jnts[joint_idx], c=foot_colors[i], s=100, 
                  alpha=1.0, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

def plot_vertices_by_height(ax, mesh_jnts, mesh_verts, title):
    """
    높이별로 색상을 달리해서 버텍스 표시
    """
    z_coords = mesh_verts[:, 2]
    
    # 높이에 따른 색상 매핑
    scatter = ax.scatter(mesh_verts[:, 0], mesh_verts[:, 1], mesh_verts[:, 2], 
                        c=z_coords, s=1, alpha=0.6, cmap='viridis')
    
    # 관절들
    ax.scatter(mesh_jnts[:, 0], mesh_jnts[:, 1], mesh_jnts[:, 2], 
              c='red', s=50, alpha=0.8, edgecolors='black')
    
    plt.colorbar(scatter, ax=ax, shrink=0.5, label='Height (m)')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

def plot_foot_regions(ax, mesh_jnts, mesh_verts, title):
    """
    발 부위 버텍스들 하이라이트
    """
    # 전체 버텍스 (회색)
    ax.scatter(mesh_verts[:, 0], mesh_verts[:, 1], mesh_verts[:, 2], 
              c='lightgray', s=0.3, alpha=0.2)
    
    # 발 관절 주변 버텍스들
    foot_vertex_indices = find_foot_vertices(mesh_jnts, mesh_verts, xy_radius=0.02)
    
    colors = {'L_ANKLE': 'blue', 'R_ANKLE': 'red', 'L_TOE': 'cyan', 'R_TOE': 'magenta'}
    
    for name, indices in foot_vertex_indices.items():
        if len(indices) > 0:
            foot_verts = mesh_verts[indices]
            ax.scatter(foot_verts[:, 0], foot_verts[:, 1], foot_verts[:, 2], 
                      c=colors[name], s=5, alpha=0.7, label=f'{name} ({len(indices)})')
    
    # 발 관절들
    foot_joints = [7, 8, 10, 11]
    for i, joint_idx in enumerate(foot_joints):
        name = list(colors.keys())[i]
        ax.scatter(*mesh_jnts[joint_idx], c=colors[name], s=100, 
                  alpha=1.0, edgecolors='black', linewidths=2, marker='o')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

def plot_mesh_wireframe(ax, mesh_verts, mesh_faces, title):
    """
    메쉬 와이어프레임 (샘플링해서 표시)
    """
    # 너무 많은 면을 그리면 느리므로 샘플링
    num_faces = len(mesh_faces)
    sample_size = min(1000, num_faces)  # 최대 1000개 면만 표시
    sample_indices = np.random.choice(num_faces, sample_size, replace=False)
    sampled_faces = mesh_faces[sample_indices]
    
    # 와이어프레임 그리기
    for face in sampled_faces:
        # 삼각형의 세 점
        v0, v1, v2 = mesh_verts[face]
        
        # 삼각형 그리기
        triangle = np.array([v0, v1, v2, v0])  # 마지막에 첫 점 추가해서 닫기
        ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
               'b-', alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{title} ({sample_size}/{num_faces})')

def plot_top_view(ax, mesh_jnts, mesh_verts, title):
    """
    상위뷰 (XY 평면)
    """
    # 버텍스들
    ax.scatter(mesh_verts[:, 0], mesh_verts[:, 1], 
              c=mesh_verts[:, 2], s=0.5, alpha=0.5, cmap='viridis')
    
    # 관절들
    ax.scatter(mesh_jnts[:, 0], mesh_jnts[:, 1], 
              c='red', s=30, alpha=0.8, edgecolors='black')
    
    # 발 관절 번호 표시
    foot_joints = [7, 8, 10, 11]
    foot_names = ['L_ANK', 'R_ANK', 'L_TOE', 'R_TOE']
    for joint_idx, name in zip(foot_joints, foot_names):
        pos = mesh_jnts[joint_idx]
        ax.annotate(name, (pos[0], pos[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

def plot_side_view(ax, mesh_jnts, mesh_verts, title):
    """
    측면뷰 (XZ 평면)
    """
    # 버텍스들
    ax.scatter(mesh_verts[:, 0], mesh_verts[:, 2], 
              c=mesh_verts[:, 1], s=0.5, alpha=0.5, cmap='viridis')
    
    # 관절들
    ax.scatter(mesh_jnts[:, 0], mesh_jnts[:, 2], 
              c='red', s=30, alpha=0.8, edgecolors='black')
    
    # 발 관절 번호 표시
    foot_joints = [7, 8, 10, 11]
    foot_names = ['L_ANK', 'R_ANK', 'L_TOE', 'R_TOE']
    for joint_idx, name in zip(foot_joints, foot_names):
        pos = mesh_jnts[joint_idx]
        ax.annotate(name, (pos[0], pos[2]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def main():
    """
    메인 함수
    """
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python visualize_all_vertices.py <프레임번호>")
        print("  python visualize_all_vertices.py 0")
        print("  python visualize_all_vertices.py 60")
        sys.exit(1)
    
    frame_t = int(sys.argv[1])
    
    # 데이터 로드
    frame_data = load_frame_data('sample.npz', frame_t)
    if frame_data is None:
        return
    
    # 버텍스 분포 분석
    analyze_vertex_distribution(frame_data['mesh_verts'])
    
    # 발 관절 주변 버텍스 분석
    print(f"\n🦶 발 관절 주변 버텍스 분석:")
    find_foot_vertices(frame_data['mesh_jnts'], frame_data['mesh_verts'])
    
    # 전체 버텍스 3D 시각화
    visualize_all_vertices_3d(frame_data, show_mesh=False, show_foot_regions=True)

if __name__ == "__main__":
    main()
