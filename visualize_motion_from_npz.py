import os
import sys
import argparse
import numpy as np
import torch
import trimesh
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append('/home/jeongyoon/HSI/chois_release')

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object
from manip.data.cano_traj_dataset import CanoObjectTrajDataset
from pytorch3d import transforms

# trainer_chois.py에서 run_smplx_model 함수 import
from trainer_chois import run_smplx_model

class MotionVisualizer:
    def __init__(self, data_root_folder, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_root_folder = data_root_folder
        
        # 데이터셋 초기화 (SMPL 모델 등을 위해 필요)
        self.ds = CanoObjectTrajDataset(
            train=False,
            data_root_folder=data_root_folder,
            window=120,
            use_object_splits=False,
            use_object_keypoints=True
        )
        #npz에서 로드하는 부분 추가로 필요한 부분이 있다면 입력
    def load_npz_data(self, npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            motion_data = {
                'seq_name': str(data['seq_name']),
                'obj_name': str(data['obj_name']),
                'betas': torch.from_numpy(data['betas']).float().to(self.device),
                'gender': str(data['gender']),
                'local_rot_aa': torch.from_numpy(data['local_rot_aa']).float().to(self.device),
                'root_trans': torch.from_numpy(data['root_trans']).float().to(self.device),
                'seq_len': int(data['seq_len']),
                'step': int(data['step']),
                'idx': int(data['idx'])
            }
            
            # 추가 데이터가 있는 경우
            if 'obj_rot_mat' in data and data['obj_rot_mat'] is not None:
                motion_data['obj_rot_mat'] = torch.from_numpy(data['obj_rot_mat']).float().to(self.device)
            if 'obj_com_pos' in data and data['obj_com_pos'] is not None:
                motion_data['obj_com_pos'] = torch.from_numpy(data['obj_com_pos']).float().to(self.device)

            return motion_data
            
        except Exception as e:
            print(f"Error loading NPZ file {npz_path}: {e}")
            return None
    
    def generate_human_mesh(self, motion_data):

        try:
            betas = motion_data['betas']
            gender = motion_data['gender']
            local_rot_aa = motion_data['local_rot_aa']  # T x 22 x 3
            root_trans = motion_data['root_trans']      # T x 3
            
            # SMPL 모델 실행
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
                root_trans[None].to(self.device),      # (1, T, 3)
                local_rot_aa[None].to(self.device),    # (1, T, 22, 3)
                betas.to(self.device),                  # (10,) or similar
                [gender],                               # ['male'] or ['female']
                self.ds.bm_dict,                        # SMPL body model dictionary
                return_joints24=True
            )
            
            print(f"Generated human mesh:")
            print(f"  - Joints shape: {mesh_jnts.shape}")
            print(f"  - Vertices shape: {mesh_verts.shape}")
            print(f"  - Faces shape: {mesh_faces.shape}")
            
            return mesh_jnts[0], mesh_verts[0], mesh_faces
            
        except Exception as e:
            print(f"Error generating human mesh: {e}")
            return None, None, None
    
    def generate_object_mesh(self, motion_data):

        try:
            object_name = motion_data['obj_name']
            
            # 객체 rest pose geometry 로드
            obj_rest_verts, obj_mesh_faces = self.ds.load_rest_pose_object_geometry(object_name)
            obj_rest_verts = torch.from_numpy(obj_rest_verts).float().to(self.device)
            
            if 'obj_rot_mat' in motion_data and 'obj_com_pos' in motion_data:
                # 객체 회전과 위치 정보가 있는 경우
                obj_rot_mat = motion_data['obj_rot_mat']    # T x 3 x 3
                obj_com_pos = motion_data['obj_com_pos']    # T x 3
                
                # 객체 geometry 생성
                obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(
                    obj_rot_mat, obj_com_pos, obj_rest_verts
                )
            else:
                # 기본적으로 rest pose 사용
                seq_len = motion_data['seq_len']
                obj_mesh_verts = obj_rest_verts[None].repeat(seq_len, 1, 1)  # T x Nv x 3
            
            print(f"Generated object mesh:")
            print(f"  - Object: {object_name}")
            print(f"  - Vertices shape: {obj_mesh_verts.shape}")
            print(f"  - Faces shape: {obj_mesh_faces.shape}")
            
            return obj_mesh_verts, obj_mesh_faces
            
        except Exception as e:
            print(f"Error generating object mesh: {e}")
            # 객체가 없는 경우 더미 데이터 반환
            seq_len = motion_data['seq_len']
            dummy_verts = torch.zeros(seq_len, 1, 3).to(self.device)
            dummy_faces = np.array([[0, 0, 0]])
            return dummy_verts, dummy_faces
    
    def export_to_mesh_files(self, mesh_verts, mesh_faces, obj_verts, obj_faces, 
                            output_folder, motion_data):

        try:
            seq_len = motion_data['seq_len']
            
            # 실제 시퀀스 길이만큼만 저장
            human_verts_trimmed = mesh_verts.detach().cpu().numpy()[:seq_len]
            obj_verts_trimmed = obj_verts.detach().cpu().numpy()[:seq_len]
            human_faces = mesh_faces.detach().cpu().numpy()
            
            # 메쉬 파일들 저장
            save_verts_faces_to_mesh_file_w_object(
                human_verts_trimmed,
                human_faces,
                obj_verts_trimmed,
                obj_faces,
                output_folder
            )
            
            print(f"Exported mesh files to: {output_folder}")
            
        except Exception as e:
            print(f"Error exporting mesh files: {e}")
    
    def create_condition_balls(self, motion_data, output_folder):

        try:
            root_trans = motion_data['root_trans']  # T x 3
            
            # 시작점과 끝점
            start_pos = root_trans[0]  # 3
            end_pos = root_trans[-1]   # 3
            
            # 구 메쉬 생성
            ball_radius = 0.05
            start_ball = trimesh.creation.icosphere(radius=ball_radius)
            end_ball = trimesh.creation.icosphere(radius=ball_radius)
            
            # 위치 조정
            start_ball.vertices += start_pos.detach().cpu().numpy()
            end_ball.vertices += end_pos.detach().cpu().numpy()
            
            # 저장
            ball_folder = os.path.join(output_folder, "conditions")
            os.makedirs(ball_folder, exist_ok=True)
            
            start_ball.export(os.path.join(ball_folder, "start_pos.ply"))
            end_ball.export(os.path.join(ball_folder, "end_pos.ply"))
            
            # conditions.ply 파일도 생성 (기존 코드 호환성)
            combined_positions = torch.stack([start_pos, end_pos], dim=0)  # 2 x 3
            self.create_ball_mesh(combined_positions, os.path.join(ball_folder, "conditions.ply"))
            
            print(f"Created condition balls in: {ball_folder}")
            
        except Exception as e:
            print(f"Error creating condition balls: {e}")
    
    def create_ball_mesh(self, positions, output_path):

        try:
            ball_radius = 0.05
            combined_mesh = None
            
            for i, pos in enumerate(positions):
                ball = trimesh.creation.icosphere(radius=ball_radius)
                ball.vertices += pos.detach().cpu().numpy()
                
                if combined_mesh is None:
                    combined_mesh = ball
                else:
                    combined_mesh = trimesh.util.concatenate([combined_mesh, ball])
            
            if combined_mesh is not None:
                combined_mesh.export(output_path)
                
        except Exception as e:
            print(f"Error creating ball mesh: {e}")
    
    def render_video(self, mesh_folder, output_video_path, condition_folder=None):
        try:
            # 임시 이미지 폴더 생성
            img_folder = os.path.join(os.path.dirname(output_video_path), "temp_imgs")
            os.makedirs(img_folder, exist_ok=True)
            
            # 바닥 블렌더 파일 경로
            floor_blend_path = os.path.join(self.data_root_folder, "blender_files/floor_colorful_mat.blend")
            
            # Blender 렌더링 실행
            run_blender_rendering_and_save2video(
                mesh_folder,
                img_folder,
                output_video_path,
                condition_folder=condition_folder,
                vis_object=True,
                vis_condition=(condition_folder is not None),
                scene_blend_path=floor_blend_path,
                fps=30
            )
                      
        except Exception as e:
            print(f"Error rendering video: {e}")
    
    def visualize_motion(self, npz_path, output_dir, render_video=True, create_conditions=True):

    
        # NPZ 데이터 로드
        motion_data = self.load_npz_data(npz_path)
        if motion_data is None:
            return
        
        # 출력 폴더 설정
        npz_name = os.path.splitext(os.path.basename(npz_path))[0]
        output_folder = os.path.join(output_dir, npz_name)
        mesh_folder = os.path.join(output_folder, "meshes")
        
        os.makedirs(mesh_folder, exist_ok=True)
        
        # 조건 구 폴더는 필요한 경우에만 생성
        condition_folder = None
        if create_conditions:
            condition_folder = os.path.join(output_folder, "conditions")
            os.makedirs(condition_folder, exist_ok=True)
        
        # 인간 메쉬 생성
        print("Generating human mesh...")
        mesh_jnts, mesh_verts, mesh_faces = self.generate_human_mesh(motion_data)
        if mesh_verts is None:
            print("Failed to generate human mesh")
            return
        
        # 객체 메쉬 생성
        print("Generating object mesh...")
        obj_verts, obj_faces = self.generate_object_mesh(motion_data)
        
        # 메쉬 파일 저장
        print("Exporting mesh files...")
        self.export_to_mesh_files(mesh_verts, mesh_faces, obj_verts, obj_faces, 
                                 mesh_folder, motion_data)
        
        # 조건 구 생성 (옵션)
        if create_conditions:
            print("Creating condition balls...")
            self.create_condition_balls(motion_data, output_folder)
        
        # 비디오 렌더링
        if render_video:
            print("Rendering video...")
            video_path = os.path.join(output_folder, f"{npz_name}.mp4")
            self.render_video(mesh_folder, video_path, condition_folder)
        
        print(f"Visualization completed! Output saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='NPZ 파일로부터 120프레임 모션 시각화')
    parser.add_argument('--npz_file', type=str, 
                       help='시각화할 NPZ 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./motion_visualization',
                       help='출력 디렉토리 (기본값: ./motion_visualization)')
    parser.add_argument('--data_root', type=str, default='./processed_data',
                       help='데이터 루트 폴더 경로 (기본값: ./processed_data)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='사용할 디바이스 (기본값: cuda:0)')
    parser.add_argument('--no_video', action='store_true',
                       help='비디오 렌더링 스킵 (메쉬 파일만 생성)')
    parser.add_argument('--no_conditions', action='store_true',
                       help='조건 구 메쉬 생성하지 않음')
    parser.add_argument('--batch_process', type=str, default=None,
                       help='여러 NPZ 파일이 있는 폴더 경로 (배치 처리)')
    
    args = parser.parse_args()
    
    # 입력 검증
    if not args.npz_file and not args.batch_process:
        parser.error("--npz_file")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 시각화기 초기화
    print("Initializing motion visualizer...")
    visualizer = MotionVisualizer(args.data_root, args.device)
    
    if args.batch_process:
        # 배치 처리
        print(f"Batch processing NPZ files in: {args.batch_process}")
        npz_files = []
        for root, dirs, files in os.walk(args.batch_process):
            for file in files:
                if file.endswith('.npz'):
                    npz_files.append(os.path.join(root, file))
        
        print(f"Found {len(npz_files)} NPZ files")
        for i, npz_file in enumerate(npz_files):
            print(f"Processing {i+1}/{len(npz_files)}: {npz_file}")
            try:
                visualizer.visualize_motion(npz_file, args.output_dir, 
                                          render_video=not args.no_video,
                                          create_conditions=not args.no_conditions)
            except Exception as e:
                print(f"Error processing {npz_file}: {e}")
                continue
    else:
        # 단일 파일 처리
        visualizer.visualize_motion(args.npz_file, args.output_dir, 
                                  render_video=not args.no_video,
                                  create_conditions=not args.no_conditions)

if __name__ == "__main__":
    main()
