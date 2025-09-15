#!/usr/bin/env python3
"""
단일 모션에 대한 evaluation 스크립트
cleaned_sample.npz와 sample.npz를 비교하여 메트릭을 계산합니다.

Usage:
    python evaluate_single_motion.py --gt_npz path/to/cleaned_sample.npz --pred_npz path/to/sample.npz --data_root /path/to/data
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
from pathlib import Path

# 현재 스크립트의 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import required modules
from evaluation_metrics import compute_metrics
from trainer_chois import run_smplx_model
from body_model import BodyModel


class SingleMotionEvaluator:
    def __init__(self, data_root_folder, device='cuda:0'):
        """
        단일 모션 evaluation을 위한 클래스
        
        Args:
            data_root_folder (str): 데이터 루트 폴더 경로
            device (str): 사용할 디바이스 ('cuda:0' 또는 'cpu')
        """
        self.data_root_folder = data_root_folder
        self.device = device
        self.bm_dict = self._load_smpl_models()
        
    def _load_smpl_models(self):
        """SMPL body model dictionary 로드"""
        try:
            # SMPL 모델 파일 경로 설정
            support_base_dir = os.path.join(self.data_root_folder, 'support_files')
            surface_model_type = "smplx"
            
            male_model_path = os.path.join(support_base_dir, surface_model_type, "male", 'model.npz')
            female_model_path = os.path.join(support_base_dir, surface_model_type, "female", 'model.npz')
            
            # SMPL 모델 로드
            male_bm = BodyModel(bm_fname=male_model_path, num_betas=16)
            female_bm = BodyModel(bm_fname=female_model_path, num_betas=16)
            
            bm_dict = {'male': male_bm, 'female': female_bm}
            print("SMPL body models loaded successfully")
            return bm_dict
            
        except Exception as e:
            print(f"Error loading SMPL models: {e}")
            print("Please ensure SMPL model files are available in the data folder")
            return None
    
    def load_npz_data(self, npz_path):
        """NPZ 파일에서 모션 데이터 로드"""
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # 데이터 구조 확인
            motion_data = {}
            for key in data.keys():
                motion_data[key] = data[key]
            
            print(f"Loaded NPZ file: {npz_path}")
            print(f"Available keys: {list(motion_data.keys())}")
            
            return motion_data
            
        except Exception as e:
            print(f"Error loading NPZ file {npz_path}: {e}")
            return None
    
    def generate_human_mesh_from_smpl_params(self, motion_data):
        """
        SMPL 파라미터로부터 인간 메쉬 생성
        
        Args:
            motion_data (dict): NPZ에서 로드된 모션 데이터
            
        Returns:
            tuple: (mesh_joints, mesh_vertices, mesh_faces)
        """
        try:
            if self.bm_dict is None:
                print("SMPL models not loaded. Cannot generate human mesh.")
                return None, None, None
            
            # SMPL 파라미터 추출
            betas = torch.from_numpy(motion_data['betas']).float().to(self.device)
            gender = str(motion_data['gender']) if isinstance(motion_data['gender'], np.ndarray) else motion_data['gender']
            local_rot_aa = torch.from_numpy(motion_data['local_rot_aa']).float().to(self.device)  # T x 22 x 3
            root_trans = torch.from_numpy(motion_data['root_trans']).float().to(self.device)      # T x 3
            
            # SMPL 모델 실행
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
                root_trans[None],           # (1, T, 3)
                local_rot_aa[None],         # (1, T, 22, 3)
                betas,                      # (10,) or similar
                [gender],                   # ['male'] or ['female']
                self.bm_dict,               # SMPL body model dictionary
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
    
    def load_object_geometry(self, object_name):
        """
        객체의 rest pose geometry 로드
        
        Args:
            object_name (str): 객체 이름
            
        Returns:
            tuple: (obj_vertices, obj_faces)
        """
        try:
            # 객체 geometry 파일 경로 설정
            obj_geo_root_folder = os.path.join(self.data_root_folder, "captured_objects")
            obj_mesh_path = os.path.join(obj_geo_root_folder, f"{object_name}_cleaned_simplified.obj")
            
            # 대안 경로들 시도
            alternative_paths = [
                os.path.join(self.data_root_folder, "object_pcs", f"{object_name}_cleaned_simplified.ply"),
                os.path.join(obj_geo_root_folder, f"{object_name}_cleaned_simplified.ply"),
            ]
            
            if object_name in ["vacuum", "mop"]:
                alternative_paths.insert(0, 
                    os.path.join(self.data_root_folder, "object_pcs", f"{object_name}_cleaned_simplified_top.ply"))
            
            import trimesh
            mesh = None
            for path in [obj_mesh_path] + alternative_paths:
                if os.path.exists(path):
                    mesh = trimesh.load_mesh(path)
                    print(f"Loaded object geometry: {path}")
                    break
            
            if mesh is None:
                print(f"Could not find object geometry for {object_name}")
                return None, None
            
            obj_vertices = np.array(mesh.vertices, dtype=np.float32)
            obj_faces = np.array(mesh.faces, dtype=np.int32)
            
            return obj_vertices, obj_faces
            
        except Exception as e:
            print(f"Error loading object geometry for {object_name}: {e}")
            return None, None
    
    def generate_object_mesh_from_params(self, motion_data, obj_rest_verts):
        """
        객체 파라미터로부터 객체 메쉬 생성
        
        Args:
            motion_data (dict): 모션 데이터
            obj_rest_verts (np.ndarray): 객체 rest pose vertices
            
        Returns:
            torch.Tensor: 변환된 객체 vertices (T x Nv x 3)
        """
        try:
            if 'obj_rot_mat' not in motion_data or 'obj_com_pos' not in motion_data:
                print("Object transformation data not found in motion_data")
                seq_len = motion_data['seq_len']
                dummy_verts = torch.zeros(seq_len, obj_rest_verts.shape[0], 3).to(self.device)
                return dummy_verts
            
            obj_rot_mat = torch.from_numpy(motion_data['obj_rot_mat']).float().to(self.device)  # T x 3 x 3
            obj_com_pos = torch.from_numpy(motion_data['obj_com_pos']).float().to(self.device)  # T x 3
            obj_rest_verts = torch.from_numpy(obj_rest_verts).float().to(self.device)           # Nv x 3
            
            # 객체 변환 적용
            T = obj_rot_mat.shape[0]
            Nv = obj_rest_verts.shape[0]
            
            # Rest vertices를 각 프레임에 대해 복제
            rest_verts_expanded = obj_rest_verts[None].repeat(T, 1, 1)  # T x Nv x 3
            
            # 회전 적용
            rotated_verts = torch.bmm(obj_rot_mat, rest_verts_expanded.transpose(1, 2))  # T x 3 x Nv
            rotated_verts = rotated_verts.transpose(1, 2)  # T x Nv x 3
            
            # 평행이동 적용
            transformed_verts = rotated_verts + obj_com_pos[:, None, :]  # T x Nv x 3
            
            print(f"Generated object mesh: {transformed_verts.shape}")
            return transformed_verts
            
        except Exception as e:
            print(f"Error generating object mesh: {e}")
            seq_len = motion_data.get('seq_len', 120)
            dummy_verts = torch.zeros(seq_len, obj_rest_verts.shape[0], 3).to(self.device)
            return dummy_verts
    
    def evaluate_single_motion(self, gt_npz_path, pred_npz_path, save_results=True):
        """
        단일 모션에 대한 evaluation 수행
        
        Args:
            gt_npz_path (str): Ground truth NPZ 파일 경로
            pred_npz_path (str): Prediction NPZ 파일 경로
            save_results (bool): 결과를 파일로 저장할지 여부
            
        Returns:
            dict: evaluation 결과 딕셔너리
        """
        print("="*60)
        print("SINGLE MOTION EVALUATION")
        print("="*60)
        
        # NPZ 파일 로드
        print("\n1. Loading NPZ files...")
        gt_data = self.load_npz_data(gt_npz_path)
        pred_data = self.load_npz_data(pred_npz_path)
        
        if gt_data is None or pred_data is None:
            print("Failed to load NPZ files")
            return None
        
        # 기본 정보 확인
        seq_name = gt_data.get('seq_name', 'unknown')
        obj_name = gt_data.get('obj_name', 'unknown')
        seq_len = int(gt_data.get('seq_len', 120))
        
        print(f"Sequence: {seq_name}")
        print(f"Object: {obj_name}")
        print(f"Sequence length: {seq_len}")
        
        # Human mesh 생성
        print("\n2. Generating human meshes...")
        gt_human_jnts, gt_human_verts, human_faces = self.generate_human_mesh_from_smpl_params(gt_data)
        pred_human_jnts, pred_human_verts, _ = self.generate_human_mesh_from_smpl_params(pred_data)
        
        if gt_human_jnts is None or pred_human_jnts is None:
            print("Failed to generate human meshes")
            return None
        
        # Object geometry 로드
        print("\n3. Loading object geometry...")
        obj_rest_verts, obj_faces = self.load_object_geometry(obj_name)
        
        if obj_rest_verts is None:
            print(f"Warning: Could not load object geometry for {obj_name}")
            # 더미 객체 생성
            obj_rest_verts = np.zeros((100, 3), dtype=np.float32)
            obj_faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        # Object mesh 생성
        print("\n4. Generating object meshes...")
        gt_obj_verts = self.generate_object_mesh_from_params(gt_data, obj_rest_verts)
        pred_obj_verts = self.generate_object_mesh_from_params(pred_data, obj_rest_verts)
        
        # Evaluation 메트릭 계산
        print("\n5. Computing evaluation metrics...")
        try:
            # 필요한 데이터 준비
            gt_trans = torch.from_numpy(gt_data['root_trans']).float().to(self.device)
            pred_trans = torch.from_numpy(pred_data['root_trans']).float().to(self.device)
            
            # Rotation matrix 생성 (local_rot_aa에서 첫 번째 관절의 회전만 사용)
            gt_root_rot = torch.from_numpy(gt_data['local_rot_aa'][:, 0, :]).float().to(self.device)  # T x 3
            pred_root_rot = torch.from_numpy(pred_data['local_rot_aa'][:, 0, :]).float().to(self.device)  # T x 3
            
            # Axis-angle을 rotation matrix로 변환
            from pytorch3d.transforms import axis_angle_to_matrix
            gt_rot_mat = axis_angle_to_matrix(gt_root_rot)[:, None, :, :]  # T x 1 x 3 x 3 (22개 관절 중 root만)
            pred_rot_mat = axis_angle_to_matrix(pred_root_rot)[:, None, :, :]  # T x 1 x 3 x 3
            
            # Object transformation 데이터
            gt_obj_com_pos = torch.from_numpy(gt_data['obj_com_pos']).float().to(self.device)
            pred_obj_com_pos = torch.from_numpy(pred_data['obj_com_pos']).float().to(self.device)
            gt_obj_rot_mat = torch.from_numpy(gt_data['obj_rot_mat']).float().to(self.device)
            pred_obj_rot_mat = torch.from_numpy(pred_data['obj_rot_mat']).float().to(self.device)
            
            # compute_metrics 호출
            metrics = compute_metrics(
                gt_human_verts, pred_human_verts,           # ori_verts_gt, ori_verts_pred
                gt_human_jnts, pred_human_jnts,             # ori_jpos_gt, ori_jpos_pred
                human_faces,                                 # human_faces
                gt_trans[None], pred_trans[None],           # gt_trans, pred_trans (add batch dim)
                gt_rot_mat, pred_rot_mat,                   # gt_rot_mat, pred_rot_mat
                gt_obj_com_pos, pred_obj_com_pos,           # gt_obj_com_pos, pred_obj_com_pos
                gt_obj_rot_mat, pred_obj_rot_mat,           # gt_obj_rot_mat, pred_obj_rot_mat
                gt_obj_verts, pred_obj_verts,               # gt_obj_verts, pred_obj_verts
                obj_faces,                                   # obj_faces
                seq_len,                                     # actual_len
                use_joints24=True
            )
            
            # 결과 정리
            metric_names = [
                'lhand_jpe', 'rhand_jpe', 'hand_jpe', 'mpvpe', 'mpjpe', 'rot_dist', 'trans_err',
                'gt_contact_percent', 'contact_percent', 'gt_foot_sliding_jnts', 'foot_sliding_jnts',
                'contact_precision', 'contact_recall', 'contact_acc', 'contact_f1_score',
                'obj_rot_dist', 'obj_com_pos_err', 'start_obj_com_pos_err', 'end_obj_com_pos_err',
                'waypoints_xy_pos_err', 'gt_floor_height', 'pred_floor_height'
            ]
            
            results = dict(zip(metric_names, metrics))
            
            # 추가 정보
            results['seq_name'] = seq_name
            results['obj_name'] = obj_name
            results['seq_len'] = seq_len
            results['gt_npz_path'] = gt_npz_path
            results['pred_npz_path'] = pred_npz_path
            
            # 결과 출력
            print("\n6. Evaluation Results:")
            print("-" * 40)
            print(f"Hand Joint Position Error (mm):")
            print(f"  - Left Hand JPE: {results['lhand_jpe']:.2f}")
            print(f"  - Right Hand JPE: {results['rhand_jpe']:.2f}")
            print(f"  - Average Hand JPE: {results['hand_jpe']:.2f}")
            print(f"\nBody Metrics (mm):")
            print(f"  - MPVPE: {results['mpvpe']:.2f}")
            print(f"  - MPJPE: {results['mpjpe']:.2f}")
            print(f"  - Translation Error: {results['trans_err']:.2f}")
            print(f"  - Rotation Distance: {results['rot_dist']:.4f}")
            print(f"\nFoot Sliding:")
            print(f"  - GT Foot Sliding: {results['gt_foot_sliding_jnts']:.2f}")
            print(f"  - Pred Foot Sliding: {results['foot_sliding_jnts']:.2f}")
            print(f"\nContact Metrics:")
            print(f"  - Contact Precision: {results['contact_precision']:.3f}")
            print(f"  - Contact Recall: {results['contact_recall']:.3f}")
            print(f"  - Contact F1-Score: {results['contact_f1_score']:.3f}")
            print(f"\nObject Metrics (mm):")
            print(f"  - Object COM Error: {results['obj_com_pos_err']:.2f}")
            print(f"  - Object Rotation Distance: {results['obj_rot_dist']:.4f}")
            print(f"  - Start Position Error: {results['start_obj_com_pos_err']:.2f}")
            print(f"  - End Position Error: {results['end_obj_com_pos_err']:.2f}")
            print(f"  - Waypoints XY Error: {results['waypoints_xy_pos_err']:.2f}")
            
            # 결과 저장
            if save_results:
                output_dir = Path(gt_npz_path).parent / "evaluation_results"
                output_dir.mkdir(exist_ok=True)
                
                result_filename = f"{seq_name}_{obj_name}_evaluation.json"
                result_path = output_dir / result_filename
                
                # JSON으로 저장
                json_results = {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                              for k, v in results.items()}
                
                with open(result_path, 'w') as f:
                    json.dump(json_results, f, indent=2)
                
                print(f"\nResults saved to: {result_path}")
            
            return results
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate single motion sequence')
    parser.add_argument('--gt_npz', required=True, help='Path to ground truth NPZ file')
    parser.add_argument('--pred_npz', required=True, help='Path to prediction NPZ file')
    parser.add_argument('--data_root', required=True, help='Path to data root folder')
    parser.add_argument('--device', default='cuda:0', help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--no_save', action='store_true', help='Do not save results to file')
    
    args = parser.parse_args()
    
    # 파일 경로 확인
    if not os.path.exists(args.gt_npz):
        print(f"Ground truth NPZ file not found: {args.gt_npz}")
        return
    
    if not os.path.exists(args.pred_npz):
        print(f"Prediction NPZ file not found: {args.pred_npz}")
        return
    
    if not os.path.exists(args.data_root):
        print(f"Data root folder not found: {args.data_root}")
        return
    
    # Evaluator 생성 및 실행
    evaluator = SingleMotionEvaluator(args.data_root, device=args.device)
    results = evaluator.evaluate_single_motion(
        args.gt_npz, 
        args.pred_npz, 
        save_results=not args.no_save
    )
    
    if results is None:
        print("Evaluation failed")
        return
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
