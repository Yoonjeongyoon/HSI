from manip.lafan1.utils import extract_feet_contacts
from footskate import Cleaner, calculate_skating_ratio
import numpy as np
import torch
import pytorch3d.transforms as transforms 
from scipy.ndimage import uniform_filter1d
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
def ranges_from_binary(contact_1d_bool):

    contact_1d_bool = contact_1d_bool.astype(bool)
    T = len(contact_1d_bool)
    ranges, in_run, start = [], False, 0
    for t in range(T):
        if contact_1d_bool[t] and not in_run:
            in_run, start = True, t
        elif not contact_1d_bool[t] and in_run:
            in_run = False
            ranges.append((start, t))
    if in_run:
        ranges.append((start, T))
    return ranges
def print_contact_ranges(left_foot_contacts, right_foot_contacts):
    """
    left_foot_contacts, right_foot_contacts: (T,2) ndarray/bool
      [:,0] = ankle, [:,1] = toe
    """
    print("=== Contact Ranges (frame indices) ===")
    names = [
        ("Left Ankle",  left_foot_contacts[:,0]),
        ("Left Toe",    left_foot_contacts[:,1]),
        ("Right Ankle", right_foot_contacts[:,0]),
        ("Right Toe",   right_foot_contacts[:,1]),
    ]
    for name, seq in names:
        ranges = ranges_from_binary(seq)
        print(f"{name}: {ranges}")

def clean_footskate(body_data):
    '''
        np.savez(npz_path,
                # 시퀀스 정보
                seq_name=curr_seq_name,
                obj_name=object_name,
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                seq_len=actual_len,
                step=step,
                idx=idx,
                rest_human_offsets = rest_human_offsets,
                
                # SMPL 파라미터들 (현재 루프에서 계산된 값들)
                betas=data_dict['betas'][0].detach().cpu().numpy(),  # Body shape parameters
                gender=data_dict['gender'][0],  # Gender
                local_rot_aa=curr_local_rot_aa_rep.detach().cpu().numpy(),  # T X 22 X 3 - Local rotations in axis-angle
                local_rot_quat=curr_local_rot_quat.detach().cpu().numpy(),  # T X 22 X 4 - Local rotations in quaternion
                root_trans=root_trans.detach().cpu().numpy(),  # T X 3 - Root translations
                global_jpos=curr_global_jpos.detach().cpu().numpy(),  # T X 24 X 3 - Global joint positions
                # 추가 유용한 정보
                obj_rot_mat=data_dict['obj_rot_mat'][0].detach().cpu().numpy() if 'obj_rot_mat' in data_dict else None,
                obj_com_pos=data_dict['obj_com_pos'][0].detach().cpu().numpy() if 'obj_com_pos' in data_dict else None,
                )   
    '''
    # 원본 데이터 백업
    original_root_trans = body_data['root_trans'].copy()
    original_local_rot_quat = body_data['local_rot_quat'].copy()
    global_jpos = body_data['global_jpos']
    # calculate_skating_ratio는 numpy 배열 [bs, T, 22, 3] 형태를 기대하므로 차원을 맞춰줌
    # global_jpos는 [T, 24, 3] 형태이므로 뒤의 2개 관절을 빼서 22개로 만들고 배치 차원 추가
    global_jpos_22 = global_jpos[:, :-2, :]  # [T, 22, 3]
    global_jpos_feet = np.expand_dims(global_jpos_22, axis=0)  # [1, T, 22, 3] - numpy로 차원 맞춤
    skating_ratio, skate_vel = calculate_skating_ratio(global_jpos_feet)
    print(f"Initial Skating ratio: {skating_ratio}")
    left_foot_contacts, right_foot_contacts = extract_feet_contacts(global_jpos, lfoot_idx=[7, 10], rfoot_idx=[8, 11], velfactor=0.02, start_th=None, stop_th=None) # (120, 2)
    print_contact_ranges(left_foot_contacts, right_foot_contacts)
    
    # 텐서보드 로거 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/footskate_cleaning_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"텐서보드 로그 저장 위치: {log_dir}")
    print(f"텐서보드 실행 명령어: tensorboard --logdir=runs")
    
    # 초기 스케이팅 비율 기록
    writer.add_scalar('Metrics/Initial_Skating_Ratio', skating_ratio[0], 0)
    
    # 역할 기반 가중치 설정: C(주목표) >> T(보조) >> Q(규제)
    # 현재 Loss 크기: C=7.0, T=0.04, Q=0.000007
    # 원본 UnderPressure 가중치 기반으로 CHOIS 스케일에 맞게 조정
    cleaner = Cleaner(
        iterations=200,  
        tweight=2e2,    
        cweight=1e-2,   
        zweight=1e-3,   
        margin=2,       # 원본과 동일
        device="cuda",
        tensorboard_writer=writer  # 텐서보드 writer 전달
    )
    
    try:
        cleaned_angles, cleaned_trajectory = cleaner(
            angles=torch.from_numpy(body_data['local_rot_quat']).float(), 
            skeleton=torch.from_numpy(body_data['rest_human_offsets']).float(), 
            trajectory=torch.from_numpy(body_data['root_trans']).float(),
            trans2joint=body_data['trans2joint'],
            left_foot_contacts=torch.from_numpy(left_foot_contacts).float(),
            right_foot_contacts=torch.from_numpy(right_foot_contacts).float(),
        )
        
        # 결과 검증 - 너무 큰 변화가 있으면 원본 사용
        original_root_trans_tensor = torch.from_numpy(original_root_trans).float().to(cleaned_trajectory.device)
        original_local_rot_quat_tensor = torch.from_numpy(original_local_rot_quat).float().to(cleaned_angles.device)
        trajectory_diff = torch.norm(cleaned_trajectory - original_root_trans_tensor)
        angle_diff = torch.norm(cleaned_angles - original_local_rot_quat_tensor)

        
        # 쿼터니언 norm은 자연스럽게 큰 값을 가짐 (T×J×4 차원)
        # trajectory_diff: 미터 단위 변화량, angle_diff: 쿼터니언 전체 차이 norm
        if trajectory_diff > 5.0 or angle_diff > 50.0:  # 현실적인 임계값 설정
            print(f"경고: 너무 큰 변화 감지됨 (trajectory_diff: {trajectory_diff:.2f}, angle_diff: {angle_diff:.2f})")
            print("원본 데이터를 유지합니다.")
            return body_data
        
        body_data['local_rot_quat'] = cleaned_angles.detach().cpu().numpy()
        body_data['root_trans'] = cleaned_trajectory.detach().cpu().numpy()
        cleaned_axis_angle = transforms.quaternion_to_axis_angle(cleaned_angles.detach().cpu())
        body_data['local_rot_aa'] = cleaned_axis_angle.detach().cpu().numpy()
        
        # 전역 관절 위치 재계산 (SMPL FK 필요)
        # 현재는 기존 global_jpos를 유지하지만, 정확한 결과를 위해서는 
        # SMPL forward kinematics로 다시 계산해야 함
        print("Footskate cleaning 완료")
        print(f"Trajectory 변화량: {trajectory_diff:.3f}")
        print(f"Angle 변화량: {angle_diff:.3f}")
        
        # 텐서보드 writer 닫기
        writer.close()
        print("텐서보드 로깅 완료")
        
    except Exception as e:
        import traceback
        print(f"Footskate cleaning 실패: {e}")
        print("전체 스택 트레이스:")
        traceback.print_exc()
        print("원본 데이터를 유지합니다.")
        # 원본 데이터로 복원
        body_data['root_trans'] = original_root_trans
        body_data['local_rot_quat'] = original_local_rot_quat
        
        # 예외 발생 시에도 writer 닫기
        if 'writer' in locals():
            writer.close()
    
    return body_data    

def clean_motion(rest_human_offsets, local_rot_quat, root_trans, trans2joint, global_jpos):

    # Convert inputs to numpy if they are tensors
    if torch.is_tensor(rest_human_offsets):
        rest_human_offsets = rest_human_offsets.detach().cpu().numpy()
    if torch.is_tensor(local_rot_quat):
        local_rot_quat = local_rot_quat.detach().cpu().numpy()
    if torch.is_tensor(root_trans):
        root_trans = root_trans.detach().cpu().numpy()
    if torch.is_tensor(trans2joint):
        trans2joint = trans2joint.detach().cpu().numpy()
    if torch.is_tensor(global_jpos):
        global_jpos = global_jpos.detach().cpu().numpy()
    
    
    # Contact detection
    left_foot_contacts, right_foot_contacts = extract_feet_contacts(
        global_jpos, lfoot_idx=[7, 10], rfoot_idx=[8, 11], velfactor=0.02, start_th=None, stop_th=None
    )
    
    cleaner = Cleaner(
        iterations=200,  
        tweight=10,    
        cweight=1e-2,  
        zweight=1e-3,   
        margin=2,       
        device="cuda"
    )
    

    cleaned_angles, cleaned_trajectory = cleaner(
        angles=torch.from_numpy(local_rot_quat).float(), 
        skeleton=torch.from_numpy(rest_human_offsets).float(), 
        trajectory=torch.from_numpy(root_trans).float(),
        trans2joint=trans2joint,
        left_foot_contacts=torch.from_numpy(left_foot_contacts).float(),
        right_foot_contacts=torch.from_numpy(right_foot_contacts).float(),
    )
        
    return cleaned_angles, cleaned_trajectory



if __name__ == "__main__":
    loaded = np.load("plas.npz", allow_pickle=True)
    body_data = {key: loaded[key] for key in loaded.files}
    clean_data=clean_footskate(body_data)   
    np.savez("clean_plas.npz", **clean_data)
