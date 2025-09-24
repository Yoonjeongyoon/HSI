from footskate import Cleaner
import numpy as np
import torch
import pytorch3d.transforms as transforms 
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
#발 접촉여부 판단 코드  
def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02, min_len=10, fill_gap=3, start_th=None, stop_th=None):
    def hysteresis_mask(vel, s_th, e_th):
        contact = np.zeros_like(vel, dtype=bool)
        in_contact = False
        for t, v in enumerate(vel):
            if not in_contact and v < s_th:
                in_contact = True
            elif in_contact and v > e_th:
                in_contact = False
            contact[t] = in_contact
        return np.concatenate([contact, contact[-1:]], axis=0)

    def joint_contact(idx):
        v = np.linalg.norm(pos[1:, idx, :] - pos[:-1, idx, :], axis=-1)  # (T-1,)
        s_th = start_th if start_th is not None else max(1e-12, 0.95 * velfactor)
        e_th = stop_th  if stop_th  is not None else max(s_th + 1e-12, 1.05 * velfactor)  # ensure s_th < e_th
        return hysteresis_mask(v, s_th, e_th)  # (T,)
    # 각 관절별로 따로 속도 → 접촉 마스크 (거리/frame 기준)
    def runs_from_mask(mask):
            T = len(mask)
            iv, in_run, s = [], False, 0
            for t in range(T):
                v = bool(mask[t])
                if v and not in_run:
                    in_run, s = True, t
                elif not v and in_run:
                    in_run = False
                    iv.append((s, t))
            if in_run:
                iv.append((s, T))
            return iv

    def clean_intervals(intervals, min_len=1, fill_gap=0):
        if not intervals:
            return []
        # 1) 짧은 구간 제거
        iv = [(s, e) for (s, e) in intervals if (e - s) >= min_len]
        if not iv:
            return []
        # 2) 짧은 갭 병합
        iv.sort()
        merged = [iv[0]]
        for s, e in iv[1:]:
            ps, pe = merged[-1]
            if s - pe <= fill_gap:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def mask_from_intervals(intervals, T):
        m = np.zeros(T, dtype=bool)
        for s, e in intervals:
            m[s:e] = True
        return m

    # 관절별 접촉 마스크 (T,)
    l_ank = joint_contact(lfoot_idx[0])
    l_toe = joint_contact(lfoot_idx[1])
    r_ank = joint_contact(rfoot_idx[0])
    r_toe = joint_contact(rfoot_idx[1])

    T = pos.shape[0]
    # 후처리: 최소 길이 & 짧은 갭 메우기
    l_ank = mask_from_intervals(clean_intervals(runs_from_mask(l_ank), min_len, fill_gap), T)
    l_toe = mask_from_intervals(clean_intervals(runs_from_mask(l_toe), min_len, fill_gap), T)
    r_ank = mask_from_intervals(clean_intervals(runs_from_mask(r_ank), min_len, fill_gap), T)
    r_toe = mask_from_intervals(clean_intervals(runs_from_mask(r_toe), min_len, fill_gap), T)

    # (T,2)로 스택: [:,0]=ankle(heel), [:,1]=toe
    left_foot_contacts  = np.stack([l_ank, l_toe], axis=1)
    right_foot_contacts = np.stack([r_ank, r_toe], axis=1)

    return left_foot_contacts, right_foot_contacts
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
def clean_footskate(body_data):

    global_jpos = body_data['global_jpos']
    # global_jpos는 [T, 24, 3] 형태이므로 뒤의 2개 관절을 빼서 22개로 만들고 배치 차원 추가
    global_jpos_22 = global_jpos[:, :-2, :]  # [T, 22, 3]

    left_foot_contacts, right_foot_contacts = extract_feet_contacts(global_jpos, lfoot_idx=[7, 10], rfoot_idx=[8, 11], velfactor=0.02, start_th=None, stop_th=None) # (120, 2)
    # 텐서보드 로거 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/footskate_cleaning_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    cleaner = Cleaner(
        iterations=200,  
        tweight=1,    
        cweight=1e-2,   
        zweight=1e-5,
        Vxyweight=100,   
        margin=2,       
        device="cuda",
        tensorboard_writer=writer  
    )
    cleaned_angles, cleaned_trajectory = cleaner(
        angles=torch.from_numpy(body_data['local_rot_quat']).float(), 
        skeleton=torch.from_numpy(body_data['rest_human_offsets']).float(), 
        trajectory=torch.from_numpy(body_data['root_trans']).float(),
        trans2joint=body_data['trans2joint'],
        left_foot_contacts=torch.from_numpy(left_foot_contacts).float(),
        right_foot_contacts=torch.from_numpy(right_foot_contacts).float(),
    )
    body_data['local_rot_quat'] = cleaned_angles.detach().cpu().numpy()
    body_data['root_trans'] = cleaned_trajectory.detach().cpu().numpy()
    cleaned_axis_angle = transforms.quaternion_to_axis_angle(cleaned_angles.detach().cpu())
    body_data['local_rot_aa'] = cleaned_axis_angle.detach().cpu().numpy() 
    writer.close()
    return body_data    
# trainer_chois.py에서 사용할 경우  
def clean_motion(rest_human_offsets, local_rot_quat, root_trans, trans2joint, global_jpos):
    # Convert inputs to numpy
    rest_human_offsets = rest_human_offsets.detach().cpu().numpy()
    local_rot_quat = local_rot_quat.detach().cpu().numpy()
    root_trans = root_trans.detach().cpu().numpy()
    trans2joint = trans2joint.detach().cpu().numpy()
    global_jpos = global_jpos.detach().cpu().numpy()
    # Contact detection
    left_foot_contacts, right_foot_contacts = extract_feet_contacts(
        global_jpos, lfoot_idx=[7, 10], rfoot_idx=[8, 11], velfactor=0.02, start_th=None, stop_th=None
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/footskate_cleaning_{timestamp}"
    writer = SummaryWriter(log_dir)
    cleaner = Cleaner(
        iterations=200,  
        tweight=1,    
        cweight=1e-2,   
        zweight=1e-5,
        Vxyweight=100,  
        margin=2,       
        device="cuda",
        tensorboard_writer=writer
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
