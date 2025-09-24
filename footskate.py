import pytorch3d.transforms as transforms
import anim, util
import torch
import numpy as np
import math
from typing import Tuple, Dict
from torch.nn import functional as F

def quat_slerp(q0, q1, t):
    # q0,q1:[T,J,4], t:[T,1,1] or scalar
    # 코사인 각
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # [T,J,1]
    # 같은 방향 유지(부호 뒤집어 단거리 경로)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()

    # 소각 근사(선형 보간)
    eps = 1e-8
    use_lerp = dot > (1.0 - 1e-6)
    # 각도
    theta = torch.acos(dot.clamp(-1+eps, 1-eps))
    sin_theta = torch.sin(theta)

    # slerp 계수
    w0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w1 = torch.sin(t * theta) / (sin_theta + eps)

    out = w0 * q0 + w1 * q1
    out = torch.where(use_lerp, (1.0 - t) * q0 + t * q1, out)
    return out

def quat_normalize(q, eps=1e-8):
        norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=eps)
        qn = q / norm
        flip = (qn[..., :1] < 0).to(qn.dtype)
        qn = torch.where(flip.bool().expand_as(qn), -qn, qn)
        return qn
#  가중치 구현하는 클래스 
class ContactWeightFactory:
    def __init__(self,
                 device="cpu",
                 margin:int=3,
                 mode:str="gauss",
                 normalize:str=None, 
                 poly_p:int=2,        
                 gauss_sigma:float=None, 
                 eps:float=1e-6):
        self.device = device
        self.margin = int(margin)
        self.mode = mode
        self.normalize = normalize
        self.poly_p = int(poly_p)
        self.gauss_sigma = gauss_sigma
        self.eps = eps

        self._cache: Dict[Tuple, torch.Tensor] = {}

    def get_weights(self, length:int) -> torch.Tensor:
        key = (self.mode, int(length), int(self.margin), int(self.poly_p),
               float(self.gauss_sigma) if self.gauss_sigma is not None else None,
               self.normalize)
        if key not in self._cache:
            w = self._make_weights(length)
            if self.normalize == 'max':
                w = w / (w.max().clamp_min(self.eps))
            elif self.normalize == 'sum':
                w = w / (w.sum().clamp_min(self.eps))
            self._cache[key] = w
        return self._cache[key]

    def _make_weights(self, L:int) -> torch.Tensor:
        if L <= 0:
            return torch.empty(0, device=self.device)
        if L == 1:
            return torch.ones(1, device=self.device)

        if self.mode == 'gauss':
            return self._gauss_center(L, self.gauss_sigma)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _cosine_taper(self, L:int, m:int) -> torch.Tensor:

        m = max(0, min(int(m), L//2))
        if 2*m >= L:
            # Hann window (non-periodic)
            w = torch.hann_window(L, periodic=False, device=self.device)
            return w.clamp_min(self.eps)

        w = torch.ones(L, device=self.device)
        ramp = 0.5 * (1 - torch.cos(math.pi * torch.linspace(0, 1, m, device=self.device)))
        w[:m] = ramp
        w[L-m:] = torch.flip(ramp, dims=[0])
        return w.clamp_min(self.eps)

    def _gauss_center(self, L:int, sigma:float=None) -> torch.Tensor:
        n = torch.arange(L, device=self.device, dtype=torch.float32)
        c = 0.5*(L-1)  # 중앙
        if sigma is None:
            # L이 길어질수록 plateau 느낌을 조금 주되 꼬리는 짧게
            sigma = max(1.0, 0.18 * L)  # 경험적 기본값
        w = torch.exp(-0.5 * ((n - c)/sigma)**2)
        return w.clamp_min(self.eps)

def _mask_from_ranges_nested(T, contact_ranges, device, margin=0):
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    for fb in range(len(contact_ranges)):    
        for lr in range(len(contact_ranges[fb])):  
            for (s, e) in contact_ranges[fb][lr]:
                s = max(0, s - margin)
                e = min(T, e + margin)
                if s < e:
                    mask[s:e] = True
    return mask
#단위 쿼터니언 생성 
def identity_quaternion(shape, device=None, dtype=None):
    quat = torch.zeros(*shape, 4, device=device, dtype=dtype)
    quat[..., 3] = 1.0
    return quat
    # 가우시안 이동평균 (smooth 용)
def gaussian_moving_average(x, size, std, dim=-3, zero_phase=True):

    if size <= 1:
        return x
    if size % 2 == 0:
        raise ValueError(f"kernel size must be odd, got {size}")
    sigma = float(max(std, 1e-6))
    # 가우시안 커널 생성
    k = torch.arange(size, dtype=x.dtype, device=x.device) - (size - 1) / 2
    kernel = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    kernel = kernel / (kernel.sum() + torch.finfo(x.dtype).eps) 
    kernel = kernel.view(1, 1, -1)
    
    # Convolution을 위해 차원 조정
    original_shape = x.shape
    if dim < 0:
        dim = len(original_shape) + dim
    
    # 차원을 마지막으로 이동
    x_moved = x.transpose(dim, -1)
    T = x_moved.shape[-1]
    C = int(x_moved.numel() // T)
    # 1D convolution 적용
    x_flat = x_moved.reshape(1, C, T)     # (1, C, T)
    pad = (size - 1) // 2
    x_pad = F.pad(x_flat, (pad, pad), mode='reflect')  
    y = F.conv1d(x_pad, kernel.expand(C, 1, -1), padding=0, groups=C)
    
    # Padding
    if zero_phase:
        y_rev = torch.flip(y, dims=[-1])
        y_rev = F.pad(y_rev, (pad, pad), mode='reflect')
        y2 = F.conv1d(y_rev, kernel.expand(C, 1, -1), padding=0, groups=C)
        y = torch.flip(y2, dims=[-1])

    # 차원을 원래대로 복원
    y = y.reshape(x_moved.shape).transpose(-1, dim).contiguous()
    
    return y
FRAMERATE = 30
TOPOLOGY = anim.Topology([
    ("pelvis",			[]),  #0
    ("left_hip",		["pelvis"]),  #1
    ("right_hip",		["pelvis"]),  #2
    ("spine1",			["pelvis"]),  #3
    ("left_knee",		["left_hip"]),  #4
    ("right_knee",		["right_hip"]),  #5
    ("spine2",			["spine1"]),  #6
    ("left_ankle",		["left_knee"]),  #7
    ("right_ankle",		["right_knee"]),  #8
    ("spine3",			["spine2"]), #9
    ("left_foot",		["left_ankle"]), #10
    ("right_foot",		["right_ankle"]),  	 #11
    ("neck",			["spine3"]),  		 #12
    ("left_collar",		["spine3"]),  	 #13
    ("right_collar",	["spine3"]),  	 #14
    ("head",			["neck"]),  #15
    ("left_shoulder",	["left_collar"]),  #16
    ("right_shoulder",	["right_collar"]),  	 #17
    ("left_elbow",		["left_shoulder"]),  		 #18
    ("right_elbow",		["right_shoulder"]),  	 #19
    ("left_wrist",		["left_elbow"]),	 #20
    ("right_wrist",		["right_elbow"]),	 #21
])
# extended topology with contact joints
# 기존 관절에 4개 접촉 관절 추가 접촉관절은 부모관절의 자식으로 설정 
EXTENDED_TOPOLOGY = anim.Topology(
    list(zip(TOPOLOGY.joints(), TOPOLOGY.parents())) + [
    ("right_ankle_contacts",	["right_ankle"]),  #22
    ("right_foot_contacts",		["right_foot"]),  #23
    ("left_ankle_contacts",		["left_ankle"]),   #24
    ("left_foot_contacts",		["left_foot"]),  #25
])
# 속도 계산 함수
def get_velocity(trajectory):
    return trajectory.diff(dim=0) * FRAMERATE

def quat_fk_torch(lrot_quat, lpos_rest, root_trans, use_joints26=True):
    if use_joints26:
        parents = get_smpl_parents(use_joints26=True)
        J=26
    else:
        parents = get_smpl_parents()
        J = 22 
    Tlen=120
    # 2) lpos_rest 브로드캐스트 보정 (시간불변이면 [1,J,3] → [T,J,3])
    if lpos_rest.ndim == 3 and lpos_rest.shape[0] == 1:
        lpos = lpos_rest.expand(lrot_quat.shape[0], -1, -1).contiguous()
    else:
        lpos = lpos_rest  # 이미 [T,J,3]

    grot_list = []
    gpos_list = []

    root = 0
    grot_root = quat_normalize(lrot_quat[:, root, :])  # 루트의 global = local (기준)
    # root_trans를 동일한 배치 크기로 확장
    T = lrot_quat.shape[0]
    if root_trans.shape[0] != T:
        gpos_root = root_trans.expand(T, -1).to(lrot_quat.device)  # [T, 3]로 확장
    else:
        gpos_root = root_trans.to(lrot_quat.device)
    
    grot_list.append(grot_root)
    gpos_list.append(gpos_root)

    for j in range(1, J):
        p = parents[j]
        grot_j = transforms.quaternion_multiply(grot_list[p], lrot_quat[:, j, :])
        grot_j = quat_normalize(grot_j)
        grot_list.append(grot_j)
        # 위치: pg_j = Rg_p ∘ lpos_j + pg_p (새로운 tensor 생성)
        rotated = transforms.quaternion_apply(grot_list[p], lpos[:, j, :])
        gpos_j = rotated + gpos_list[p]
        gpos_list.append(gpos_j)
    
    # 최종 tensor로 변환
    grot = torch.stack(grot_list, dim=1)  # [T, J, 4]
    gpos = torch.stack(gpos_list, dim=1)  # [T, J, 3]

    return grot, gpos

def get_smpl_parents(use_joints26=True):
    ori_kintree_table = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 8, 11, 7, 10]
    if use_joints26:
        parents = np.asarray(ori_kintree_table)
    else: # 첫 22개 요소 
        parents = np.asarray(parents)
    return parents

class Feet:
    JOINTS = [
        ["left_foot_contacts", "right_foot_contacts"],    
        ["left_ankle_contacts", "right_ankle_contacts"],  
    ]
    JIDXS = torch.as_tensor([[EXTENDED_TOPOLOGY.index(joint) for joint in joints] for joints in JOINTS])	
    LUT = EXTENDED_TOPOLOGY.lut(TOPOLOGY)
    
    @classmethod
    # 4개 발 접촉관절 확장 부분 smplx 는 상대 회전과 오프셋 정보가 있음  따라서 월드 좌표계에서 z 축으로 일부만큼 빼려면 월드 좌표계와 해당 좌표계의 변환을 먼저 진행하고 스케일링 해주어야 함... 
    def extend(cls, angles, skeleton, trajectory):
        # SMPL은 22관절이므로 njoints = 22+ 4 = 26
        njoints = TOPOLOGY.njoints + 4  # SMPL 기본 22관절 + 발 접촉 4관절

        # extend angles
        angles_26 = torch.zeros(*angles.shape[:-2], njoints, 4).to(angles) 
        angles_26[..., cls.LUT, :] = angles
        quat_global, _ = quat_fk_torch(angles, skeleton, trajectory, use_joints26=False) # [T,22,4]
                
        # extend skeleton
        # right_ankle_contacts, right_foot_contacts, left_ankle_contacts, left_foot_contacts 순서 22 23 24 25 
        angles_ref = angles[0:1, :, :]                         # [1,22,4]
        skeleton_ref = skeleton[0:1, :, :]                     # [1,22,3]로 맞춤
        root_zero  = torch.zeros(1, 3, device=angles.device, dtype=angles.dtype)
        grot_ref, _ = quat_fk_torch(angles_ref, skeleton_ref, root_zero, use_joints26=False)  # [1,22,4]
        skeleton_extended = torch.zeros(*skeleton.shape[:-2], njoints, skeleton.shape[-1], 
                                        device=skeleton.device, dtype=skeleton.dtype)
        skeleton_extended[..., cls.LUT, :] = skeleton  # 원본 22관절 복사 (보통 [1,22,3])
            # 월드 z축 단위벡터  나중에 더 좋은 방법 고려 
        z_hat = torch.tensor([0., 0., 1], device=skeleton.device, dtype=skeleton.dtype).view(1, 3)

        # 각 접촉 관절의 부모와 하강량(d): (new_idx, parent_idx, drop)
        specs = [
            (24,  7,  float(0.06)),   # L_ankle_contact  ← L_ankle
            (22,  8,  float(0.06)),  # R_ankle_contact  ← R_ankle
            (25, 10,  float(0.02)),    # L_foot_contact   ← L_toe
            (23, 11,  float(0.02)),   # R_foot_contact   ← R_toe
        ]
        # 프레임 0의 부모 전역 회전으로만 변환하여 '고정' 로컬 오프셋 생성
        for new_idx, parent_idx, d in specs:

            delta_world = (-d) * z_hat                              # [1,3]
            # 부모 글로벌 회전(프레임0)의 역으로 월드→로컬
            R_p_ref     = grot_ref[:, parent_idx, :]                # [1,4]
            R_p_ref_inv = transforms.quaternion_invert(R_p_ref)     # [1,4]
            delta_local = transforms.quaternion_apply(R_p_ref_inv, delta_world)  # [1,3]
            # 새 관절의 상대 오프셋 = delta_local (시간불변)
            skeleton_extended[..., new_idx, :] = delta_local
        q_target_global = torch.tensor([0., 1., 0., 0.], device=angles.device, dtype=angles.dtype) \
                    .view(1, 4).expand(angles.shape[0], 4) 
        skeleton = skeleton_extended  # [1,26,3]
        for new_idx, parent_idx, _ in specs:
            Qp = quat_global[:, parent_idx, :]  # [T,4] 부모 글로벌 회전
            q_loc = torch.tensor([0., 0., 0., 1.], device=angles.device, dtype=angles.dtype).expand_as(Qp) #2025 09 16 수정 새롭게 추가 그냥 0 0 0 1 로 초기화 해보자
            angles_26[:, new_idx, :] = q_loc
        return angles_26, skeleton

    @classmethod
    def reduce(cls, angles, skeleton):
        # 22개 관절만 선택
        original_joint_count = 22 
        angles = angles[..., :original_joint_count, :]
        skeleton = skeleton[..., :original_joint_count, :]
        return angles, skeleton

class Cleaner:
    def __init__(self, iterations: int, tweight=1e2, cweight=1e-5, zweight=1e-3, Vxyweight=1, margin=0, device="cpu", tensorboard_writer=None):
        self._niters = int(iterations)
        self._weights = dict(T=float(tweight), C=float(cweight), Z=float(zweight), Vxy=float(Vxyweight))
        self._margin = int(margin)
        self._device = device
        self._writer = tensorboard_writer

    @property
    def tloss(self) -> bool:
        return isinstance(self._weights["T"], float) and self._weights["T"] > 0.0
    @property
    def closs(self) -> bool:
        return isinstance(self._weights["C"], float) and self._weights["C"] > 0.0
    @property
    def zloss(self) -> bool:
        return isinstance(self._weights["Z"], float) and self._weights["Z"] > 0.0
    @property
    def Vxyloss(self) -> bool:
        return isinstance(self._weights["Vxy"], float) and self._weights["Vxy"] > 0.0
    @property
    def device(self):
        return self._device

    @classmethod
    def smooth(cls, angles, skeleton, size, std, contact_ranges, contact_margin=2, w_all=1e-1, w_xy_foot=2e-2, w_z_foot=5e-1): # 전역관절 위치를 시간축으로 가우시안 이동평균해서 타겟
        # 입력 텐서들을 동일한 장치로 이동
        device = angles.device
        skeleton = skeleton.to(device)
        angles = quat_normalize(angles)
        T = angles.shape[0]
        dummy_trajectory = torch.zeros(angles.shape[0], 3, device=angles.device, dtype=angles.dtype)
        # 원본 각도로 타겟 위치 계산하고 완전히 detach하여 그래프에서 분리
        with torch.no_grad():
            _, positions_for_targets = quat_fk_torch(angles.detach(), skeleton.detach(), dummy_trajectory, use_joints26=False)
            targets = gaussian_moving_average(positions_for_targets, size, std, dim=-3, zero_phase=True)
            positions_for_targets = positions_for_targets.clone().detach()
            targets = targets.clone().detach()
        
        angles = torch.nn.Parameter(angles.clone())
        optimiser = torch.optim.Adam([angles], lr=3e-3)
        contact_mask = _mask_from_ranges_nested(
            T, contact_ranges, device=device, margin=contact_margin
        )
        weights = dict(S=1e-1)
        foot_jidxs = torch.tensor([7, 8, 10, 11], device=device, dtype=torch.long)
        for it in range(50):
            optimiser.zero_grad(set_to_none=True)

            angles_n = quat_normalize(angles)

            _, pos = quat_fk_torch(angles_n, skeleton, dummy_trajectory, use_joints26=False)
            L_all = w_all * (pos - targets).pow(2).sum(dim=-1).mean()

            if foot_jidxs.numel() > 0 and contact_mask.any():
            # (T,1,1) 가중: 컨택트면 1, 아니면 0
                cm = contact_mask.view(-1, 1, 1).float()
                # 발 XY 차이 (T, F, 2)
                xy_diff = (pos[:, foot_jidxs, :2] - targets[:, foot_jidxs, :2])
                # 컨택트일 때만 벌점: cm * (diff^2)
                L_xy_foot = w_xy_foot * (cm * xy_diff.pow(2)).mean()
            else:
                L_xy_foot = pos.sum() * 0.0  # zero scalar on correct device/dtype

            # (C) 발 Z: 컨택트 프레임은 원본 z 유지(스무딩 전 pos_orig의 z를 타깃)
            if foot_jidxs.numel() > 0 and contact_mask.any():
                z_target = positions_for_targets[:, foot_jidxs, 2]  # (T,F) - 이미 no_grad로 계산됨
                z_diff = (pos[:, foot_jidxs, 2] - z_target)     # (T,F)
                cmz = contact_mask.view(-1, 1).float()          # (T,1)
                L_z_foot = w_z_foot * (cmz * (z_diff ** 2)).mean()
            else:
                L_z_foot = pos.sum() * 0.0

            loss = L_all + L_xy_foot + L_z_foot
            loss.backward()

            torch.nn.utils.clip_grad_norm_([angles], max_norm=0.25)  
            optimiser.step()
            
            del loss, L_all, L_xy_foot, L_z_foot, pos, angles_n
            if 'xy_diff' in locals():
                del xy_diff
            if 'z_diff' in locals():
                del z_diff

        return quat_normalize(angles.data)

    ## Losses	
    def trajectory_loss(self, trajectory, velocity_init):
        velocity = get_velocity(trajectory)
        return (velocity - velocity_init).square().sum(dim=-1).mean()
    def contacts_loss(self, positions, ranges, targets, weights):
        loss = 0.0
        for fb in (0, 1):  # heel/toe
            for lr in (0, 1):  # left/right
                jidx = Feet.JIDXS[fb][lr]
                for r, target in zip(ranges[fb][lr], targets[fb][lr]):
                    dists2 = (positions[r[0]:r[1], jidx, :] - target).square().sum(dim=-1)
                    loss += (weights[r[1]-r[0]] * dists2).sum()    
        return loss
    # 접촉 관절들의 z 좌표가 0에 가까워지도록 손실 계산
    def z_hovering_loss(self, positions, left_foot_contacts, right_foot_contacts):
        device = positions.device
        T = positions.shape[0]
        # 접촉 관절 인덱스: [22, 23, 24, 25]
        contact_joint_indices = torch.tensor([22, 23, 24, 25], device=device, dtype=torch.long)
        # 접촉 레이블을 하나의 텐서로 결합: [T, 4] (right_ankle, right_foot, left_ankle, left_foot)
        contact_labels = torch.cat([
            right_foot_contacts[:, 0:1],  # right_ankle (heel) 22
            right_foot_contacts[:, 1:2],  # right_foot (toe) 23
            left_foot_contacts[:, 0:1],   # left_ankle (heel) 24
            left_foot_contacts[:, 1:2]    # left_foot (toe) 25
        ], dim=1)  # [T, 4]
        # 접촉 관절들의 z 좌표 추출: [T, 4]
        contact_z_coords = positions[:, contact_joint_indices, 2]  # [T, 4]
        
        # 접촉이 감지된 프레임에서만 z 좌표가 0에 가까워지도록 손실 계산
        # contact_labels가 1인 경우에만 손실 적용
        contact_mask = contact_labels.bool()  # [T, 4]
        z_distances_sq = (100.0 * contact_z_coords).pow(2)
        # 접촉이 감지된 경우에만 손실 적용
        masked_z_distances_sq = contact_mask.float() * z_distances_sq  # [T, 4]
        # 전체 손실 계산 (접촉이 감지된 모든 관절과 프레임에 대해 평균)
        total_contact_frames = contact_mask.sum()
        if total_contact_frames > 0:
            z_loss = masked_z_distances_sq.sum() / total_contact_frames
        else:
            z_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return z_loss	
    ## Optimisation
    def __call__(self, angles, skeleton, trajectory, trans2joint, left_foot_contacts, right_foot_contacts, feet_heights=None):
        # SMPL 입력 형태: angles (120, 22, 4), skeleton (24, 3), trajectory (120, 3)
        angles = angles.to(self.device)
        skeleton = skeleton.to(self.device)
        trajectory = trajectory.to(self.device)
        left_foot_contacts = left_foot_contacts.to(self.device)
        right_foot_contacts = right_foot_contacts.to(self.device)
        trans2joint_tensor = torch.from_numpy(trans2joint).to(self.device)
        # Precomputations
        skeleton = skeleton[:-2, :]  # 24, 3 -> 22, 3
        skeleton = skeleton.repeat(120, 1, 1) # 120, 22, 3
        devices = dict(angles=angles.device, skeleton=skeleton.device, trajectory=trajectory.device)
        _, positions = quat_fk_torch(angles, skeleton, trajectory, use_joints26=False)
        move_to_zero_trajectory = trajectory[0:1, :2].clone()  # 첫 번째 프레임의 XY만
        trajectory_normalized = trajectory.clone()
        trajectory_normalized[:, :2] -= move_to_zero_trajectory  # XY만 정규화, Z는 유지
        # 정규화된 trajectory로 velocity_init 계산
        velocity_init = get_velocity(trajectory_normalized)  # T-1, 3
        # Prepare data
        angles, skeleton = Feet.extend(angles, skeleton, trajectory)
        _, positions_init = quat_fk_torch(angles, skeleton, trajectory_normalized, use_joints26=True)
        contact_ranges = [[[], []],  # fb=0(foot): [LR=0(left)리스트, LR=1(right)리스트]
                          [[], []]]  # fb=1(ankle): [LR=0(left)리스트, LR=1(right)리스트]
        def ranges_from_binary(contact_1d_bool):  # 접촉 레이블을 구간으로 반환 
            contact_1d_bool = contact_1d_bool.bool()
            T = len(contact_1d_bool)
            ranges, in_run, start = [], False, 0
            for t in range(T):
                v = bool(contact_1d_bool[t].item())
                if v and not in_run:
                    in_run, start = True, t
                elif not v and in_run:
                    in_run = False
                    ranges.append((start, t))
            if in_run:
                ranges.append((start, T))
            return ranges
        # JIDXS 구조에 맞게 수정: fb=0(foot_contacts), fb=1(ankle_contacts)
        contact_ranges[0][0] = ranges_from_binary(left_foot_contacts[:, 1])   # left_foot_contacts (toe)
        contact_ranges[0][1] = ranges_from_binary(right_foot_contacts[:, 1])  # right_foot_contacts (toe)
        contact_ranges[1][0] = ranges_from_binary(left_foot_contacts[:, 0])   # left_ankle_contacts (heel)  
        contact_ranges[1][1] = ranges_from_binary(right_foot_contacts[:, 0])  # right_ankle_contacts (heel)
        # 접촉 앵커와 구간 가중치 계산
        contact_weights = {}   # 길이별 weight 캐시
        contact_locations = [[[] , []],  # fb=0: [L anchors[], R anchors[]]
                            [[] , []]]  # fb=1: [L anchors[], R anchors[]]
        weighter = ContactWeightFactory(
            device=self.device,
            margin=self._margin, 
            mode="gauss",      
            normalize='sum',     
            poly_p=2,
            gauss_sigma=None
        )
        for fb in (0, 1):         # heel/toe
            for lr in (0, 1):     # left/right
                jidx = Feet.JIDXS[fb][lr]      # 확장된 J'에서의 접촉 관절 인덱스
                for (start, stop) in contact_ranges[fb][lr]:
                    t_mid = (start + stop) // 2
                    xy = positions_init[t_mid, jidx, :2]               # (2,)
                    z  = torch.zeros_like(xy[..., :1])             
                    anchor = torch.cat([xy, z], dim=-1)                # (3,) 그 프레임에서의 xy z좌표 z=0 
                    contact_locations[fb][lr].append(anchor)  # 앵커의 xyz좌표저장 
                    length = stop - start
                    if length not in contact_weights:   #길이별 가중치 계산 -> 앵커와 가까울수록 크게 제약 
                        contact_weights[length] = weighter.get_weights(length)

        angles_orig = angles.detach().clone()   # 원본 포즈 보존용 앵커
        angles_orig = quat_normalize(angles_orig)
        angles = torch.nn.Parameter(angles)							
        trajectory_param = torch.nn.Parameter(trajectory_normalized)  
        optimiser = torch.optim.Adam(
            [
                {"params": [angles],           "lr": 1e-3, "betas": (0.5, 0.9995)},  
                {"params": [trajectory_param], "lr": 1e-3, "betas": (0.8, 0.9995)},  
            ],
            amsgrad=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=150, eta_min=1e-4) 
        for iter in range(self._niters):
            # Compute loss
            losses = {}
            raw_losses = {}
            
            if self.tloss: ## 원본 이미지와 전체 이동자체는 동일하게 이동하도록 하는 loss 
                current_trajectory_full = trajectory_param.clone()
                current_trajectory_full[:, :2] += move_to_zero_trajectory  
                t_loss = self.trajectory_loss(current_trajectory_full, velocity_init)
                losses["T"] = t_loss
                raw_losses["T"] = t_loss.item()
            if self.closs: ## Contact position loss
                angles_normalized = quat_normalize(angles)
                _, positions = quat_fk_torch(angles_normalized, skeleton, trajectory_param, use_joints26=True)
                c_loss = self.contacts_loss(positions, contact_ranges, contact_locations, contact_weights)
                losses["C"] = c_loss
                raw_losses["C"] = c_loss.item()
            if self.zloss: ## Z hovering loss
                z_loss = self.z_hovering_loss(positions, left_foot_contacts, right_foot_contacts)
                losses["Z"] = z_loss
                raw_losses["Z"] = z_loss.item()        
            if self.Vxyloss: ## XY velocity loss
                foot_jidxs = torch.tensor([25, 23, 24, 22], device=positions.device)
                xy = positions[:, foot_jidxs, :2]
                vxy = xy[1:] - xy[:-1]                     # [T-1,4
                # contact_weights를 사용하여 가우시안 웨이트 적용
                cm = torch.zeros_like(vxy[..., :1])         
                # 각 접촉 구간에 대해 가우시안 웨이트 적용
                for fb in (0, 1):         # heel/toe
                    for lr in (0, 1):     # left/right
                        jidx_in_foot_jidxs = fb * 2 + lr  # foot_jidxs에서의 인덱스 (0,1,2,3)
                        for (start, stop) in contact_ranges[fb][lr]:
                            length = stop - start
                            if length in contact_weights:
                                weights = contact_weights[length]  # [length]
                                # 접촉 구간 내에서만 웨이트 적용 (start+1부터 stop까지, vxy는 T-1이므로)
                                contact_start = max(0, start)
                                contact_end = min(vxy.shape[0], stop)
                                if contact_start < contact_end:
                                    # 웨이트를 해당 구간에 적용
                                    weight_start = max(0, contact_start - start)
                                    weight_end = weight_start + (contact_end - contact_start)
                                    cm[contact_start:contact_end, jidx_in_foot_jidxs, 0] = weights[weight_start:weight_end]
                # 웨이트가 0인 부분은 0.0으로 설정 (접촉이 아닌 구간에서는 속도 손실 없음)
                cm = torch.where(cm > 0, cm, torch.zeros_like(cm))
                L_xy_vel = (vxy.pow(2) * cm).mean()
                losses["Vxy"] =L_xy_vel 
                raw_losses["Vxy"] = L_xy_vel.item()
            loss = sum(self._weights[key] * value for key, value in losses.items())

            # 텐서보드 로깅 (매 이터레이션)
            if self._writer is not None:
                self._writer.add_scalar('Loss/Total', loss.item(), iter)
                for key in raw_losses:
                    weighted_loss = self._weights[key] * raw_losses[key]
                    self._writer.add_scalar(f'Loss/{key}_Raw', raw_losses[key], iter)
                    self._writer.add_scalar(f'Loss/{key}_Weighted', weighted_loss, iter)
                    self._writer.add_scalar(f'Weight/{key}', self._weights[key], iter)

            # Optimise
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            k = 5                 
            alpha0 = 0.35            
            alpha_min = 0.05       
            N_apply = 40 
            n_apply = iter // k                      
            phase = min(n_apply, N_apply)
            alpha_scalar = alpha_min + 0.5*(alpha0 - alpha_min)*(1.0 + math.cos(math.pi * phase / N_apply))
            if (iter >0 and iter % k == 0):
                with torch.no_grad():
                    angles_n = quat_normalize(angles)
                    same_hemi = (angles_n * angles_orig).sum(dim=-1, keepdim=True) >= 0
                    anchor = torch.where(same_hemi, angles_orig, -angles_orig)
                    alpha = torch.full(angles_n.shape[:-1] + (1,), alpha_scalar, device=angles.device)
                    # 하체만 강하게, 상체는 약하게: per-joint 마스크
                    lower_idx = torch.tensor([1,2,4,5,7,8,10,11,22,23,24,25], device=angles.device)  # 예시
                    mask = torch.zeros_like(alpha); mask[:, lower_idx, 0] = 1.0
                    alpha = alpha * (0.5 + 0.5*mask)  # 상체 0.5*α, 하체 1.0*α 같은 식
                    #"원본(angles_orig) ↔ 현재(angles_n)" slerp로 자연스러움 보존
                    q_new = quat_slerp(anchor, angles_n, alpha)   # [T,J,4]
                    angles.copy_(quat_normalize(q_new))
            scheduler.step()
        # remove contacts joints and smooth animation
        angles_detached = quat_normalize(angles.data).detach()
        skeleton_detached = skeleton.detach()
        # 원본 스케일로 복원 (XY만 복원)
        trajectory_detached = trajectory_param.data.detach().clone()
        trajectory_detached[:, :2] += move_to_zero_trajectory  # XY만 복원
        
        angles, skeleton = Feet.reduce(angles_detached, skeleton_detached)
        angles = Cleaner.smooth(angles, skeleton, size=5, std=1, contact_ranges=contact_ranges, contact_margin=2, w_all=1e-1, w_xy_foot=2e-2, w_z_foot=5e-1)
        final_trajectory = trajectory_detached  # trajectory는 smooth하지 않고 그대로 사용
        # Unflatten outputs
        _, positions = quat_fk_torch(angles, skeleton, final_trajectory, use_joints26=False)
        positions_after_numpy = positions.detach().cpu().numpy()
        positions = np.expand_dims(positions_after_numpy, axis=0)
        return angles.to(devices["angles"]), final_trajectory.to(devices["trajectory"])

