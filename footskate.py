"""
    Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
    code is made available under the license found in the LICENSE.txt at the root
    directory of the repository.
"""
import pytorch3d.transforms as transforms
# PyTorch
import anim, util
import torch
import numpy as np
import math
from typing import Tuple, Dict
from scipy.ndimage import uniform_filter1d
from torch.nn import functional as F


def calculate_skating_ratio(motions):
    thresh_height = 0.05 # 10
    fps = 30.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 8 # frames
    # input is [bs, T, 22, 3]
    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    motions = torch.from_numpy(motions).permute(0, 2, 3, 1)
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 1], 1:] - verts_feet[:, :, [0, 1], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 2, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames 
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
    
    return skating_ratio, skate_vel
    
    # verts_feet_gt = markers_got[:, [16, 47], :].detach().cpu().numpy() # [119, 2, 3] heels
    # verts_feet_horizon_vel_gt = np.linalg.norm(verts_feet_gt[1:, :, :-1] - verts_feet_gt[:-1, :, :-1],  axis=-1) * 30
    
    # verts_feet_height_gt = verts_feet_gt[:, :, -1][0:-1] # [118,2]
    # min_z = markers_gt[:, :, 2].min().detach().cpu().numpy()
    # verts_feet_height_gt  = verts_feet_height_gt - min_z

    # skating_gt = (verts_feet_horizon_vel_gt > thresh_vel) * (verts_feet_height_gt < thresh_height)
    # skating_gt = np.sum(np.logival_and(skating_gt[:, 0], skating_gt[:, 1])) / 118
    # skating_gt_list.append(skating_gt)
#쿼터니언 정규화 파이토치 3d 에없어서 구현함 
def quat_normalize(q, eps=1e-8):
        norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=eps)
        qn = q / norm
        # w<0이면 부호 뒤집어 주는 것도 관례상 깔끔 (동일 회전)
        # (필수는 아님. 학습/일관성 위해 권장)
        flip = (qn[..., :1] < 0).to(qn.dtype)
        qn = torch.where(flip.bool().expand_as(qn), -qn, qn)
        return qn



#  가중치 구현하는 클래스 
class ContactWeightFactory:
    """
    접촉 구간(길이 L)에 대한 시간 가중치 w ∈ [0,1]^L 생성기.
    모드:
      - 'cosine' : raised-cosine taper(+plateau)
      - 'tukey'  : tapered-cosine, 테이퍼 비율 α 자동 산정
      - 'poly'   : 다항 ramp(경계부를 더 타이트하게/느슨하게)
      - 'gauss'  : 시간축 Gaussian(중앙부 peak, 꼬리 짧게)
    """
    def __init__(self,
                 device="cpu",
                 margin:int=3,
                 mode:str="cosine",
                 normalize:str=None,   # None | 'max' | 'sum'
                 poly_p:int=2,         # poly 모드의 지수 p
                 gauss_sigma:float=None, # gauss 모드의 σ (None이면 length 기반 자동)
                 eps:float=1e-6):
        self.device = device
        self.margin = int(margin)
        self.mode = mode
        self.normalize = normalize
        self.poly_p = int(poly_p)
        self.gauss_sigma = gauss_sigma
        self.eps = eps

        # (mode, L, margin, poly_p, gauss_sigma, normalize) → weight 텐서 캐시
        self._cache: Dict[Tuple, torch.Tensor] = {}

    # ---- public ----
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

    # ---- core builders ----
    def _make_weights(self, L:int) -> torch.Tensor:
        if L <= 0:
            return torch.empty(0, device=self.device)
        if L == 1:
            return torch.ones(1, device=self.device)

        if self.mode == 'cosine':
            return self._cosine_taper(L, self.margin)
        elif self.mode == 'tukey':
            return self._tukey_taper(L, self.margin)
        elif self.mode == 'poly':
            return self._poly_taper(L, self.margin, self.poly_p)
        elif self.mode == 'gauss':
            return self._gauss_center(L, self.gauss_sigma)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ---- modes ----
    def _cosine_taper(self, L:int, m:int) -> torch.Tensor:
        """
        중앙 plateau=1.0, 양끝 m프레임은 raised-cosine ramp.
        2*m >= L이면 Hann으로 fallback(plateau 없음).
        """
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

    def _tukey_taper(self, L:int, m:int) -> torch.Tensor:
        """
        Tukey(=tapered cosine). alpha=2m/(L-1)로 자동 설정.
        alpha=1이면 Hann, 0이면 Box(=all ones).
        """
        if L == 1: return torch.ones(1, device=self.device)
        alpha = 0.0 if L <= 1 else min(1.0, max(0.0, 2.0 * max(0, m) / max(1, (L - 1))))
        n = torch.arange(L, device=self.device, dtype=torch.float32)
        w = torch.ones(L, device=self.device)

        if alpha <= 0.0:
            return w
        if alpha >= 1.0:
            return torch.hann_window(L, periodic=False, device=self.device).clamp_min(self.eps)

        # 앞쪽 테이퍼
        t1 = alpha*(L-1)/2
        idx1 = n < t1
        w[idx1] = 0.5 * (1 + torch.cos(math.pi * (2*n[idx1]/(alpha*(L-1)) - 1)))

        # 뒤쪽 테이퍼
        t2 = (L-1) * (1 - alpha/2)
        idx2 = n >= t2
        w[idx2] = 0.5 * (1 + torch.cos(math.pi * (2*n[idx2]/(alpha*(L-1)) - 2/alpha + 1)))

        return w.clamp_min(self.eps)

    def _poly_taper(self, L:int, m:int, p:int=2) -> torch.Tensor:
        """
        다항 ramp. 2m >= L이면 plateau 없이 중앙 peak의 bump로 전환.
        p↑ → 경계부에서 더 급격히 0로.
        """
        m = max(0, min(int(m), L//2))
        if 2*m >= L:
            # 중앙이 1, 양끝 0으로 내려가는 다항 bump
            x = torch.linspace(-1, 1, L, device=self.device).abs()
            w = (1 - x.pow(p)).clamp_min(0)
            return w.clamp_min(self.eps)

        w = torch.ones(L, device=self.device)
        x = torch.linspace(0, 1, m, device=self.device)
        ramp = x.pow(p)  # 0→1
        w[:m] = ramp
        w[L-m:] = torch.flip(ramp, dims=[0])
        return w.clamp_min(self.eps)

    def _gauss_center(self, L:int, sigma:float=None) -> torch.Tensor:
        """
        시간축 Gaussian. 중앙 frame에 peak=1.0.
        sigma가 None이면 L에 비례해 자동 설정(짧은 꼬리).
        """
        n = torch.arange(L, device=self.device, dtype=torch.float32)
        c = 0.5*(L-1)  # 중앙
        if sigma is None:
            # L이 길어질수록 plateau 느낌을 조금 주되 꼬리는 짧게
            sigma = max(1.0, 0.18 * L)  # 경험적 기본값
        w = torch.exp(-0.5 * ((n - c)/sigma)**2)
        return w.clamp_min(self.eps)




def _mask_from_ranges_nested(T, contact_ranges, device, margin=0):
    """
    contact_ranges: [[left_foot_ranges, right_foot_ranges],
                     [left_ankle_ranges, right_ankle_ranges]]
    각 원소는 [(s,e), ...], e는 exclusive
    """
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    for fb in range(len(contact_ranges)):       # 보통 2 (foot / ankle)
        for lr in range(len(contact_ranges[fb])):  # 보통 2 (L / R)
            for (s, e) in contact_ranges[fb][lr]:
                s = max(0, s - margin)
                e = min(T, e + margin)
                if s < e:
                    mask[s:e] = True
    return mask




def identity_quaternion(shape, device=None, dtype=None):
    quat = torch.zeros(*shape, 4, device=device, dtype=dtype)
    # 단위 쿼터니언은 새로 생성하는 텐서이므로 in-place 연산 문제 없음
    quat[..., 3] = 1.0
    return quat

def gaussian_moving_average(x, size, std, dim=-3, zero_phase=True):
    """Gaussian Moving Average (util.gma 대체)"""
    if size <= 1:
        return x
        # 1) sanity
    if size % 2 == 0:
        raise ValueError(f"kernel size must be odd, got {size}")
    sigma = float(max(std, 1e-6))
    # 가우시안 커널 생성
    k = torch.arange(size, dtype=x.dtype, device=x.device) - (size - 1) / 2
    kernel = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    kernel = kernel / (kernel.sum() + torch.finfo(x.dtype).eps)  # 안전 정규화
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
    x_pad = F.pad(x_flat, (pad, pad), mode='reflect')  # 경계 왜곡 적음
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
# SMPL 적용을 위한 추가 import 필요
# import pytorch3d.transforms as transforms  # 축각도 <-> 회전행렬 변환용
# from human_body_prior 등에서 SMPL 모델 import

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
# smpl 모델에서 관절 정보를 가져와야 할듯? 조인트 이름도 가져와야하나? 
EXTENDED_TOPOLOGY = anim.Topology(
    list(zip(TOPOLOGY.joints(), TOPOLOGY.parents())) + [
    ("right_ankle_contacts",	["right_ankle"]),  #22
    ("right_foot_contacts",		["right_foot"]),  #23
    ("left_ankle_contacts",		["left_ankle"]),   #24
    ("left_foot_contacts",		["left_foot"]),  #25
])
# 속도 계산 함수 -> root_trans = trajectory 값 토대로 계산하면될듯? 루트 관절의 이동속도  출력은 (T-1,3)
def get_velocity(trajectory):
    return trajectory.diff(dim=0) * FRAMERATE

# 아예 쿼터니언으로 받을까? ???  쿼터니언, 오프셋, root_trans 
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

    # 3) 컨테이너 - 리스트로 구성하여 in-place 연산 방지
    grot_list = []
    gpos_list = []

    # 4) 루트 처리 
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

    # 5) 체인 전개 (완전히 새로운 tensor 생성 방식)
    for j in range(1, J):
        p = parents[j]
        # 회전: Rg_j = Rg_p ∘ r_j (새로운 tensor 생성)
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
    else:
        parents = ori_kintree_table[:22]  # 첫 22개 요소 
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
            q_loc = transforms.quaternion_multiply(transforms.quaternion_invert(Qp), q_target_global)  # [T,4]
            q_loc = quat_normalize(q_loc)
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
    def __init__(self, iterations: int, tweight=1e2, cweight=1e-5, zweight=1e-3, margin=0, device="cpu", tensorboard_writer=None):
        self._niters = int(iterations)
        self._weights = dict(T=float(tweight), C=float(cweight), Z=float(zweight))
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
            # 완전히 새로운 텐서로 복사하여 그래프 연결 차단
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

            # 쿼터니언 정규화본으로 FK (수치 안정)
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

            torch.nn.utils.clip_grad_norm_([angles], max_norm=0.25)  # 추천
            optimiser.step()
            
            # 명시적으로 중간 변수들 정리
            del loss, L_all, L_xy_foot, L_z_foot, pos, angles_n
            if 'xy_diff' in locals():
                del xy_diff
            if 'z_diff' in locals():
                del z_diff

        return quat_normalize(angles.data)

    @classmethod
    def sigmoid_like(cls, x, degree=2):
        m = (x > 0.5).float()
        s = 1 - 2 * m
        return m + 0.5 * s * (2 * (m + s * x))**degree

    def weights(self, t, m):
        w = self.sigmoid_like(torch.arange(m, device=self.device) / (m-1), degree=2)
        if t >= 2 * m:
            return torch.cat([w, torch.ones(t - 2 * m, device=self.device), (1 - w)])
        else:
            return torch.cat([w[:t//2+t%1], (1 - w)[-t//2:]])

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
                    loss += (weights[r[1]-r[0]] * dists2).mean()
        return loss
    
    def z_hovering_loss(self, positions, left_foot_contacts, right_foot_contacts):
        """
        접촉 레이블이 접촉으로 감지했을 때 해당 관절(22,23,24,25)의 z 좌표가 0에 가까워지도록 하는 손실
        
        Args:
            positions: [T, J, 3] 관절 위치 텐서 (J=26, 확장된 관절 포함)
            left_foot_contacts: [T, 2] 왼발 접촉 레이블 (heel, toe)
            right_foot_contacts: [T, 2] 오른발 접촉 레이블 (heel, toe)
        
        Returns:
            z_loss: hovering을 줄이는 z 좌표 손실
        """
        device = positions.device
        T = positions.shape[0]
        
        # 접촉 관절 인덱스: [22, 23, 24, 25]
        contact_joint_indices = torch.tensor([22, 23, 24, 25], device=device, dtype=torch.long)
        
        # 접촉 레이블을 하나의 텐서로 결합: [T, 4] (right_ankle, right_foot, left_ankle, left_foot)
        contact_labels = torch.cat([
            right_foot_contacts[:, 0:1],  # right_ankle (heel)
            right_foot_contacts[:, 1:2],  # right_foot (toe)  
            left_foot_contacts[:, 0:1],   # left_ankle (heel)
            left_foot_contacts[:, 1:2]    # left_foot (toe)
        ], dim=1)  # [T, 4]
        
        # 접촉 관절들의 z 좌표 추출: [T, 4]
        contact_z_coords = positions[:, contact_joint_indices, 2]  # [T, 4]
        
        # 접촉이 감지된 프레임에서만 z 좌표가 0에 가까워지도록 손실 계산
        # contact_labels가 1인 경우에만 손실 적용
        contact_mask = contact_labels.bool()  # [T, 4]
        
        # z 좌표의 절대값 거리 계산 (0에서 얼마나 떨어져 있는지)
        z_distances = torch.abs(contact_z_coords)  # [T, 4]
        
        # 접촉이 감지된 경우에만 손실 적용
        masked_z_distances = contact_mask.float() * z_distances  # [T, 4]
        
        # 전체 손실 계산 (접촉이 감지된 모든 관절과 프레임에 대해 평균)
        total_contact_frames = contact_mask.sum()
        if total_contact_frames > 0:
            z_loss = masked_z_distances.sum() / total_contact_frames
        else:
            z_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return z_loss	

    ## Optimisation
    def __call__(self, angles, skeleton, trajectory, trans2joint, left_foot_contacts, right_foot_contacts, feet_heights=None):
        # SMPL 입력 형태: angles (120, 22, 4), skeleton (24, 3), trajectory (120, 3)
        # 모든 입력을 동일한 장치로 이동
        angles = angles.to(self.device)
        skeleton = skeleton.to(self.device)
        trajectory = trajectory.to(self.device)
        left_foot_contacts = left_foot_contacts.to(self.device)
        right_foot_contacts = right_foot_contacts.to(self.device)
        trans2joint_tensor = torch.from_numpy(trans2joint).to(self.device)
        # Precomputations
 
        skeleton = skeleton[:-2, :]  # 24, 3 -> 22, 3
        skeleton = skeleton.repeat(120, 1, 1) # 120, 22, 3
        # in-place 연산 대신 새로운 tensor 생성
        # skeleton_updated = skeleton.clone()
        # skeleton_updated[:, 0, :] = trajectory - trans2joint_tensor[None]
        #skeleton = skeleton_updated 
        devices = dict(angles=angles.device, skeleton=skeleton.device, trajectory=trajectory.device)
        shape = angles.shape[:-3]
        # velocity_init는 정규화된 trajectory로 계산해야 함 (정규화 후에 계산)
        # 여기서는 먼저 정규화를 해야 함
        
        # SMPL forward kinematics 사용 필요
        # positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
        _, positions = quat_fk_torch(angles, skeleton, trajectory, use_joints26=False)
        
        # 120프레임 동안 발 관절들의 절대 좌표 출력 (use_joints26=False 케이스)
        max_frames = min(120, positions.shape[0])
        print(f"120프레임 동안의 발 관절 절대 좌표 (use_joints26=False, 총 {max_frames}프레임):")
        for frame in range(10):
            print(f"프레임 {frame}:")
            print(f"  관절 7: {positions[frame, 7]}")
            print(f"  관절 8: {positions[frame, 8]}")
            print(f"  관절 10: {positions[frame, 10]}")
            print(f"  관절 11: {positions[frame, 11]}")
            if positions.shape[1] > 25:  # 관절이 충분히 많은 경우
                print(f"  관절 22: {positions[frame, 22]}")
                print(f"  관절 23: {positions[frame, 23]}")
                print(f"  관절 24: {positions[frame, 24]}")
                print(f"  관절 25: {positions[frame, 25]}")
        print("---")
    
        # Trajectory 정규화 (XY만 정규화, Z는 유지)
        move_to_zero_trajectory = trajectory[0:1, :2].clone()  # 첫 번째 프레임의 XY만
        trajectory_normalized = trajectory.clone()
        trajectory_normalized[:, :2] -= move_to_zero_trajectory  # XY만 정규화, Z는 유지
        
        # 정규화된 trajectory로 velocity_init 계산
        velocity_init = get_velocity(trajectory_normalized)  # T-1, 3
        
        # Prepare data
        angles, skeleton = Feet.extend(angles, skeleton, trajectory)
        
        
        _, positions_init = quat_fk_torch(angles, skeleton, trajectory_normalized, use_joints26=True)
        
        # 120프레임 동안 발 관절들의 절대 좌표 출력 (use_joints26=True 케이스)
        max_frames_init = min(120, positions_init.shape[0])
        print(f"120프레임 동안의 발 관절 절대 좌표 (use_joints26=True, 정규화된 trajectory, 총 {max_frames_init}프레임):")
        for frame in range(10):
            print(f"프레임 {frame}:")
            print(f"  관절 7: {positions_init[frame, 7]}")
            print(f"  관절 8: {positions_init[frame, 8]}")
            print(f"  관절 10: {positions_init[frame, 10]}")
            print(f"  관절 11: {positions_init[frame, 11]}")
            if positions_init.shape[1] > 25:  # 관절이 충분히 많은 경우
                print(f"  관절 22: {positions_init[frame, 22]}")
                print(f"  관절 23: {positions_init[frame, 23]}")
                print(f"  관절 24: {positions_init[frame, 24]}")
                print(f"  관절 25: {positions_init[frame, 25]}")
        print("---")
        # 외부 접촉 라벨을 이용해 contact_ranges 생성 (배치 없음)
        # left_foot_contacts:  (T, 2)  # [:, 0]=heel, [:, 1]=toe
        # right_foot_contacts: (T, 2)
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
        # left_foot_contacts[:, 0] = heel, left_foot_contacts[:, 1] = toe
        # JIDXS[0] = [left_foot, right_foot], JIDXS[1] = [left_ankle, right_ankle]
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
            margin=self._margin,   # 보통 2~3
            mode="gauss",         # "cosine" | "tukey" | "poly" | "gauss"
            normalize=sum,        # None | 'max' | 'sum'
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

        
        angles = torch.nn.Parameter(angles)							
        trajectory_param = torch.nn.Parameter(trajectory_normalized)  # 정규화된 trajectory 최적화
        optimiser = torch.optim.Adam([angles, trajectory_param], lr=1e-4) 

        for iter in range(self._niters):
            # Compute loss
            losses = {}
            raw_losses = {}
            
            if self.tloss: ## Trajectory velocity loss
                current_trajectory_full = trajectory_param.clone()
                current_trajectory_full[:, :2] += move_to_zero_trajectory  # XY만 복원
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
                if 'positions' not in locals():  # positions가 아직 계산되지 않은 경우
                    angles_normalized = quat_normalize(angles)
                    _, positions = quat_fk_torch(angles_normalized, skeleton, trajectory_param, use_joints26=True)
                z_loss = self.z_hovering_loss(positions, left_foot_contacts, right_foot_contacts)
                losses["Z"] = z_loss
                raw_losses["Z"] = z_loss.item()
            
            loss = sum(self._weights[key] * value for key, value in losses.items())
            
            # 텐서보드 로깅 (매 이터레이션)
            if self._writer is not None:
                self._writer.add_scalar('Loss/Total', loss.item(), iter)
                for key in raw_losses:
                    weighted_loss = self._weights[key] * raw_losses[key]
                    self._writer.add_scalar(f'Loss/{key}_Raw', raw_losses[key], iter)
                    self._writer.add_scalar(f'Loss/{key}_Weighted', weighted_loss, iter)
                    self._writer.add_scalar(f'Weight/{key}', self._weights[key], iter)
            
            # 각 웨이트별 로스 출력 (10 이터레이션마다 또는 첫/마지막 이터레이션)
            if iter % 10 == 0 or iter == 0 or iter == self._niters - 1:
                print(f"Iter {iter:3d}: Total Loss = {loss.item():.6f}")
                for key in raw_losses:
                    weighted_loss = self._weights[key] * raw_losses[key]
                    print(f"  {key} Loss: raw={raw_losses[key]:.6f}, weight={self._weights[key]:.2e}, weighted={weighted_loss:.6f}")
           
            # Optimise
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # remove contacts joints and smooth animation
        # SMPL 정규화 후 22관절로 복원
        # PyTorch3D로 교체: util.SU2.normalize → quat_normalize
        angles_detached = quat_normalize(angles.data).detach()
        skeleton_detached = skeleton.detach()
        # 원본 스케일로 복원 (XY만 복원)
        trajectory_detached = trajectory_param.data.detach().clone()
        trajectory_detached[:, :2] += move_to_zero_trajectory  # XY만 복원
        
        angles, skeleton = Feet.reduce(angles_detached, skeleton_detached)
        #angles = Cleaner.smooth(angles, skeleton, size=5, std=1, contact_ranges=contact_ranges, contact_margin=2, w_all=1e-1, w_xy_foot=2e-2, w_z_foot=5e-1)
        final_trajectory = trajectory_detached  # trajectory는 smooth하지 않고 그대로 사용
        # Unflatten outputs
        _, positions = quat_fk_torch(angles, skeleton, final_trajectory, use_joints26=False)
        max_frames = min(120, positions.shape[0])
        print(f"120프레임 동안의 발 관절 절대 좌표 (use_joints26=False, 총 {max_frames}프레임):")
        for frame in range(10):
            print(f"프레임 {frame}:")
            print(f"  관절 7: {positions[frame, 7]}")
            print(f"  관절 8: {positions[frame, 8]}")
            print(f"  관절 10: {positions[frame, 10]}")
            print(f"  관절 11: {positions[frame, 11]}")
        positions = positions.detach().cpu().numpy()
        positions = np.expand_dims(positions, axis=0)
        skating_ratio, skate_vel = calculate_skating_ratio(positions)

        # 최종 스케이팅 비율을 텐서보드에 기록
        if self._writer is not None:
            self._writer.add_scalar('Metrics/Final_Skating_Ratio', skating_ratio[0], 0)

        print(f"Skating ratio: {skating_ratio}")
        return angles.to(devices["angles"]), final_trajectory.to(devices["trajectory"])

