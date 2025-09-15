# SMPL Motion Parameters NPZ 저장 기능

이 기능은 `trainer_chois.py`에서 motion generation 과정에서 생성되는 **SMPL 입력용 파라미터들**을 **npz 형태**로 저장합니다.

## 저장되는 데이터

SMPL 모델에 직접 입력할 수 있는 형태로 다음 파라미터들이 저장됩니다:

- **`betas`**: SMPL-X body shape parameters (numpy array)
- **`gender`**: 성별 정보 (string)
- **`local_rot_aa`**: Local rotation in axis-angle representation - Shape: T x 22 x 3 (numpy array)
- **`root_trans`**: Root translation - Shape: T x 3 (numpy array)

## 파일명 형식

기존 코드베이스의 패턴을 따라 다음 형식으로 저장됩니다:

### 일반 객체의 경우:
```
{seq_name}_sidx_{start_frame_idx}_eidx_{end_frame_idx}_sample_cnt_0_idx_{idx}.npz
```

### Unseen objects의 경우:
```
{seq_name}_{obj_name}_sidx_{start_frame_idx}_eidx_{end_frame_idx}_sample_cnt_0_idx_{idx}.npz
```

예시:
- `sub16_clothesstand_000_sidx_0_eidx_119_sample_cnt_0_idx_0.npz`
- `frl_apartment_4_sub16_trashcan_000_trashcan_sidx_0_eidx_119_sample_cnt_0_idx_1.npz`

## 사용 방법

### 1. 기능 활성화

커맨드 라인에서 다음과 같이 옵션을 설정합니다:

```bash
# 테스트 시
python test.py --save_motion_params

# 트레이닝 시  
python train.py --save_motion_params
```

### 2. 자동 저장

기능이 활성화되면 `sample_vis_res` 함수의 메인 루프에서 자동으로 저장됩니다.

저장 위치: `{vis_folder}/smpl_motion_params_npz/{step}/`

### 3. 데이터 로드 및 SMPL 사용

```python
import numpy as np
import torch

# NPZ 파일 로드
data = np.load('seq_name_sidx_0_eidx_119_sample_cnt_0_idx_0.npz')

# SMPL 파라미터 추출
betas = torch.from_numpy(data['betas']).float()  # Body shape
gender = str(data['gender'])  # Gender
local_rot_aa = torch.from_numpy(data['local_rot_aa']).float()  # T x 22 x 3
root_trans = torch.from_numpy(data['root_trans']).float()  # T x 3

# SMPL 모델에 바로 입력 가능
# mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
#     root_trans[None].cuda(), 
#     local_rot_aa[None].cuda(),
#     betas.cuda(), 
#     [gender], 
#     bm_dict, 
#     return_joints24=True
# )

print(f"Sequence: {data['seq_name']}")
print(f"Object: {data['obj_name']}")
print(f"Sequence length: {data['seq_len']}")
print(f"Local rotation shape: {local_rot_aa.shape}")
print(f"Root translation shape: {root_trans.shape}")
```

## 저장되는 NPZ 구조

```python
npz_data = {
    # 시퀀스 정보
    'seq_name': string,           # 시퀀스 이름
    'obj_name': string,           # 객체 이름
    'start_frame_idx': int,       # 시작 프레임 인덱스
    'end_frame_idx': int,         # 끝 프레임 인덱스
    'seq_len': int,               # 실제 시퀀스 길이
    'step': int,                  # 트레이닝 스텝
    'idx': int,                   # 배치 인덱스
    
    # SMPL 파라미터들 (바로 사용 가능)
    'betas': numpy_array,         # Body shape parameters
    'gender': string,             # Gender information
    'local_rot_aa': numpy_array,  # Shape: (T, 22, 3) - Local rotations
    'root_trans': numpy_array,    # Shape: (T, 3) - Root translations
    
    # 추가 정보 (optional)
    'obj_rot_mat': numpy_array,   # Object rotation matrices
    'obj_com_pos': numpy_array,   # Object center of mass positions
}
```

## 장점

1. **SMPL 호환성**: 바로 SMPL 모델에 입력 가능한 형태
2. **기존 패턴 준수**: 코드베이스의 기존 파일명 패턴과 일치
3. **효율적 저장**: npz 형태로 압축된 numpy 배열 저장
4. **개별 저장**: 각 시퀀스별로 개별 파일로 저장되어 관리 용이

## 주의사항

1. `self.compute_metrics`가 False일 때만 저장됩니다 (시각화 모드)
2. 메인 루프에서 실시간으로 저장되므로 성능에 약간의 영향이 있을 수 있습니다
3. 저장된 파일들은 상당한 용량을 차지할 수 있습니다

## 구현 세부사항

- 구현 위치: `trainer_chois.py`의 `save_individual_smpl_params_npz` 메서드
- 호출 위치: `sample_vis_res` 함수의 메인 루프 내부 (curr_local_rot_aa_rep, root_trans 계산 직후)
- 저장 형식: NumPy compressed archive (.npz) 파일
