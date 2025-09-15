import argparse
import os
import torch
from trainer_chois import *
from trainer_chois import Trainer

from pathlib import Path
import yaml
from manip.model.transformer_object_motion_cond_diffusion import ObjectCondGaussianDiffusion 
#store_true(있으면 그냥 true)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='chois_projects', help='project name')
    parser.add_argument('--entity', default='', help='W&B entity')
    parser.add_argument('--exp_name', default='chois', help='save to project/name')

    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--pretrained_model', type=str, default="", help='checkpoint')

    parser.add_argument('--data_root_folder', type=str, default="", help='data root folder')

    parser.add_argument('--save_res_folder', type=str, default="", help='save res folder')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results w planned path 
    parser.add_argument("--use_long_planned_path", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_on_train", action="store_true")

    # For quantitative evaluation. 
    parser.add_argument("--for_quant_eval", action="store_true")

    # Train and test on different objects. 
    parser.add_argument("--use_object_split", action="store_true")

    # Add language conditions. 
    parser.add_argument("--add_language_condition", action="store_true")

    # Input the first human pose, maybe can connect the windows better.  
    parser.add_argument("--input_first_human_pose", action="store_true")

    parser.add_argument("--use_guidance_in_denoising", action="store_true")

    parser.add_argument("--compute_metrics", action="store_true")

    # Add rest offsets for body shape information. 
    parser.add_argument("--use_random_frame_bps", action="store_true")

    parser.add_argument('--test_object_name', type=str, default="", help='object name for long sequence generation testing')
    parser.add_argument('--test_scene_name', type=str, default="", help='scene name for long sequence generation testing')

    parser.add_argument("--use_object_keypoints", action="store_true")

    parser.add_argument('--loss_w_feet', type=float, default=1, help='the loss weight for feet contact loss')
    parser.add_argument('--loss_w_fk', type=float, default=1, help='the loss weight for fk loss')
    parser.add_argument('--loss_w_obj_pts', type=float, default=1, help='the loss weight for fk loss')

    parser.add_argument("--add_semantic_contact_labels", action="store_true")

    parser.add_argument("--test_unseen_objects", action="store_true")

    # Save SMPL motion parameters as NPZ files
    parser.add_argument("--save_motion_params", action="store_true", help="Save SMPL motion parameters as NPZ files")
   
    opt = parser.parse_args()
    return opt
def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  모델부분 학습시작 제발 봐 여기야 
    repr_dim = 3 + 9 # Object relative translation (3) and relative rotation matrix (9)  오브젝트 차원 12

    repr_dim += 24 * 3 + 22 * 6 # Global human joint positions and rotation 6D representation 

    if opt.use_object_keypoints: #오브젝트 키포인트?? 
        repr_dim += 4 
# repr_dim 220
    loss_type = "l1"
# d_model->트랜스포머의 중간표현의 차원, n_head-> multi head self attention의 헤드 개수, d_k -> 트랜스포머 키 차원, d_v -> 트랜스포머 벨류차원
# max_timesteps-> 120프레임 + conditional 1개  loss는 L1  총차원이 220?
# 디퓨전 모델 클래스 정의 
    diffusion_model = ObjectCondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                input_first_human_pose=opt.input_first_human_pose, \
                use_object_keypoints=opt.use_object_keypoints) 
   #디바이스 올리고
    diffusion_model.to(device)
#트레이너 정의 
    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=400001,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
    )
    #디버깅 시작
    trainer.train()

    torch.cuda.empty_cache()
if __name__ == "__main__":
    opt = parse_opt()
    
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    print(">>> opt.data_root_folder =", opt.data_root_folder)
    run_train(opt, device)