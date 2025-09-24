import argparse
import os
import torch
from trainer_chois import run_sample
import time 
from datetime import timedelta, datetime 

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

if __name__ == "__main__":
    start = time.perf_counter()
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    run_sample(opt, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # GPU 작업 끝까지 기다리기
    
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f} 초")