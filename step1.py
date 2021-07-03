import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)


# image_folder_path = 'images_beta/'
# output_path = 'out_test'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_information(image_folder_cam):
    mot = MPT(
            device=device,
            batch_size=10,
            detector_type='yolo',
            output_format='dict',
            yolo_img_size=416,
        )
    
    tracking_results = mot(image_folder_cam)
    
    del mot
    
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < 32:
            del tracking_results[person_id]
            
    
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    
    
    img = cv2.imread(image_folder_cam+'/000001.png').shape
    image_folder = image_folder_cam
    orig_height, orig_width = img[:2]
    
    vibe_results = {}


    for person_id in tqdm(list(tracking_results.keys())):
            bboxes = joints2d = None

            bboxes = tracking_results[person_id]['bbox']

            frames = tracking_results[person_id]['frames']

            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=1,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

            with torch.no_grad():

                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    batch = batch.unsqueeze(0)
                    batch = batch.to(device)

                    batch_size, seqlen = batch.shape[:2]
                    output = model(batch)[-1]

                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                    smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))


                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)
                smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
                del batch


                pred_cam = pred_cam.cpu().numpy()
                pred_verts = pred_verts.cpu().numpy()
                pred_pose = pred_pose.cpu().numpy()
                pred_betas = pred_betas.cpu().numpy()
                pred_joints3d = pred_joints3d.cpu().numpy()
                smpl_joints2d = smpl_joints2d.cpu().numpy()


                orig_cam = convert_crop_cam_to_orig_img(
                    cam=pred_cam,
                    bbox=bboxes,
                    img_width=orig_width,
                    img_height=orig_height
                )

                joints2d_img_coord = convert_crop_coords_to_orig_img(
                    bbox=bboxes,
                    keypoints=smpl_joints2d,
                    crop_size=224,
                )

                output_dict = {
                    'pred_cam': pred_cam,
                    'orig_cam': orig_cam,
                    'verts': pred_verts,
                    'pose': pred_pose,
                    'betas': pred_betas,
                    'joints3d': pred_joints3d,
                    'joints2d': joints2d,
                    'joints2d_img_coord': joints2d_img_coord,
                    'bboxes': bboxes,
                    'frame_ids': frames,
                }

                vibe_results[person_id] = output_dict
            
    return vibe_results



def main(args):
    
    
    cam_folders = os.listdir(args.image_folder_path)

    



    for cam in cam_folders:
        vib_result = extract_information(args.image_folder_path+'/'+cam)
        joblib.dump(vib_result, os.path.join(args.output_path, cam+".pkl"))
        print(f"__________________________\n {cam} Finished\n_____________________")
        
        
if __name__ == '__main__':
    
    image_folder_path = 'images_beta/'
    output_path = 'out_test'
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_folder_path', type=str,default='images_beta/',
                        help='input images path')
    
    parser.add_argument('--output_path', type=str,default='out_test/',
                        help='output path')

        
    
    
    args = parser.parse_args()

    main(args)
        

