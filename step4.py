
import os.path as osp
import argparse

import numpy as np
import torch
from DS import PoseBetaData

import smplx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import grad
from pyglet.gl import Config
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#these are just sample to initialize data. dont take them seriouasly and just skip it:)
t = torch.tensor([[5, 5, 5]])
b = [ 2.0092829e-01, -4.7343564e-01,  3.0960462e+00, -4.0517915e-02,
        2.7154531e-02, -6.4170644e-02, -1.1001365e-01,  7.1431272e-02,
       -1.2412101e-02,  1.1070399e-01, -7.5347456e-03,  1.7039023e-02,
        4.8988912e-01, -2.0403357e-01, -4.5633268e-02,  7.5145507e-01,
        1.2691019e-01,  2.0166133e-02, -7.1991138e-02, -1.0963257e-02,
       -1.4927958e-03, -5.1684940e-01,  2.7989370e-01, -9.7674258e-02,
       -4.0945524e-01, -2.0095254e-01, -4.2464063e-02,  1.2847630e-02,
       -3.9882913e-02,  2.5017265e-02,  1.0204447e-01, -9.1677077e-02,
        3.6221676e-02, -1.9838466e-01,  2.2355971e-01, -8.4209684e-03,
        2.1558946e-01,  8.0781616e-02, -8.0528513e-02, -2.2696927e-01,
       -3.3637226e-01,  3.9448011e-01, -2.3582873e-01,  3.5388765e-01,
       -4.7148848e-01, -2.0268786e-01, -2.0875392e-02,  1.2159009e-01,
       -3.7764913e-01, -3.1121764e-01,  4.6921107e-01, -5.3582758e-01,
        3.4337753e-01, -4.8270905e-01, -9.7129893e-01, -6.3348162e-01,
        1.9891281e-01, -1.0862539e+00,  7.1804100e-01,  8.2117822e-03,
       -7.6154143e-01, -4.2353433e-02, -2.4707901e-01, -6.7676985e-01,
       -9.1297671e-02,  1.1944697e-01,  1.2532522e-01,  6.0560659e-02,
       -4.5478052e-01,  1.8053646e-01, -2.2184888e-02,  3.5248435e-01]

shapeme0 = [-0.05973619,  0.3738634 ,  0.93987787,  2.8293848 ,  0.43564498,
        0.6554886 , -0.44000575,  0.3319269 ,  0.22023651, -0.09098779]
shapeme = torch.tensor([shapeme0])
body_pose = torch.tensor([b[3:]])
###end of random samples
print('body pos size is: ',body_pose.size())



def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=True,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='matplotlib',
         use_face_contour=False):

    #create our model.
    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext,transl=None,body_pose=body_pose,
                         global_orient=None,joint_mapper= None,create_body_pose=True,
                         betas = shapeme)
    print(model)
    model.train()
    # load the dataset as a dataloader
    ds = PoseBetaData(args.batch_pickle)
    it = iter(ds)
    # for i in range(500):
    #     a,b = next(it)
    # for i in range(4):
    # print(f'shape size {a.size()}',f'pose size {b.size()}')

    train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=False)

    optimizer = optim.SGD(model.parameters(), lr=1,momentum=0.9)
    gb = torch.ones([4,3],dtype=torch.float32)/2

    #########
    #In this part we want to optimize our smpl dataset based on the 4 different view 
    #Out put of the batch given o dataset is the same as size of batch and same as each other.
    # We commented this part because of our limited process power.
    #########
    # for epoch in range(args.epochs):
    #     for batch_idx, (pose,shape) in enumerate(ds):
    #             pose,shape = pose.to(device) , shape.to(device)
    #             pose = pose.requires_grad_(True)
    #             shape = shape.requires_grad_(True)
    #             optimizer.zero_grad()
                

    #             output = model(body_pose=pose,betas=shape,return_verts=True,global_orient= gb)

    #             vertices = output.vertices
    #             joints = output.joints
    #             # faces = model.faces

    #             vert_mean = torch.mean(vertices,dim=0)
    #             joint_mean = torch.mean(joints,dim=0)

    #             loss_vert,loss_joint = torch.zeros([4,1]),torch.zeros([4,1])
    #             for i in range(4):
    #                 temp1 = vert_mean - vertices[i]
    #                 temp2 = torch.pow(temp1,2)
    #                 temp3 = torch.sqrt(torch.sum(temp2))
    #                 loss_vert[i]=torch.mean(temp3)
    #             loss = torch.mean(loss_vert)
    #             # print(loss)

    #             # print(f'out vertices size: {vertices.size()}, out oint size: {joints.size()}')

    #             # loss = sord_loss(vertices,joints)
    #             loss.backward()
    #             optimizer.step()
    #             if batch_idx % 50 == 0:
    #                 print(f'loss is :{loss} on batch: {batch_idx}')
    #             if batch_idx == 100:
    #                 break

                
    #             # if batch_idx == 1000:
    #             #     break
    #     break

    
    



    #This is just a sample.
    gb = torch.ones([4,3],dtype=torch.float32)/2
    it = iter(ds)


    import open3d as o3d
    geometry = []
    V = []
    for i in range(len(ds)):
        
        a,b = next(it)
        output = model(body_pose=a,betas=b,return_verts=True,global_orient= gb)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        
        if i < 6:
            V.append(np.mean(vertices,0))
            
        else:
            v2 = np.mean(vertices,0)
            V.append(np.mean([V[-1], V[-2], V[-3], V[-4], v2], 0))
            
        
        
        joints = output.joints.detach().cpu().numpy().squeeze()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            V[-1])
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry.append(mesh)
        if i%50 == 0:
            print(f'{i} sample out of {len(ds)} sample rendered.')
#             print(v2.shape)
            
        
        
        
    # for i in range(4):

    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(
    #         vertices[i])
    #     mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    #     mesh.compute_vertex_normals()
    #     mesh.paint_uniform_color([0.3, 0.3, 0.3])

    #     geometry.append(mesh)
# if plot_joints:
#     joints_pcl = o3d.geometry.PointCloud()
#     joints_pcl.points = o3d.utility.Vector3dVector(joints)
#     joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
#     geometry.append(joints_pcl)
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(
    #     np.mean(vertices,0))
    # mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    # mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([0.3, 0.3, 0.3])

    # geometry.append(mesh)

    #rendering and save files to pics folder. 

    vis= o3d.visualization.Visualizer()
    vis.create_window(window_name = 'test', width = 960, height=540,  visible=True)
    vis.add_geometry(geometry[0])
    print(geometry[0])


    #View Controll
    ctr = vis.get_view_control()
    # ctr.set_front([1,0,0])
    # ctr.set_up([0,0,1])
    # ctr.set_zoom(0.9)
    # Updates

    for i in range(len(geometry) -1):
        vis.add_geometry(geometry[i+1])
        vis.remove_geometry(geometry[i])
        vis.update_geometry(geometry[i+1])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"pics/img_{i+1}.png")
        # time.sleep(5)


        # o3d.visualization.draw_geometries(geometry)
    

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str, default='models/',
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smpl', type=str, 
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')
    parser.add_argument('--batch_size', default=10,
                        type=int,
                        help='batch size for smpl training.')
    parser.add_argument('--num_workers', default=2,
                        type=int,
                        help='dataloader workers.')
    parser.add_argument('--epochs', default=5,
                        type=int,
                        help='num epochs.')
    parser.add_argument('--batch_pickle', default='ashpickle',
                        type=str,
                        help='path to batched pickle of step 3.')
    

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression
    batch_size = args.batch_size

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour)
