#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from s1.utils.system_utils import searchForMaxIteration
from s1.scene.dataset_readers import readInfo 
from s1.scene.gaussian_model import GaussianModel
from s1.scene.gaussian_model_dpsr_dynamic_anchor import GaussianModelDPSRDynamicAnchor
from s1.scene.deform_model import DeformModel, DeformModelNormal, DeformModelNormalSep
from s1.scene.appearance_model import AppearanceModel
from s1.arguments import ModelParams
from s1.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        # self.test_cameras = {}


        scene_info = readInfo(args.source_path, 
                            args.white_background, 
                            args.eval, 
                            downsample=args.downsample,
                            views_index=args.views_index,
                            views_num=args.views_num,
                            time_index=args.time_index,
                            time_num=args.time_num,
                            time_indexs_that_use_novel_views=args.time_indexs_that_use_novel_views,
                            train_json=args.train_json,
                            test_json=args.test_json
                            )
        
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Note that this vale: ", self.cameras_extent)

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                                args)

        if self.gaussians is not None:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "point_cloud.ply"),
                                        og_number_points=len(scene_info.point_cloud.points))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
                
        self.scene_info = scene_info

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_dynamic(self, iteration, deform, time=0.0):
        point_cloud_path = os.path.join(self.model_path, "point_cloud_dynamic/iteration_{}".format(iteration))
        self.gaussians.save_ply_dynamic(os.path.join(point_cloud_path, "point_cloud.ply"), deform=deform, time=time)

    def getCameras(self, scale=1.0):
        return self.train_cameras[scale]

