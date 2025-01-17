#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
import time 
import os
import sys

import platform 

from config import Config

from slam import Slam, SlamState
from slam_plot_drawer import SlamPlotDrawer
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory, SensorType
from trajectory_writer import TrajectoryWriter


if platform.system()  == 'Linux':     
    from display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 
from utils_img import ImgWriter

from feature_tracker_configs import FeatureTrackerConfigs

from loop_detector_configs import LoopDetectorConfigs

from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
from utils_depth import img_from_depth, filter_shadow_points

from config_parameters import Parameters  

from rerun_interface import Rerun

import traceback

from typing import Literal
import typer

app = typer.Typer()

@app.command()
def run_test(feature_extractor: str , loop_detector: str, num_features: int = 2000, scale_factor: float = 1.2, dataset_idx: int = 1, sample_freq:int = 1):                                      
    config = Config()
    assert feature_extractor in ['SUPERPOINT', 'XFEAT','BRISK','ORB2','ORB2_FREAK'], f"Feature Extractor Not One Of: {' '.join(['SUPERPOINT', 'XFEAT','BRISK','ORB2','ORB2_FREAK'])}"
    assert loop_detector in ['DBOW3','ALEXNET','SAD','HDC_DELF','COSPLACE'], f"Feature Extractor Not One Of: {' '.join(['DBOW3','ALEXNET','SAD','HDC_DELF','COSPLACE'])}"

    config.dataset_settings['base_path'] = f'./data/videos/testdata_{dataset_idx}_{sample_freq}'
    dataset = dataset_factory(config)
    

    trajectory_writer = None
    if True:
        trajectory_writer = TrajectoryWriter(format_type='kitti', filename='output/kitti.txt')
        trajectory_writer.open_file()
    
    groundtruth = groundtruth_factory(config.dataset_settings)

    camera = PinholeCamera(config)
    



    # Select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, FAST_TFEAT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    # WARNING: At present, SLAM does not support LOFTR and other "pure" image matchers (further details in the commenting notes about LOFTR in feature_tracker_configs.py).
    feature_tracker_config = getattr(FeatureTrackerConfigs,feature_extractor)
    feature_tracker_config['num_features'] = num_features
    #feature_tracker_config['num_levels'] = 1
    feature_tracker_config['scale_factor'] = scale_factor
    Printer.green('feature_tracker_config: ',feature_tracker_config)    
    



    # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing. 
    # LoopDetectorConfigs: DBOW2, DBOW3, IBOW, OBINDEX2, VLAD, HDC_DELF, SAD, ALEXNET, NETVLAD, COSPLACE, EIGENPLACES  etc.
    # NOTE: under mac, the boost/text deserialization used by DBOW2 and DBOW3 may be very slow.
    loop_detection_config = getattr(LoopDetectorConfigs, loop_detector)
    Printer.green('loop_detection_config: ',loop_detection_config)
        
    # Select your depth estimator in the front-end (EXPERIMENTAL, WIP)
    depth_estimator = None
    if Parameters.kUseDepthEstimatorInFrontEnd:
        Parameters.kVolumetricIntegrationUseDepthEstimator = False  # Just use this depth estimator in the front-end
        # Select your depth estimator (see the file depth_estimator_factory.py)
        # DEPTH_ANYTHING_V2, DEPTH_PRO, DEPTH_RAFT_STEREO, DEPTH_SGBM, etc.
        depth_estimator_type = DepthEstimatorType.DEPTH_PRO
        max_depth = 20
        depth_estimator = depth_estimator_factory(depth_estimator_type=depth_estimator_type, max_depth=max_depth,
                                                  dataset_env_type=dataset.environmentType(), camera=camera) 
        Printer.green(f'Depth_estimator_type: {depth_estimator_type.name}, max_depth: {max_depth}')       
                
    # create SLAM object
    slam = Slam(camera, feature_tracker_config, 
                loop_detection_config, dataset.sensorType(), 
                environment_type=dataset.environmentType(), 
                config=config) 
    slam.set_viewer_scale(dataset.scale_viewer_3d)
     
    
    # load system state if requested         
    if config.system_state_load: 
        slam.load_system_state(config.system_state_folder_path)
        viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
        print(f'viewer_scale: {viewer_scale}')
        slam.set_tracking_state(SlamState.INIT_RELOCALIZE)

    
    if platform.system() == 'Linux':    
        display2d = None # Display2D(camera.width, camera.height)  # pygame interface 
    else: 
        display2d = None  # enable this if you want to use opencv window
    # if display2d is None:
    #     cv2.namedWindow('Camera', cv2.WINDOW_NORMAL) # to make it resizable if needed


    img_writer = ImgWriter(font_scale=0.7)

    do_step = False      # proceed step by step on GUI 
    do_reset = False     # reset on GUI 
    is_paused = False    # pause/resume on GUI 
    is_map_save = True  # save map on GUI
    
    key_cv = None
            
    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
    output_index = 0

    keep_repeating = True


    while keep_repeating:
        
        img, img_right, depth = None, None, None    
        
        if do_step:
            Printer.orange('do step: ', do_step)
            
        if do_reset: 
            Printer.yellow('do reset: ', do_reset)
            slam.reset()
               
        if not is_paused or do_step:
        
            if dataset.isOk():
                print('..................................')         
                img = dataset.getImageColor(img_id)
                depth = dataset.getDepth(img_id)
                img_right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
            
            if img is not None:
                timestamp = dataset.getTimestamp()          # get current timestamp 
                next_timestamp = dataset.getNextTimestamp() # get next timestamp 
                frame_duration = next_timestamp-timestamp if (timestamp is not None and next_timestamp is not None) else -1

                print(f'image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}') 
                
                time_start = None 
                if img is not None:
                    time_start = time.time()    
                    
                    if depth is None and depth_estimator is not None:
                        depth_prediction = depth_estimator.infer(img, img_right)
                        if Parameters.kDepthEstimatorRemoveShadowPointsInFrontEnd:
                            depth = filter_shadow_points(depth_prediction)
                        else: 
                            depth = depth_prediction
                        depth_img = img_from_depth(depth_prediction, img_min=0, img_max=50)
                        cv2.imshow("depth prediction", depth_img)
                                  
                    slam.track(img, img_right, depth, img_id, timestamp)  # main SLAM function 
                                    


                    img_draw = slam.map.draw_feature_trails(img)
                    
                   
                    cv2.imwrite(f'./test/camera{output_index}.jpg', img_draw)
                    output_index += 1
                    
                        
                if trajectory_writer is not None and slam.tracking.cur_R is not None and slam.tracking.cur_t is not None:
                    trajectory_writer.write_trajectory(slam.tracking.cur_R, slam.tracking.cur_t, timestamp)
                    
                if time_start is not None: 
                    duration = time.time()-time_start     
                img_id += 1

                if img_id >= dataset.num_frames:
                    keep_repeating = False
            else: 
                pass
        else:
            break      



            
                    
            # manage interface infos  
    print("DONE!")                    
    if True:
        config.system_state_folder_path = "./output"
        slam.save_system_state(config.system_state_folder_path)
        dataset.save_info(config.system_state_folder_path)
        groundtruth.save(config.system_state_folder_path)
        Printer.green('uncheck pause checkbox on GUI to continue...\n')        
        

if __name__ == "__main__":
    typer.run(run_test)