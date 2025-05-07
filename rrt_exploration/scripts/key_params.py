
scene_type = 'scene1'

params = {
    'lab': {
        ## global planner
        'min_inter_dist': 1.5,
        'max_inter_dist': 2.5,
        'inter_cluster_dist': 1.5,
        'frontier_cluster_dist': 1.2,
        'inter_point': True,
        'inter_buffer_size': 4,
        'erode_kernal_size':3, 

        ## local planner
        'speed': 0.5,
        'angular_speed': 0.1,
        'yaw_tolerance': 0.3,
        'xyRadius': 1.5,
        'validDist': 2.5,
        'path_interval': 2.5,
        'replan_freq': 6,
        'max_replan_num': 4,
        'viewpoint_tolerance':0.5,
    },

    'scene1': {
        ## global planner
        'min_inter_dist': 3,
        'max_inter_dist': 6,
        'inter_cluster_dist': 2,
        'frontier_cluster_dist': 1.5,
        'inter_point': False,
        'replan_freq': 10,
        'max_replan_num': 4,
        'erode_kernal_size':5, 

        ## local planner
        'speed': 0.8,
        'angular_speed': 0.1,
        'yaw_tolerance': 0.3,
        'xyRadius': 1.3,
        'validDist': 3,
        'path_interval': 2.5,
        'viewpoint_tolerance':0.5,
    },

    
    'scene2': {
        ## global planner
        'min_inter_dist': 3,
        'max_inter_dist': 6,
        'inter_cluster_dist': 2,
        'frontier_cluster_dist': 2,
        'inter_point': False,
        'replan_freq': 8,
        'max_replan_num': 4,
        'erode_kernal_size':6, 

        ## local planner
        'speed': 0.8,
        'angular_speed': 0.1,
        'yaw_tolerance': 0.3,
        'xyRadius': 1.5,
        'validDist': 3,
        'path_interval': 3,
        'viewpoint_tolerance':0.5,
    },
}

_params = params[scene_type]