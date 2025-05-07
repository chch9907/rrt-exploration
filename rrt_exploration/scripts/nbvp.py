#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from numpy import array, mean
from numpy.linalg import norm
from math import pi
import rospy
import os
import random
# import dynamic_reconfigure.client
import actionlib
from tf.transformations import quaternion_from_euler
from tf import TransformListener
from datetime import datetime
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker
from rrt_exploration.msg import PointArray  # type: ignore
from copy import copy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from scipy.spatial import KDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tf
from tsp_solver import solve_tsp
from nav_utils import informationGain, informationGainFOV, _world_to_map, _world_to_map_ori, \
WAYPOINT_YAW, OBSTACLE, UNKNOWN, get_list_dist, get_elements_by_idxs, unknown_check
from key_params import _params


class Planner:
    '''Balance between exploration and exploitation'''
    def __init__(self, max_range, sim_thred, plan_mode, hysteresis_gain, hysteresis_radius, info_multiplier):
        print('init planner...')
        self.max_range = max_range
        self.hysteresis_gain = hysteresis_gain
        self.hysteresis_radius = hysteresis_radius
        self.info_multiplier = info_multiplier
        self.cost_gamma = 0.1
        self.beta = self.hysteresis_gain / sim_thred
        self.plan_mode = plan_mode
        
    def frontier_utility(self, frontiers, cur_pose, next_subgoal_pose):
        frontiers = array(frontiers)
        next_subgoal = array(next_subgoal_pose)
        vector_current = cur_pose - next_subgoal
        vector_frontier = cur_pose - frontiers
        ## cosine similarity
        dir_heuristic = (vector_frontier @ vector_current.T) \
                        / ((norm(vector_current) * norm(vector_frontier, axis=1)))
        rospy.loginfo(f"dir_heuristic:{dir_heuristic}")
        costs = norm(frontiers - cur_pose, axis=1)
        gain = np.ones(len(frontiers)).T #* self.hysteresis_gain
        gain[np.where(costs <= self.hysteresis_radius)] = self.hysteresis_gain
        dir_heuristic = dir_heuristic.T * gain * self.info_multiplier - self.cost_gamma * costs
        return dir_heuristic.tolist()
    
    
    def viewpoint_utility(self, cur_pose, viewpoints, viewpoints_score):
        costs = norm(viewpoints - cur_pose, axis=1)
        utility = self.beta * np.array(viewpoints_score).T - self.cost_gamma * costs
        return utility.tolist()
    
    
    def __call__(self, frontiers, viewpoints, cur_pose, next_subgoal_pose, ftr_infogains, viewpoints_score):
        '''both unvisited frontiers, unvisited viewpoints are in world coordinate.'''
        cur_pose = np.array(cur_pose)
        if not len(frontiers) and not len(viewpoints):
            goal_idx = -1
        elif self.plan_mode == 'random':
            goal_idx = random.choice([i for i in range(len(frontiers))])
        elif self.plan_mode == 'infor_gain':
            
            if len(viewpoints):
                rospy.loginfo('select viewpoints')
                viewpoints = np.array(viewpoints)[:, :2]
                viewpoints_utility = self.viewpoint_utility(cur_pose, viewpoints, viewpoints_score)
                goal_idx = len(frontiers) + np.argmax(viewpoints_utility)
            elif len(frontiers):
                norm_fov_gain = ftr_infogains
                goal_idx = np.argmax(norm_fov_gain)
                rospy.loginfo(f'max gain:{norm_fov_gain[goal_idx]}, min gain:{np.min(norm_fov_gain)}, goal_idx:{goal_idx}')
            else:
                goal_idx = -1
                
        elif self.plan_mode == 'utility':
            if len(frontiers):
                frontier_utility = self.frontier_utility(frontiers, cur_pose, next_subgoal_pose)
                goal_idx = np.argmax(frontier_utility)
            else:
                goal_idx = -1

        elif self.plan_mode == 'utility-viewpoint':
            viewpoints_utility = []
            if len(viewpoints):
                viewpoints = np.array(viewpoints)[:, :2]
                viewpoints_utility = self.viewpoint_utility(cur_pose, viewpoints, viewpoints_score)
                rospy.loginfo(f'viewpoint heuristics:{viewpoints_utility}')
                
            frontiers_utility = []
            if next_subgoal_pose is None:
                rospy.loginfo('no next_subgoal_pose, select frontier by inforgain')
                frontiers_utility = ftr_infogains
            elif len(frontiers):
                rospy.loginfo(f'next_subgoal_pose:{next_subgoal_pose}, select frontier by heuristic')
                frontiers_utility = self.frontier_utility(frontiers, cur_pose, next_subgoal_pose)
            # rospy.loginfo(f'ftr_infogains:{ftr_infogains}')
            overall_utility = frontiers_utility + viewpoints_utility
            rospy.loginfo(f'frontier heuristics:{frontiers_utility}, overall:{overall_utility}')
            # overall_softmax = np.exp(overall_utility) / np.sum(np.exp(overall_utility))
            # rospy.loginfo(f'overall softmax:{overall_softmax}')
            if len(ftr_infogains):
                rospy.loginfo(f'infor gain argmax:{np.argmax(ftr_infogains)}')
            goal_idx = np.argmax(overall_utility)
        else:
            raise ValueError(self.plan_mode)
        return goal_idx
        

class NBVP:
    def __init__(self):
        # print('init next best view planner ...')
        ## topics 
        rospy.init_node('next_best_view_planner', anonymous=False)
        frontiers_topic = rospy.get_param('~frontiers_topic','/filtered_points')
        frontier_pub_topic = rospy.get_param('~frontier_pub_topic','/cur_frontiers')
        map_topic = rospy.get_param('~map_topic','/projected_map')
        goal_topic = rospy.get_param('~goal_topic','/nbv_points')
        finished_topic = rospy.get_param('~finished_topic','/finished_goal') 
        intermediate_topic = rospy.get_param('~inter_points_topic','/inter_points')
        viewpoint_topic = rospy.get_param('~viewpoint_topic','/viewpoints')
        viewpoint_score_topic = rospy.get_param('~viewpoint_score_topic','/viewpoints_score')
        viewpoint_goal_topic = rospy.get_param('~viewpoint_goal_topic','/viewpoints_goal')
        subgoal_topic = rospy.get_param('~subgoal_topic','/next_subgoal')
        
        ## params and variables
        self.decimal = rospy.get_param('~decimal', 2)
        self.frontiers = []
        self.ftr_visited = []
        self.inter_points = []
        self.inter_centroids = []
        self.inter_visited = []
        self.viewpoints = []
        self.viewpoints_score = []
        self.viewpoints_visited = []
        self.cur_pose = []
        
        self.ftr_cluster_dist = _params['frontier_cluster_dist'] 
        
        # infor gain
        self.max_depth_dist = rospy.get_param('~max_depth_dist', 2)  # depth max dist
        self.info_radius = rospy.get_param('~info_radius', 1.0)
        self.hysteresis_radius = rospy.get_param('~hysteresis_radius', 10.0)
        self.hysteresis_gain = rospy.get_param('~hysteresis_gain', 3.0)
        self.info_multiplier = rospy.get_param('~info_multiplier', 3.0)
        self.hfov_angle = rospy.get_param('~hfov_angle', 70.5)
        self.stride = rospy.get_param('~stride', 15)
        
        self.delay_after_assignement = rospy.get_param('~delay_after_assignement', 0.5)
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.mapData = OccupancyGrid()
        self.map_array = np.array([])
        self.goal_frame = rospy.get_param('~goal_frame','map')
        rateHz = rospy.get_param('~rateHz', 10)
        self.rate = rospy.Rate(rateHz)
        self.tf_listener = tf.TransformListener()
        self.use_planner = rospy.get_param('~use_planner', True)
        if self.use_planner:
            max_range = rospy.get_param('~max_range', 10)
            sim_thred = rospy.get_param('~sim_thred', 0.6)  # balance between exploration and exploitation
            plan_mode = rospy.get_param('~plan_mode', 'utility-viewpoint')  # random, infor_gain, utility, utility-viewpoint
            self.planner = Planner(max_range, sim_thred, plan_mode, 
                                self.hysteresis_gain, self.hysteresis_radius, self.info_multiplier)
            print('use planner, mode:', plan_mode)
        self.next_best_pose = [0, 0]
        self.next_subgoal_pose = None
        self.next_subgoal_idx = -1
        self.finished_subgoal_idxs = []
        self.finished_next_poses = []
        self.isFinished = True
        
        ## subscribers and publishers
        self.goal_pub = rospy.Publisher(goal_topic, PointArray, queue_size=10)
        self.viewpoint_goal_pub = rospy.Publisher(viewpoint_goal_topic, PointArray, queue_size=10)
        self.frontier_pub = rospy.Publisher(frontier_pub_topic, PoseArray, queue_size=20)
        self.nbv_pub = rospy.Publisher('/nbv_point', Point, queue_size=10)
        rospy.Subscriber(frontiers_topic, PointArray, self.ftr_callback)
        rospy.Subscriber(viewpoint_topic, PointStamped, self.viewpoint_callback)
        rospy.Subscriber(viewpoint_score_topic, Point, self.viewpoint_score_callback)
        rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)
        while not len(self.mapData.data) or not self.get_tf():
            pass
        rospy.sleep(2)
        self.cur_pose = self.get_tf()[:2]
        rospy.Subscriber(finished_topic, PointStamped, self.finished_callback)
        rospy.Subscriber(subgoal_topic, PointStamped, self.subgoal_callback)

        #! only for test
        if _params['inter_point']:
            # cluster
            self.inter_buffer_size = _params['inter_buffer_size'] 
            self.min_inter_dist = _params['min_inter_dist'] 
            self.max_inter_dist = _params['max_inter_dist'] 
            self.inter_cluster_dist = _params['inter_cluster_dist']
            self.inter_cluster_fn = AgglomerativeClustering(n_clusters=None, metric='euclidean', 
                                            linkage='ward', distance_threshold=self.inter_cluster_dist, 
                                            compute_full_tree=True)
            self.nearest_centroid = NearestCentroid()
            rospy.Subscriber(intermediate_topic, PointStamped, self.interpoint_callback)
        
        trial_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
        self.output_path = f'/home/oem/rrt_explore/src/multi-robot-rrt-exploration-noetic/rrt_exploration/scripts/exp_figs/nbvp_{trial_id}'
        os.makedirs(self.output_path, exist_ok=True)


    def get_tf(self, duration=5.0):
        try:
            self.tf_listener.waitForTransform(self.map_frame,
                                            self.base_frame,
                                            rospy.Time(0),
                                            rospy.Duration(duration))
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame,
                                                            self.base_frame,
                                                            rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException) as e:
            rospy.loginfo(f"TF exception, cannot get odom pose. {e}")
            return
        point = Point(*trans)
        return [round(point.x, self.decimal), 
                round(point.y, self.decimal), 
                round(point.z, self.decimal)]

                
    def map_callback(self, msg):
        self.mapData = msg
        self.map_array = np.asarray(msg.data).reshape((msg.info.height,
                                        msg.info.width))
        
    def pub_cur_frontiers(self,):
        ftr_unvisited = [self.frontiers[j] for j in range(len(self.frontiers)) if not self.ftr_visited[j]]
        ftr_unvisited_map = [_world_to_map(self.mapData, pose) for pose in ftr_unvisited]
        filter_idxs = [j for j, pose_map in enumerate(ftr_unvisited_map) \
                       if self.map_array[pose_map] == OBSTACLE \
                       or not unknown_check(pose_map, self.map_array)]
        for idx in filter_idxs:
            ftr = ftr_unvisited[idx]
            self.ftr_visited[self.frontiers.index(ftr)] = 0
        ftr_unvisited = [ftr_unvisited[j] for j in range(len(ftr_unvisited)) if j not in filter_idxs]
        msg = PoseArray()
        msg.header.frame_id = self.goal_frame
        msg.header.stamp = rospy.Time().now()
        for ftr in ftr_unvisited:
            pose = Pose()
            pose.position.x = ftr[0]
            pose.position.y = ftr[1]
            msg.poses.append(pose)
        self.frontier_pub.publish(msg)
        
    def ftr_callback(self, msg):
        valid_ftr = False
        for point in msg.points:
            frontier = [round(point.x, self.decimal), round(point.y, self.decimal)]
            if not len(self.frontiers) or np.min(norm(array(frontier) 
                                                      - array(self.frontiers), axis=1)) > self.ftr_cluster_dist:
                valid_ftr = True
                self.frontiers.append(frontier)
                self.ftr_visited.append(0)
                rospy.loginfo(f'get new frontier:{frontier}')
        if valid_ftr:
            self.pub_cur_frontiers()
                
    def viewpoint_score_callback(self, msg):
        self.viewpoints_score.append(msg.x)
    
    def viewpoint_callback(self, msg):
        x, y, z = msg.point.x, msg.point.y, msg.point.z
        self.viewpoints.append([x, y, z])
        self.viewpoints_visited.append(0)
        while len(self.viewpoints) != len(self.viewpoints_score):
            pass
        rospy.loginfo(f'get a viewpoint:{[x,y,z]}, score:{self.viewpoints_score[-1]}')

    
    def interpoint_callback(self, msg):
        point = [round(msg.point.x, self.decimal), round(msg.point.y, self.decimal)]

        if not len(self.inter_centroids) \
            or np.min(norm(array(self.inter_centroids + self.frontiers + [self.cur_pose]) 
                           - array(point), axis=1)) > self.inter_cluster_dist:
            obstacles = np.where(self.map_array == OBSTACLE)
            obstacle_kd_tree = KDTree(np.vstack((obstacles[0], obstacles[1])).T)
            point_map = _world_to_map(self.mapData, point)
            dist, _ = obstacle_kd_tree.query(point_map)
            if self.min_inter_dist <= dist:
                self.inter_points.append(point)
        if len(self.inter_points) >= self.inter_buffer_size:
            clusters = self.inter_cluster_fn.fit(self.inter_points)
            labels = list(clusters.labels_)
            if all(labels == labels[0]): #! NearestCentroid requires input labels to be greater than 1
                nearest = np.argmin(norm(array(self.inter_points) 
                                         - array(np.mean(self.inter_points, axis=0)), axis=1))
                centroids = [self.inter_points[nearest]]
            else:  
                self.nearest_centroid.fit(self.inter_points, labels)
                centroids = np.round(self.nearest_centroid.centroids_, self.decimal).tolist()
            
            self.inter_centroids.extend(centroids)
            rospy.loginfo(f'get inter point:{centroids}')
            self.inter_visited += [0] * len(centroids)
            self.inter_points = []

    
    def finished_callback(self, msg):
        point = [round(msg.point.x, self.decimal), round(msg.point.y, self.decimal)]
        viewpoints = [point[:2] for point in self.viewpoints]
        self.isFinished = True
        if point in self.frontiers:
            self.ftr_visited[self.frontiers.index(point)] = 1
            rospy.loginfo(f'finish a frontier:{point}')
            self.pub_cur_frontiers()
        elif point in self.inter_centroids:
            self.inter_visited[self.inter_centroids.index(point)] = 1
            rospy.loginfo(f'finish an intermediate:{point}')
        elif point in viewpoints:
            self.viewpoints_visited[viewpoints.index(point)] = 1
            rospy.loginfo(f'finish a viewpoints:{point}')
        else:
            rospy.loginfo(f'finish an unmatched point:{point}')
    
    
    def subgoal_callback(self, msg):
        self.next_subgoal_pose = [round(msg.point.x, self.decimal), round(msg.point.y, self.decimal)]
        self.next_subgoal_idx = msg.point.z
        rospy.loginfo(f'get next_subgoal_pose:{self.next_subgoal_pose}')
        
    
    def send_route(self, centroids, yaws, route, _map, cur_pose, next_subgoal_pose, ftr_len, itr_len, step, show=False):
        arraypoints = PointArray()
        arraypoints_view = PointArray()
        tempVisPoint = Point()
        arraypoints.points = []
        arraypoints_view.points = []
        points = []
        map_array = copy(self.map_array)
        height = _map.info.height
        obstacles = np.where(map_array == OBSTACLE)
        unknowns = np.where(map_array == UNKNOWN)
        ftrs_map = []
        inters_map = []
        views_map = []
        cur_pose_map = _world_to_map_ori(_map, cur_pose)
        lines_map = [cur_pose_map]
        for idx in route:
            tempPoint = Point()
            if idx == -1: 
                assert next_subgoal_pose is not None
                tempPoint.x = next_subgoal_pose[0]
                tempPoint.y = next_subgoal_pose[1]
                arraypoints.points.append(copy(tempPoint))
                ctr_map = _world_to_map_ori(_map, next_subgoal_pose)
                ftrs_map.append(copy(ctr_map))
            else:
                tempPoint.x = centroids[idx][0]
                tempPoint.y = centroids[idx][1]
                tempPoint.z = yaws[idx]  # use z to store yaw data
                ctr_map = _world_to_map_ori(_map, centroids[idx])
                lines_map.append(copy(ctr_map))
                if idx < ftr_len:
                    ftrs_map.append(copy(ctr_map))
                    arraypoints.points.append(copy(tempPoint))
                elif idx < ftr_len + itr_len:
                    inters_map.append(copy(ctr_map))
                    arraypoints.points.append(copy(tempPoint))
                else:
                    views_map.append(copy(ctr_map))
                    arraypoints_view.points.append(copy(tempPoint))
        if show:
            ftrs_map = array(ftrs_map)
            inters_map = array(inters_map)
            views_map = array(views_map)
            lines_map = array(lines_map)
            size = 1
            plt.scatter(obstacles[1], obstacles[0], color='black', s=size, label='obstacles')
            plt.scatter(unknowns[1], unknowns[0], color='gray', s=size, label='unknowns')
            if len(ftrs_map):
                plt.scatter(ftrs_map[:, 0], ftrs_map[:, 1], color='red', s= size, label='frontiers')
            if len(inters_map):
                plt.scatter(inters_map[:,  0], inters_map[:, 1], color='blue', s=size, label='interpoints')
            if len(views_map):
                plt.scatter(views_map[:,  0], views_map[:, 1], color='green', s=size, label='viewpoints')
            plt.scatter(cur_pose_map[0], cur_pose_map[1], color='orange', marker='*', label='current')
            plt.plot(lines_map[:, 0], lines_map[:, 1], color='skyblue', label='route')
            for point, pose in zip(lines_map, [cur_pose] + centroids):
                plt.text(point[0], point[1], f'{pose}', fontsize=10)
            # plt.legend()
            plt.savefig(self.output_path + f'/route_{step}.png')
            plt.clf()
        
        if len(views_map):
            self.viewpoint_goal_pub.publish(arraypoints_view)
        if len(ftrs_map) + len(inters_map):
            self.goal_pub.publish(arraypoints)
        self.rate.sleep()
        
    
    def run(self,):
        finished_list = []
        prev_candidates = []
        prev_yaw_list = []
        prev_matrix = None
        prev_ftr_num = 0
        prev_itr_num = 0
        prev_unvisited_viewpoints = []
        step = 0
        while input('press enter to start\n') != '':
            continue
        while not rospy.is_shutdown():
            if not self.isFinished:
                continue
            
            ## get unvisited frontiers, intermediates, viewpoints
            unvisited_ftr_idxs = np.where(array(self.ftr_visited) == 0)[0]
            unvisited_inter_idxs = np.where(array(self.inter_visited) == 0)[0]
            unvisited_view_idxs = np.where(array(self.viewpoints_visited) == 0)[0]

            frontiers_list = copy(self.frontiers)
            ftr_unvisited = get_elements_by_idxs(frontiers_list, unvisited_ftr_idxs)
            
            inters_list = copy(self.inter_centroids)
            itr_unvisited = get_elements_by_idxs(inters_list, unvisited_inter_idxs)
            
            views_list = copy(self.viewpoints)
            view_unvisited = get_elements_by_idxs(views_list, unvisited_view_idxs)
            while len(self.viewpoints_score) < len(views_list):
                pass
            view_scores = get_elements_by_idxs(self.viewpoints_score, unvisited_view_idxs)
            if len(view_unvisited) != len(prev_unvisited_viewpoints):
                rospy.loginfo(f'view_unvisited:{view_unvisited}')
            prev_unvisited_viewpoints = view_unvisited
            
            ## calculate information gain and optimize yaw for frontiers
            candidates = ftr_unvisited + itr_unvisited + view_unvisited
            mapData = copy(self.mapData)
            dist_matrix = []
            cur_pose = self.get_tf()[:2]
            self.cur_pose = cur_pose
            ftr_argmax_yaws = []
            ftr_infogains = []
            costs = []
            raw_infos = []
            
            itr_yaws = [WAYPOINT_YAW] * len(itr_unvisited)
            view_yaws = [point[2] for point in view_unvisited]
            ### filter unreachable(obstacle) candidates
            ftr_unvisited_map = [_world_to_map(self.mapData, pose) for pose in ftr_unvisited]
            filter_idxs = [j for j, pose_map in enumerate(ftr_unvisited_map) \
                           if self.map_array[pose_map] == OBSTACLE \
                           or not unknown_check(pose_map, self.map_array)]
            for idx in filter_idxs:
                cand = candidates[idx]
                if cand in ftr_unvisited:
                    self.ftr_visited[frontiers_list.index(cand)] = 1
                    ftr_unvisited.remove(cand)

            for i, ctr in enumerate(ftr_unvisited):
                infogain, argmax_yaw = informationGainFOV(mapData,[ctr[0], ctr[1]], self.max_depth_dist,
                                                        self.stride, self.hfov_angle, count_state = UNKNOWN)
                tmp_infogain = infogain
                cost = get_list_dist(ctr, cur_pose)		
                raw_infos.append(infogain)
                costs.append(cost)
                
                if cost <= self.hysteresis_radius:
                    infogain *= self.hysteresis_gain
                infogain = infogain * self.info_multiplier - cost
                ftr_argmax_yaws.append(argmax_yaw)
                ftr_infogains.append(infogain)

            yaw_list = ftr_argmax_yaws + itr_yaws + view_yaws 
            candidates = ftr_unvisited + itr_unvisited + view_unvisited
            if len(filter_idxs):
                rospy.loginfo(f'filter num:{len(filter_idxs)}, idx:{filter_idxs}')
            if not len(candidates):
                prev_candidates = []
                prev_matrix = []
                prev_yaw_list = []
                continue
            
            ## next best candidate selection
            if self.use_planner:
                
                next_best_idx = self.planner(ftr_unvisited + itr_unvisited, view_unvisited, cur_pose, self.next_subgoal_pose, ftr_infogains, view_scores)
                routes = [next_best_idx]
                if next_best_idx == -1:
                    self.next_best_pose = [0, 0]
                else:
                    self.next_best_pose = candidates[next_best_idx]
                nbv_point = Point()
                nbv_point.x = self.next_best_pose[0]
                nbv_point.y = self.next_best_pose[1]
                self.nbv_pub.publish(nbv_point)

                if next_best_idx >= len(ftr_unvisited):
                    rospy.loginfo(f'next_best_idx:{next_best_idx}, is viewpoint')
                elif next_best_idx == -1:
                    rospy.loginfo(f'next_best_idx:{next_best_idx}, is next_subgoal_pose')
                else:
                    rospy.loginfo(f'next_best_idx:{next_best_idx}, is frontier, ftr_infogains:{ftr_infogains[next_best_idx]}, dist:{costs[next_best_idx]}')
            else:
                unfinished_idxs = []
                for j, cand in enumerate(prev_candidates):
                    if cand in frontiers_list and self.ftr_visited[frontiers_list.index(cand)] == 0:
                        unfinished_idxs.append(j)
                    elif cand in inters_list and self.inter_visited[inters_list.index(cand)] == 0:
                        unfinished_idxs.append(j)
                
                routes, total_distance, dist_matrix, = \
                    solve_tsp(mapData, cur_pose, candidates, unfinished_idxs, 
                              prev_matrix, prev_candidates, show=False)
            # rospy.loginfo(f'routes:{routes}')
            # rospy.loginfo(f'ftr_infogains:{ftr_infogains}')
            
            ## send goal points to local planner
            self.send_route(candidates, yaw_list, routes, mapData, cur_pose, copy(self.next_subgoal_pose),
                            len(ftr_unvisited), len(itr_unvisited), step, show=False)  
            rospy.loginfo(f'candidates:{candidates}')
            # rospy.loginfo(f'{yaw_list}')
            prev_candidates = copy(candidates)
            prev_matrix = copy(dist_matrix)
            prev_yaw_list = copy(yaw_list)
            step += 1
            self.isFinished = False
            self.rate.sleep()
            
if __name__ == '__main__':
    node = NBVP()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass