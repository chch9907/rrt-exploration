"""Simple Travelling Salesperson Problem (TSP) between cities."""
#!/usr/bin/env python3
# coding: utf-8
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from numpy import array
from heapq import heappush, heappop
import math
import rospy
from nav_utils import _world_to_map


def _get_movements_4n():
    """
    Get all possible 4-connectivity movements.
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]


def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]


def dist2d(start, goal):
    return np.square((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)


def a_star(start, goal, raw_ogmap, movement='8N', thred=250):
    # https://github.com/richardos/occupancy-grid-a-star/blob/master/examples/occupancy_map_8n.py
    """
    A* for 2D occupancy grid.

    :param start: start node (x, y) in grid map
    :param goal: goal node (x, y) in grid map
    :param gmap: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :return: length of path
    """
    
    # ogmap = np.where(ogmap <= thred, 1, 0)  # 1: obstacles or unknown, 0: free
    ogmap = np.zeros_like(raw_ogmap)
    ogmap[np.where(raw_ogmap == 100)] = 1
    # rospy.loginfo(ogmap[start])
    # rospy.loginfo(ogmap[goal])
    # check if start and goal nodes correspond to free spaces
    # if ogmap[start]:
    #     rospy.logwarn(f'Start node {start} is not traversable')  # raise Exception
    #     return np.inf
    # if ogmap[goal]:
    #     rospy.logwarn(f'Goal node {goal} is not traversable')  # raise Exception
        # return np.inf

    # add start node to front
    # front is a list of (total estimated cost to goal, total cost from start to node, node, previous node)
    start_node_g = 0
    start_node_f = dist2d(start, goal) + start_node_g
    front = [(start_node_f, start_node_g, start, None)]

    # use a dictionary to remember where we came from in order to reconstruct the path later on
    came_from = {}  # linked list
    visited = np.zeros_like(ogmap)
    # get possible movements
    if movement == '4N':
        movements = _get_movements_4n()
    elif movement == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')

    # while there are elements to investigate in our front.
    while front:
        # get smallest item at index 0 and remove from front.
        element = heappop(front)

        # if this has been visited already, skip it
        total_cost, cost, pos, previous = element
        if visited[pos]:
            continue

        # now it has been visited, mark with cost
        visited[pos] = 1

        # set its previous node
        came_from[pos] = previous

        # if the goal has been reached, we are done!
        if pos == goal:
            break

        # check all neighbors
        for dx, dy, deltacost in movements:
            # determine new position
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            new_pos = (new_x, new_y)

            # check whether new position is inside the map
            # if not, skip node
            if (not 0 <= new_pos[0] < ogmap.shape[0]) or (not 0 <= new_pos[1] < ogmap.shape[1]):
                continue

            # add node to front if it was not visited before and is not an obstacle
            if (not visited[new_pos]) and (not ogmap[new_pos]):
                g = cost + deltacost
                h = dist2d(new_pos, goal)
                f = g + h 
                heappush(front, (f, g, new_pos, pos))

    # reconstruct path backwards (only if we reached the goal)
    path = []
    path_idx = []
    if pos == goal:
        while pos:
            path_idx.append(pos)
            # transform array indices to meters
            # pos_m_x, pos_m_y = ogmap.get_coordinates_from_index(pos[0], pos[1])
            # path.append((pos_m_x, pos_m_y))
            pos = came_from[pos]

        # reverse so that path is from start to goal.
        # path.reverse()
        # path_idx.reverse()
    return len(path_idx)
    
    
# def create_data_model(_data):
#     """Stores the data for the problem."""
#     data = {}
#     data["distance_matrix"] = [
#         [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
#         [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
#         [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
#         [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
#         [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
#         [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
#         [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
#         [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
#         [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
#         [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
#         [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
#         [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
#         [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
#     ]
#     data["num_vehicles"] = 1
#     data["depot"] = 0  #  the start and end location for the route
#     return data

    


def create_data_model(distance_matrix, start_idx=0):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = distance_matrix
    data["num_vehicles"] = 1
    data["depot"] = start_idx  #  the start and end location for the route
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    # print(f"Objective: {solution.ObjectiveValue()} miles")
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += f" {manager.IndexToNode(index)}\n"
    plan_output += f"Route distance: {route_distance}miles\n"
    print(plan_output)


def get_routes(solution, routing, manager):
  """Get vehicle routes from a solution and store them in an array."""
  # Get vehicle routes and store them in a two dimensional array whose
  # i,j entry is the jth location visited by vehicle i along its route.
  routes = []
  for route_nbr in range(routing.vehicles()):
    index = routing.Start(route_nbr)
    route = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
    routes.append(route)
  return routes[0]


def get_path_length(solution):
    return solution.ObjectiveValue()


def update_distance_matrix(ogmap, cur_pose, candidates, unfinished_idxs, prev_matrix, prev_candidates):
    ogmap_data = np.asarray(ogmap.data).reshape((ogmap.info.height,
                                        ogmap.info.width))
    if prev_matrix is not None:
        unfinished_matrix = prev_matrix[np.ix_(unfinished_idxs, unfinished_idxs)]
        unfinished_candidates = array(prev_candidates)[unfinished_idxs].tolist()
    else:
        unfinished_candidates = []
    candidates_map = [_world_to_map(ogmap, pose) for pose in candidates]
    # candidates_map = [pose_map for pose_map in candidates_map_ if not ogmap_data[pose_map] == 100]
    # filter_num = len(candidates_map_) - len(candidates_map)
    # if filter_num:
    #     rospy.loginfo(f'filter num:{filter_num}')
    candidates_map = [_world_to_map(ogmap, cur_pose)] + candidates_map
    # rospy.loginfo(f'candidates_map:{candidates_map}')
    
    num = len(candidates_map)  # 0 idx is current pose
    matrix = np.zeros((num, num))
    
    # rospy.loginfo(f'map shape:{ogmap_data.shape}, {ogmap.info.height}, {ogmap.info.width}')
    for i in range(len(candidates_map)):
        for j in range(i + 1, len(candidates_map)):
            # if i == j:
            #     matrix[i, j] = 0
            if j < len(unfinished_candidates) and i > 0:
                dist = unfinished_matrix[i, j]
            else:
                dist = a_star(candidates_map[i], candidates_map[j], ogmap_data)
            matrix[i, j] = dist
            # matrix[j, i] = dist
            if i != 0:  # any candidate to current position is zero cost
                matrix[j, i] = dist
    # rospy.loginfo(f'matrix:{matrix}')
    return matrix
                

def solve_tsp(ogmap, cur_pose, candidates, unfinished_idxs=None, prev_matrix=None, prev_candidates=[], start_idx=0, show=False):
    """Entry point of the program."""
    
    # get distance matrix
    distance_matrix = update_distance_matrix(ogmap, cur_pose, candidates, unfinished_idxs, prev_matrix, prev_candidates)
    
    # Instantiate the data problem.
    data = create_data_model(distance_matrix, start_idx)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        # routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if show:
        print_solution(manager, routing, solution)
    
    total_distance = solution.ObjectiveValue()
    routes = get_routes(solution, routing, manager)
    # rospy.loginfo(f'raw tsp routes:{routes}')
    routes = [idx - 1 for idx in routes[1:-1]]
    
    return routes, total_distance, distance_matrix

if __name__ == "__main__":
    routes, total_distance = solve_tsp()