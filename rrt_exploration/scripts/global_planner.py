## use ompl with python-bindings
# install guide: https://ompl.kavrakilab.org/core/installation.html
#!/usr/bin/env python3
# coding: utf-8

import argparse
from os.path import abspath, dirname, join
import sys
from numpy import sqrt
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import rospy
from nav_utils import _world_to_map


class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, distance_map, valid_dist):
        super().__init__(si)
        self.distance_map = distance_map
        self.valid_dist = valid_dist
        self.height = distance_map.shape[0]
        self.width = distance_map.shape[1]

    def isValid(self, state):
        return self.clearance(state) > 0.0
        # x = int(state[0])
        # y = int(state[1])
        # rospy.loginfo(f'dist:{self.distance_map[x, y]}')
        # x = min(self.height, self.height - x)
        # if self.distance_map[x, y] > self.valid_dist:
        #     rospy.loginfo(f'omple valid point:{[x, y, self.distance_map[x, y]]}')
        # return self.distance_map[x, y] > self.valid_dist  #!
        # return self.distance_map[self.height - x - 1, y] > self.valid_dist  #!
    
    def clearance(self, state):  #! define clearance function is important otherwise collision!
        x = int(state[0])
        y = int(state[1])
        return self.distance_map[self.height - x - 1, y] - self.valid_dist
#         # Extract the robot's (x,y) position from its state
#         x = state[0]
#         y = state[1]

#         # Distance formula between two points, offset by the circle's
#         # radius
#         return sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)) - 0.25

# Our "collision checker". For this demo, our robot's state space
# lies in [0,1]x[0,1], with a circular obstacle of radius 0.25
# centered at (0.5,0.5). Any states lying in this circular region are
# considered "in collision".
# class ValidityChecker(ob.StateValidityChecker):
#     # Returns whether the given state's position overlaps the
#     # circular obstacle
#     def isValid(self, state):
#         return self.clearance(state) > 0.0

#     # Returns the distance from the given state's position to the
#     # boundary of the circular obstacle.
#     def clearance(self, state):
#         # Extract the robot's (x,y) position from its state
#         x = state[0]
#         y = state[1]

#         # Distance formula between two points, offset by the circle's
#         # radius
#         return sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)) - 0.25


## Returns a structure representing the optimization objective to use
#  for optimal motion planning. This method returns an objective
#  which attempts to minimize the length in configuration space of
#  computed paths.
def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)

## Returns an optimization objective which attempts to minimize path
#  length that is satisfied when a path of length shorter than 1.51
#  is found.
def getThresholdPathLengthObj(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    return obj

## Defines an optimization objective which attempts to steer the
#  robot away from obstacles. To formulate this objective as a
#  minimization of path cost, we can define the cost of a path as a
#  summation of the costs of each of the states along the path, where
#  each state cost is a function of that state's clearance from
#  obstacles.
#
#  The class StateCostIntegralObjective represents objectives as
#  summations of state costs, just like we require. All we need to do
#  then is inherit from that base class and define our specific state
#  cost function by overriding the stateCost() method.
#
class ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    # Our requirement is to maximize path clearance from obstacles,
    # but we want to represent the objective as a path cost
    # minimization. Therefore, we set each state's cost to be the
    # reciprocal of its clearance, so that as state clearance
    # increases, the state cost decreases.
    def stateCost(self, s):
        return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s) +
                            sys.float_info.min))

## Return an optimization objective which attempts to steer the robot
#  away from obstacles.
def getClearanceObjective(si):
    return ClearanceObjective(si)

## Create an optimization objective which attempts to optimize both
#  path length and clearance. We do this by defining our individual
#  objectives, then adding them to a MultiOptimizationObjective
#  object. This results in an optimization objective where path cost
#  is equivalent to adding up each of the individual objectives' path
#  costs.
#
#  When adding objectives, we can also optionally specify each
#  objective's weighting factor to signify how important it is in
#  optimal planning. If no weight is specified, the weight defaults to
#  1.0.
def getBalancedObjective1(si):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    clearObj = ClearanceObjective(si)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, 5.0)
    opt.addObjective(clearObj, 1.0)

    return opt

## Create an optimization objective equivalent to the one returned by
#  getBalancedObjective1(), but use an alternate syntax.
#  THIS DOESN'T WORK YET. THE OPERATORS SOMEHOW AREN'T EXPORTED BY Py++.
# def getBalancedObjective2(si):
#     lengthObj = ob.PathLengthOptimizationObjective(si)
#     clearObj = ClearanceObjective(si)
#
#     return 5.0*lengthObj + clearObj


## Create an optimization objective for minimizing path length, and
#  specify a cost-to-go heuristic suitable for this optimal planning
#  problem.
def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
    return obj

# Keep these in alphabetical order and all lower case
def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

# Keep these in alphabetical order and all lower case
def allocateObjective(si, objectiveType):
    if objectiveType.lower() == "pathclearance":
        return getClearanceObjective(si)
    elif objectiveType.lower() == "pathlength":
        return getPathLengthObjective(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return getThresholdPathLengthObj(si)
    elif objectiveType.lower() == "weightedlengthandclearancecombo":
        return getBalancedObjective1(si)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")
 
# def ValidityChecker():
    

def ompl_plan(start, goal, distance_map, mapData,
    validDist=0.8, runTime=1, plannerType='informedrrtstar', objectiveType='weightedlengthandclearancecombo', stepInterval=2, fname=None):


    # Construct the robot state space in which we're planning. We're
    # planning in [0,1]x[0,1], a subset of R^2.
    space = ob.RealVectorStateSpace(2)

    # Set the bounds of space to be in [0,1].
    h, w = distance_map.shape
    # h, w = 20, 30
    # rospy.loginfo(distance_map.shape)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, 0)
    bounds.setLow(1, 0)
    bounds.setHigh(0, h)  # 0 is y axis
    bounds.setHigh(1, w)  # 1 is x axis
    space.setBounds(bounds)
    # space.setBounds(0.0, 1.0)
    
    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)

    # Set the object used to check which states in the space are valid
    validityChecker = ValidityChecker(si, distance_map, validDist) # , distance_map, validDist
    # isValidFn = ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
    si.setStateValidityChecker(validityChecker)
    si.setup()

    # Set our robot's starting state to be the bottom-left corner of
    # the environment, or (0,0).
    start_state = ob.State(space)
    start_state[0] = start[0] #start[1]  # 0.0 # start[1]   
    start_state[1] = start[1] #h - start[0] # 0.0 #  start[0]

    # Set our robot's goal state to be the top-right corner of the
    # environment, or (1,1).
    goal_state = ob.State(space)
    # print(goal_state)
    goal_state[0] =  goal[0] # goal[1] # 30.0 # goal[1]
    goal_state[1] =  goal[1] #h - goal[0]  # 20.0 # goal[0]  

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    # Set the start and goal states
    pdef.setStartAndGoalStates(start_state, goal_state)

    # Create the optimization objective specified by our command-line argument.
    # This helper function is simply a switch statement.
    pdef.setOptimizationObjective(allocateObjective(si, objectiveType))

    # Construct the optimal planner specified by our command line argument.
    # This helper function is simply a switch statement.
    optimizingPlanner = allocatePlanner(si, plannerType)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(runTime)
    if solved:
        path = pdef.getSolutionPath()
        path_list = []
        for state in path.getStates():
            path_list.append((int(state[0]), int(state[1])))
        discrete_path_list = path_list
        return discrete_path_list
        # for solved
        # # Output the length of the path found
        # print('{0} found solution of path length {1:.4f} with an optimization ' \
        #     'objective value of {2:.4f}'.format( \
        #     optimizingPlanner.getName(), \
        #     pdef.getSolutionPath().length(), \
        #     pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

        # # If a filename was specified, output the path as a matrix to
        # # that file for visualization
        # if fname:
        #     with open(fname, 'w') as outFile:
        #         outFile.write(pdef.getSolutionPath().printAsMatrix())
    else:
        # print("No solution found.")
        return None

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

    # Add a filename argument
    parser.add_argument('-t', '--runtime', type=float, default=1.0, help=\
        '(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.')
    parser.add_argument('-p', '--planner', default='RRTstar', \
        choices=['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
        'SORRTstar'], \
        help='(Optional) Specify the optimal planner to use, defaults to RRTstar if not given.')
    parser.add_argument('-o', '--objective', default='PathLength', \
        choices=['PathClearance', 'PathLength', 'ThresholdPathLength', \
        'WeightedLengthAndClearanceCombo'], \
        help='(Optional) Specify the optimization objective, defaults to PathLength if not given.')
    parser.add_argument('-f', '--file', default=None, \
        help='(Optional) Specify an output path for the found solution path.')
    parser.add_argument('-i', '--info', type=int, default=0, choices=[0, 1, 2], \
        help='(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG.' \
        ' Defaults to WARN.')

    # Parse the arguments
    args = parser.parse_args()

    # Check that time is positive
    if args.runtime <= 0:
        raise argparse.ArgumentTypeError(
            "argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)" \
            % (args.runtime,))

    # Set the log level
    if args.info == 0:
        ou.setLogLevel(ou.LOG_WARN)
    elif args.info == 1:
        ou.setLogLevel(ou.LOG_INFO)
    elif args.info == 2:
        ou.setLogLevel(ou.LOG_DEBUG)
    else:
        ou.OMPL_ERROR("Invalid log-level integer.")

    # Solve the planning problem
    
    paths = ompl_plan(None, None, None, 0.5, args.runtime, args.planner, args.objective, args.file)
    if paths is not None:
        print(paths)
        path_array = np.array([[0, 0]] + paths + [[30, 20]])
        
        plt.plot(path_array[:, 0], path_array[:, 1])
        plt.savefig('./ompl_path.png')