import itertools
import math
import pytest
from PathPlanning.AStar.a_star import AStarPlanner

# Define the parameter space for the tests.
params = {
    "grid_size": [1.0, 2.0],
    "robot_radius": [0.5, 1.0],
    "start_x": [10.0, 15.0],
    "start_y": [10.0, 15.0],
    "goal_x": [50.0, 51.0],
    "goal_y": [50.0, 55.0]
}

# Generate full factorial (all combinations) test cases.
def full_factorial(params):
    keys = list(params.keys())
    all_combos = list(itertools.product(*(params[k] for k in keys)))
    return [dict(zip(keys, combo)) for combo in all_combos]

# You can later also add a covering array generator if desired.
# For now, we use full factorial.
test_cases = full_factorial(params)

# Helper to create obstacles
def create_obstacles():
    ox, oy = [], []
    # Boundaries
    for i in range(-20, 80):
        ox.append(i)
        oy.append(-20.0)
    for i in range(-20, 80):
        ox.append(80.0)
        oy.append(i)
    for i in range(-20, 81):
        ox.append(i)
        oy.append(80.0)
    for i in range(-20, 81):
        ox.append(-20.0)
        oy.append(i)
    # Extra obstacles to force detours
    for i in range(-20, 50):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 50):
        ox.append(40.0)
        oy.append(80.0 - i)
    return ox, oy

@pytest.mark.parametrize("test_params", test_cases)
def test_a_star_planner(test_params):
    ox, oy = create_obstacles()
    grid_size   = test_params["grid_size"]
    robot_radius= test_params["robot_radius"]
    sx          = test_params["start_x"]
    sy          = test_params["start_y"]
    gx          = test_params["goal_x"]
    gy          = test_params["goal_y"]

    planner = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = planner.planning(sx, sy, gx, gy)

    # Check that the path is non-empty.
    assert rx, "Path (x coordinates) should not be empty"
    assert ry, "Path (y coordinates) should not be empty"

    # A* returns the path in reverse: first point is goal, last is start.
    tol = 1e-6
    assert math.isclose(rx[-1], sx, abs_tol=tol) and math.isclose(ry[-1], sy, abs_tol=tol), "Path should end at the start"
