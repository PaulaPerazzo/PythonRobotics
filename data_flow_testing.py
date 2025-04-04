import math

from PathPlanning.AStar.a_star import AStarPlanner

# Helper to create an AStarPlanner instance with a controlled grid.
def create_planner():
    # We provide two obstacle points far from the intended planning region.
    # This sets grid boundaries from 0 to 10 (since min=0 and max=10) with resolution 1.
    ox = [0, 10]
    oy = [0, 10]
    resolution = 1.0
    rr = 0.5
    return AStarPlanner(ox, oy, resolution, rr)

def test_calc_xy_index():
    planner = create_planner()
    # For position 5 with min_x=0 and resolution=1, expected index is round(5-0)=5.
    index = planner.calc_xy_index(5, planner.min_x)
    expected = round((5 - planner.min_x) / planner.resolution)
    assert index == expected, f"Expected index {expected}, got {index}"

def test_calc_grid_position():
    planner = create_planner()
    # For index 5, position should be 5*resolution + min_x.
    pos = planner.calc_grid_position(5, planner.min_x)
    expected = 5 * planner.resolution + planner.min_x
    assert math.isclose(pos, expected), f"Expected position {expected}, got {pos}"

def test_verify_node_out_of_bounds():
    planner = create_planner()
    # Create a node with a negative grid index to simulate an out-of-bound condition.
    node = planner.Node(-1, 5, 0, -1)
    assert not planner.verify_node(node), "Node with negative index should be out-of-bounds"

def test_verify_node_in_bounds():
    planner = create_planner()
    # Use a node that lies well within the grid (and away from obstacles)
    node = planner.Node(5, 5, 0, -1)
    assert planner.verify_node(node), "Node (5,5) should be in bounds and safe"

def test_planning_path():
    planner = create_planner()
    # Choose start and goal that lie within the grid (grid boundaries from 0 to 10)
    sx, sy = 1, 1
    gx, gy = 8, 8
    rx, ry = planner.planning(sx, sy, gx, gy)
    
    # The planning method builds the path in reverse (goal first, then back to start)
    # Check that the first point is the goal and the last is the start.
    assert rx, "Returned x path should not be empty"
    assert ry, "Returned y path should not be empty"
    
    # Allow a small tolerance in floating point comparisons
    tol = 1e-6
    assert math.isclose(rx[0], gx, abs_tol=tol) and math.isclose(ry[0], gy, abs_tol=tol), "Path should start at the goal"
    assert math.isclose(rx[-1], sx, abs_tol=tol) and math.isclose(ry[-1], sy, abs_tol=tol), "Path should end at the start"
