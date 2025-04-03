from unittest.mock import patch
import pytest
from PathPlanning.AStar.a_star import AStarPlanner


def test_invalid_obstacle_input():
  """Test that the planner raises an error when given empty obstacle lists."""
  with pytest.raises(ValueError):
    AStarPlanner([], [], 0.5, 0.2)


# ── Planning boundary tests ────────────────────────────────────────────────────

def test_start_at_map_boundary():
  """Test that planning works when the start point is at the edge of the map."""
  ox, oy = [10, 20], [10, 20]
  planner = AStarPlanner(ox, oy, 1.0, 0.5)

  rx, _ = planner.planning(planner.min_x, planner.min_y, 15, 15)
  assert len(rx) > 0

  rx, _ = planner.planning(planner.max_x, planner.max_y, 15, 15)
  assert len(rx) > 0


def test_goal_at_map_boundary():
  """Test that planning works when the goal point is at the edge of the map."""
  ox, oy = [10, 20], [10, 20]
  planner = AStarPlanner(ox, oy, 1.0, 0.5)

  rx, _ = planner.planning(15, 15, planner.min_x, planner.min_y)
  assert len(rx) > 0

  rx, _ = planner.planning(15, 15, planner.max_x, planner.max_y)
  assert len(rx) > 0


# ── Node and heuristic ─────────────────────────────────────────────────────────

def test_node_str_representation():
  """Test the string representation of a Node instance."""
  node = AStarPlanner.Node(x=5, y=10, cost=2.5, parent_index=3)
  expected = "5,10,2.5,3"
  assert str(node) == expected


def test_calc_heuristic():
  """Test the heuristic function calculates correct Euclidean distance."""
  n1 = AStarPlanner.Node(2, 3, 0, -1)
  n2 = AStarPlanner.Node(5, 7, 0, -1)
  expected = 5.0 * 1.0
  assert AStarPlanner.calc_heuristic(n1, n2) == expected


def test_calc_heuristic_failure():
  """Fail when nodes have invalid attributes (non-numeric or missing)."""
  bad_node = type("BadNode", (), {"x": "a", "y": 3})()
  good_node = AStarPlanner.Node(0, 0, 0, -1)
  with pytest.raises(TypeError):
    AStarPlanner.calc_heuristic(bad_node, good_node)


# ── Node verification ──────────────────────────────────────────────────────────

def test_verify_node():
  """Test node validity under various map and obstacle conditions using mocking."""
  ox = [1.0, 2.0, 3.0]
  oy = [1.0, 2.0, 3.0]
  planner = AStarPlanner(ox=ox, oy=oy, resolution=1.0, rr=0.5)
  planner.min_x = 0
  planner.min_y = 0
  planner.max_x = 10
  planner.max_y = 10
  planner.x_width = 10
  planner.y_width = 10
  planner.obstacle_map = [[False for _ in range(planner.y_width)] for _ in range(planner.x_width)]
  node = planner.Node(x=5, y=5, cost=0, parent_index=-1)

  with patch.object(planner, "calc_grid_position", side_effect=[5.0, 5.0]):
    assert planner.verify_node(node) is True

  planner.obstacle_map[5][5] = True
  with patch.object(planner, "calc_grid_position", side_effect=[5.0, 5.0]):
    assert planner.verify_node(node) is False
  planner.obstacle_map[5][5] = False

  with patch.object(planner, "calc_grid_position", side_effect=[-1.0, 5.0]):
    assert planner.verify_node(node) is False


def test_verify_node_failure_cases():
  """Fail when node is out of bounds or inside obstacle."""
  planner = AStarPlanner([0], [0], resolution=1.0, rr=0.1)
  planner.min_x = 0
  planner.min_y = 0
  planner.max_x = 5
  planner.max_y = 5
  planner.x_width = 5
  planner.y_width = 5
  planner.obstacle_map = [[False] * 5 for _ in range(5)]
  node = planner.Node(x=10, y=10, cost=0, parent_index=-1)

  with patch.object(planner, 'calc_grid_position', side_effect=[10, 10]):
    assert not planner.verify_node(node)


# ── Grid index and conversions ─────────────────────────────────────────────────

def test_calc_grid_index():
  """Test that the grid index is calculated correctly from node coordinates."""
  planner = AStarPlanner([1.0, 2.0], [1.0, 2.0], 1.0, 0.5)
  planner.min_x = 0
  planner.min_y = 0
  planner.x_width = 10
  planner.y_width = 10

  node = planner.Node(x=3, y=4, cost=0, parent_index=-1)
  assert planner.calc_grid_index(node) == 43

  node = planner.Node(x=0, y=0, cost=0, parent_index=-1)
  assert planner.calc_grid_index(node) == 0

  node = planner.Node(x=9, y=9, cost=0, parent_index=-1)
  assert planner.calc_grid_index(node) == 99


def test_calc_grid_index_failure():
  """Fail when node has non-integer x or y attributes."""
  planner = AStarPlanner([0], [0], 1.0, 0.1)
  planner.min_x = 0
  planner.min_y = 0
  planner.x_width = 10
  planner.y_width = 10
  bad_node = type("BadNode", (), {"x": "a", "y": 1})()
  with pytest.raises(TypeError):
    planner.calc_grid_index(bad_node)


def test_calc_xy_index():
  """Test conversion from world coordinates to grid indices."""
  planner = AStarPlanner([0.0], [0.0], 0.5, 0.1)
  planner.min_x = -5.0

  assert planner.calc_xy_index(0.0, planner.min_x) == 10
  assert planner.calc_xy_index(-5.0, planner.min_x) == 0
  assert planner.calc_xy_index(2.3, planner.min_x) == 15


def test_calc_xy_index_failure():
  """Fail when position is NaN or resolution is zero (div by zero)."""
  planner = AStarPlanner([0], [0], 1.0, 0.1)

  with pytest.raises(TypeError):
    planner.calc_xy_index("invalid", planner.min_x)

  planner.resolution = 0
  with pytest.raises(ZeroDivisionError):
    planner.calc_xy_index(1.0, planner.min_x)


def test_calc_grid_position():
  """Test conversion from grid index back to world coordinates."""
  planner = AStarPlanner([0.0], [0.0], 0.5, 0.1)
  planner.min_x = -5.0
  planner.min_y = -3.0

  assert planner.calc_grid_position(0, planner.min_x) == -5.0
  assert planner.calc_grid_position(0, planner.min_y) == -3.0
  assert planner.calc_grid_position(4, planner.min_x) == -3.0
  assert planner.calc_grid_position(6, planner.min_y) == 0.0
  assert abs(planner.calc_grid_position(3, planner.min_x) + 3.5) < 1e-9


def test_calc_grid_position_failure():
  """Fail when index or min_position is invalid (non-numeric)."""
  planner = AStarPlanner([0], [0], 1.0, 0.1)
  with pytest.raises(TypeError):
    planner.calc_grid_position("x", planner.min_x)


def test_conversion_methods_edge_cases():
  """Test edge cases for coordinate conversion and grid index calculations."""
  planner = AStarPlanner([0, 10], [0, 10], 0.5, 0.1)
  planner.min_x = -2.0
  planner.min_y = -5.0
  planner.x_width = 100
  planner.y_width = 100

  assert planner.calc_xy_index(-1.0, planner.min_x) == 2
  assert planner.calc_grid_position(0, planner.min_y) == -5.0

  node = planner.Node(x=1000, y=1000, cost=0, parent_index=-1)
  index = planner.calc_grid_index(node)
  assert index >= planner.x_width * planner.y_width


# ── Path planning ──────────────────────────────────────────────────────────────

def test_planning_deterministic():
  """Test that the planner finds a specific mocked path when possible."""
  planner = AStarPlanner([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1.0, 0.1)
  planner.min_x = 0
  planner.min_y = 0
  planner.max_x = 3
  planner.max_y = 3
  planner.x_width = 3
  planner.y_width = 3
  planner.obstacle_map = [[False for _ in range(3)] for _ in range(3)]

  with patch.object(planner, 'calc_final_path', return_value=([0, 1, 2], [0, 0, 0])):
    path = planner.planning(0, 0, 2, 0)
    assert path == ([0, 1, 2], [0, 0, 0])


def test_calc_final_path_failure():
  """Fail case: parent index not found in closed set."""
  planner = AStarPlanner([0], [0], 1.0, 0.1)
  fake_goal = planner.Node(0, 0, 0, 999)
  with pytest.raises(KeyError):
    planner.calc_final_path(fake_goal, {})


def test_planning_with_obstacles():
  """Test planning through a path with known obstacle layout and mocked path."""
  planner = AStarPlanner([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], 1.0, 0.1)

  with patch.object(planner, 'calc_final_path', return_value=([0, 1, 2, 2], [0, 0, 1, 2])):
    path = planner.planning(0, 0, 2, 2)
    assert path == ([0, 1, 2, 2], [0, 0, 1, 2])


def test_planning_no_path():
  """Test that the planner returns an empty or minimal path when no path is available."""
  ox = [0.0, 1.0, 2.0] * 3
  oy = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
  planner = AStarPlanner(ox, oy, 1.0, 0.5)
  planner.obstacle_map = [[True] * 3 for _ in range(3)]

  path = planner.planning(0, 0, 2, 2)
  assert path == ([2.0], [2.0]) or path == ([], [])


def test_planning_failure_due_to_empty_open_set():
  """Fail case: No valid path due to all nodes invalid."""
  planner = AStarPlanner([0, 1, 2], [0, 1, 2], 1.0, 0.1)

  with patch.object(planner, 'verify_node', return_value=False), \
       patch.object(planner, 'calc_final_path', return_value=([], [])):
    rx, ry = planner.planning(0, 0, 2, 2)
    assert rx == [] and ry == []


# ── Obstacle map ───────────────────────────────────────────────────────────────

def test_calc_obstacle_map():
  """Test that the obstacle map is built correctly and marks cells as occupied."""
  planner = AStarPlanner([1.0, 2.0], [1.0, 2.0], 1.0, 0.5)

  assert planner.x_width == 1
  assert planner.y_width == 1
  assert len(planner.obstacle_map) == planner.x_width
  assert len(planner.obstacle_map[0]) == planner.y_width
  assert planner.obstacle_map[0][0] is True


def test_obstacle_map_specific_case():
  """Test specific obstacle locations in the map are correctly identified."""
  planner = AStarPlanner([1.0, 3.0, 5.0], [1.0, 3.0, 5.0], 1.0, 0.5)

  assert planner.obstacle_map[0][0]
  assert planner.obstacle_map[2][2]
  assert not planner.obstacle_map[1][1]
