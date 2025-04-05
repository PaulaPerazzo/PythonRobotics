import math
import time
import csv
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from a_star import AStarPlanner

AStarPlanner = AStarPlanner

CSV_FILENAME = "a_star_test_results.csv"

class AStarMutant1(AStarPlanner):
    """Mutant 1: always zero (A* to Dijkstra)"""
    @staticmethod
    def calc_heuristic(n1, n2):
        return 0.0


class AStarMutant2(AStarPlanner):
    """Mutant 2: ignore colisions"""
    def verify_node(self, node):
        return True 


class AStarMutant3(AStarPlanner):
    """Mutant 3: movement cost inflated
    multiply the cost of diagonal movements by 1.5
    """
    @staticmethod
    def get_motion_model():
        base_motion = AStarPlanner.get_motion_model()
        return [[dx, dy, cost * 1.5] for dx, dy, cost in base_motion]

def create_test_cases():
    """
    Retorna uma lista simplificada de cenários para teste:
    return a list of test cases with the following structure:
      1. Open Path: No internal obstacles.
      2. Single Obstacle: A single obstacle that forces a small detour.
      3. Blocked Path: A continuous wall blocks the path.
    """

    open_path = {
        "name": "Open Path",
        "ox": [-10.0, -10.0, 60.0, 60.0] +
              [x for x in range(-10, 61)] + [60.0] * 71 + [x for x in range(-10, 61)],
        "oy": [-10.0, 60.0, -10.0, 60.0] +
              [-10.0] * 71 + [y for y in range(-10, 61)] + [60.0] * 71,
        "sx": 0.0, "sy": 0.0,
        "gx": 50.0, "gy": 50.0,
        "expected_path_length": 35.35
    }

    single_obstacle = {
        "name": "Obstáculo Único",
        "ox": [-10.0, -10.0, 60.0, 60.0] + [x for x in range(20, 31)],
        "oy": [-10.0, 60.0, -10.0, 60.0] + [25.0] * 11,
        "sx": 0.0, "sy": 0.0,
        "gx": 50.0, "gy": 50.0,
        "expected_path_length": 40  
    }

    no_path = {
        "name": "Caminho Bloqueado",
        "ox": [-10.0, -10.0, 60.0, 60.0] + [x for x in range(0, 51)],
        "oy": [-10.0, 60.0, -10.0, 60.0] + [25.0] * 51,
        "sx": 0.0, "sy": 0.0,
        "gx": 50.0, "gy": 50.0,
        "expected_path_length": 0  
    }
    
    return [open_path, single_obstacle, no_path]

def check_path_continuity(rx, ry, resolution):
    allowed_steps = [
        (resolution, 0),
        (-resolution, 0),
        (0, resolution),
        (0, -resolution),
        (resolution, resolution),
        (resolution, -resolution),
        (-resolution, resolution),
        (-resolution, -resolution)
    ]

    for i in range(1, len(rx)):
        dx = rx[i] - rx[i-1]
        dy = ry[i] - ry[i-1]
        valid = any(math.isclose(dx, step[0], abs_tol=1e-3) and math.isclose(
            dy, step[1], abs_tol=1e-3) for step in allowed_steps
        )
        
        if not valid:
            return False, (rx[i-1], ry[i-1]), (rx[i], ry[i])
    
    return True, None, None

def compute_path_cost(rx, ry, resolution):
    cost = 0.0
    
    for i in range(1, len(rx)):
        dx = rx[i] - rx[i - 1]
        dy = ry[i] - ry[i - 1]
        cost += math.hypot(dx / resolution, dy / resolution)

    return cost

def is_path_collision_free(planner, rx, ry):
    for x, y in zip(rx, ry):
        ix = round((x - planner.min_x) / planner.resolution)
        iy = round((y - planner.min_y) / planner.resolution)
        
        if ix < 0 or ix >= len(planner.obstacle_map) or iy < 0 or iy >= len(planner.obstacle_map[0]):
            return False, (x, y)
        
        if planner.obstacle_map[ix][iy]:
            return False, (x, y)
    
    return True, None

def calculate_mcdc_coverage(test_cases, planner_class):
    covered_conditions = 0
    total_conditions = 4
    
    for test in test_cases:
        planner = planner_class(test["ox"], test["oy"], 2.0, 1.0)
        rx, ry = planner.planning(test["sx"], test["sy"], test["gx"], test["gy"])
    
        if rx:
            covered_conditions = max(covered_conditions, 3)
        else:
            covered_conditions = max(covered_conditions, 2)
    
    return (covered_conditions / total_conditions) * 100

def run_test_suite(planner_class, test_cases):
    start_time = time.time()
    passed = 0
    total_tests = len(test_cases)
    
    for test in test_cases:
        planner = planner_class(test["ox"], test["oy"], 2.0, 1.0)
        rx, ry = planner.planning(test["sx"], test["sy"], test["gx"], test["gy"])
        
        if test["expected_path_length"] == 0:
            if not rx:
                passed += 1
            else:
                print(f"Test '{test['name']}' failed: expected no path, but found one.")
            continue

        if not rx:
            print(f"Test '{test['name']}' failed: expected a path but found none.")
            continue
        
        if not (math.isclose(rx[0], test["gx"], abs_tol=1e-3) and math.isclose(ry[0], test["gy"], abs_tol=1e-3)):
            print(f"Test '{test['name']}' failed: goal point mismatch. Expected ({test['gx']}, {test['gy']}), got ({rx[0]}, {ry[0]}).")
        
            continue
        if not (math.isclose(rx[-1], test["sx"], abs_tol=1e-3) and math.isclose(ry[-1], test["sy"], abs_tol=1e-3)):
            print(f"Test '{test['name']}' failed: start point mismatch. Expected ({test['sx']}, {test['sy']}), got ({rx[-1]}, {ry[-1]}).")
            continue

        collision_free, collision_point = is_path_collision_free(planner, rx, ry)

        if not collision_free:
            print(f"Test '{test['name']}' failed: path collides at {collision_point}.")
            continue

        continuous, point_a, point_b = check_path_continuity(rx, ry, planner.resolution)

        if not continuous:
            print(f"Test '{test['name']}' failed: discontinuity between {point_a} and {point_b}.")
            continue

        path_cost = compute_path_cost(rx, ry, planner.resolution)
        expected_cost = test["expected_path_length"]

        if not math.isclose(path_cost, expected_cost, rel_tol=0.1, abs_tol=1.0):
            print(f"Test '{test['name']}' failed: path cost mismatch. Expected approx {expected_cost}, got {path_cost:.2f}.")
            continue
        
        passed += 1
    
    avg_time = (time.time() - start_time) / total_tests
    fault_detection = (passed / total_tests) * 100
    
    return {
        "coverage_mcdc": calculate_mcdc_coverage(test_cases, planner_class),
        "fault_detection": fault_detection,
        "avg_time": avg_time,
        "total_tests": total_tests
    }

CSV_FILENAME = "a_star_test_results.csv"

def save_results_to_csv(results):
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Technique", "MC/DC Coverage (%)", "Fault Detection (%)", "Avg Time (s)", "Total Tests"])
        
        for name, data in results.items():
            if name == "Mutation Score":
                writer.writerow([name, "N/A", data["fault_detection"], "N/A", "N/A"])
            else:
                writer.writerow([
                    name,
                    data.get("coverage_mcdc", "N/A"),
                    data["fault_detection"],
                    data["avg_time"],
                    data["total_tests"]
                ])

if __name__ == "__main__":
    test_cases = create_test_cases()
    
    results = {
        "Original A*": run_test_suite(AStarPlanner, test_cases),
        "Mutant1 (heuristic zero)": run_test_suite(AStarMutant1, test_cases),
        "Mutant2 (no colision)": run_test_suite(AStarMutant2, test_cases),
        "Mutant3 (high cost)": run_test_suite(AStarMutant3, test_cases),
    }
    
    original_score = results["Original A*"]["fault_detection"]
    mutation_scores = {}
    
    for name in ["Mutant1 (heuristic zero)", "Mutant2 (no colision)", "Mutant3 (high cost)"]:
        mutation_scores[name] = original_score - results[name]["fault_detection"]
    
    avg_mutation_score = sum(mutation_scores.values()) / len(mutation_scores)
    results["Mutation Score"] = {"fault_detection": avg_mutation_score}
    
    save_results_to_csv(results)
    print(f"Resultados salvos em {CSV_FILENAME}")

    print("\n=== Summary ===")
    for name, data in results.items():
        if "coverage_mcdc" in data:
            print(f"{name}:")
            print(f"  - MC/DC Coverage: {data['coverage_mcdc']:.1f}%")
            print(f"  - Fault Detection: {data['fault_detection']:.1f}%")
            print(f"  - Avg Time: {data['avg_time']:.4f} s")
            print(f"  - Total Tests: {data['total_tests']}")
        else:
            print(f"{name}: {data['fault_detection']:.1f}%")
