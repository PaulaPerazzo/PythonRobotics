import itertools
import time
import csv
import math
import matplotlib.pyplot as plt
from PathPlanning.AStar.a_star import AStarPlanner
import coverage
import io

csv_file = "a_star_CT_test_results.csv"

# function to generate full-factorial test
def full_factorial(params):
    keys = list(params.keys())
    all_combos = list(itertools.product(*(params[k] for k in keys)))

    return [dict(zip(keys, combo)) for combo in all_combos]

# function to generate a covering array for t-way interactions
def generate_covering_array(params, t):
    keys = list(params.keys())
    required = set()

    for key_combo in itertools.combinations(keys, t): # generate all t-way combinations
        for vals in itertools.product(*(params[k] for k in key_combo)):
            required.add(tuple(sorted(zip(key_combo, vals))))
    
    full_tests = full_factorial(params)

    def covered(test):
        covers = set()

        for key_combo in itertools.combinations(keys, t):
            covers.add(tuple(sorted((k, test[k]) for k in key_combo)))

        return covers

    test_coverage = [(test, covered(test)) for test in full_tests]
    selected = []
    covered_required = set()
    
    # greedy algorithm to select tests 
    while required - covered_required:
        best_test, best_cov = max(test_coverage, key=lambda x: len(x[1] - covered_required))
        selected.append(best_test)
        covered_required |= best_cov

        test_coverage = [item for item in test_coverage if item[0] != best_test]
    
    return selected

def run_tests_with_coverage(test_suite):
    cov = coverage.Coverage(branch=True)
    cov.start()
    avg_time, total_tests, fault_det = run_tests(test_suite)
    cov.stop()
    cov.save()
    report_output = io.StringIO()
    cov.report(file=report_output)
    report_str = report_output.getvalue()
    
    try:
        total_line = report_str.strip().splitlines()[-1]
        coverage_percent = float(total_line.split()[-1].replace('%',''))
    
    except Exception:
        coverage_percent = 0.0
    
    return avg_time, total_tests, fault_det, coverage_percent

def create_obstacles():
    ox, oy = [], []
    
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
    for i in range(-20, 50):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 50):
        ox.append(40.0)
        oy.append(80.0 - i)
    
    return ox, oy

def run_tests(test_suite):
    ox, oy = create_obstacles()
    times = []
    fault_count = 0

    for test in test_suite:
        grid_size = test["grid_size"]
        robot_radius = test["robot_radius"]
        sx = test["start_x"]
        sy = test["start_y"]
        gx = test["goal_x"]
        gy = test["goal_y"]
        planner = AStarPlanner(ox, oy, grid_size, robot_radius)
        start_time = time.perf_counter()
        rx, ry = planner.planning(sx, sy, gx, gy)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        if not rx or not ry:
            fault_count += 1

    avg_time = sum(times) / len(times) if times else 0
    fault_detection = (fault_count / len(test_suite)) * 100  # em %

    return avg_time, len(test_suite), fault_detection

params = {
    "grid_size": [1.0, 2.0],
    "robot_radius": [0.5, 1.0],
    "start_x": [10.0, 15.0],
    "start_y": [10.0, 15.0],
    "goal_x": [50.0, 51.0],
    "goal_y": [50.0, 55.0]
}

techniques = {
    "CT 2-way": generate_covering_array(params, 2),
    "CT 3-way": generate_covering_array(params, 3),
    "CT 4-way": generate_covering_array(params, 4),
    "full-factorial": full_factorial(params)
}

results = []

for tech_name, suite in techniques.items():
    avg_time, total_tests, fault_det, mc_dc = run_tests_with_coverage(suite)

    results.append({
        "Technique": tech_name,
        "MC/DC Coverage (%)": round(mc_dc, 2),
        "Fault Detection (%)": round(fault_det, 2),
        "Avg Time (s)": round(avg_time, 6),
        "Total Tests": total_tests
    })

with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Technique", "MC/DC Coverage (%)", "Fault Detection (%)", "Avg Time (s)", "Total Tests"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("Métricas coletadas e salvas em", csv_file)

for r in results:
    print(r)
