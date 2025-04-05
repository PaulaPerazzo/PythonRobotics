import time
import csv
import io
import coverage
import pytest

# Custom plugin to capture test durations and failures.
class MetricsPlugin:
    def __init__(self):
        self.total_duration = 0  # Sum of durations of "call" phase
        self.failures = 0        # Number of failures

    def pytest_runtest_logreport(self, report):
        # Only capture the "call" phase (i.e. actual test execution).
        if report.when == "call":
            self.total_duration += report.duration
            if report.failed:
                self.failures += 1

# Function to run an individual test using pytest and coverage.
def run_single_test(test_name):
    plugin = MetricsPlugin()
    # Start coverage (set source to your A* module's location).
    cov = coverage.Coverage(branch=True, source=["PathPlanning/AStar"])
    cov.start()

    start_time = time.time()
    exit_code = pytest.main([test_name], plugins=[plugin])
    total_time = time.time() - start_time

    cov.stop()
    cov.save()

    # Capture the coverage report output.
    report_output = io.StringIO()
    cov.report(file=report_output)
    report_str = report_output.getvalue()
    print(report_str)
    try:
        # Assume the overall branch coverage percentage is the last token in the report.
        coverage_percent = float(report_str.strip().split()[-1].replace('%', ''))
    except Exception:
        coverage_percent = 0.0

    return {
        "Test": test_name,
        "Overall Execution Time (s)": round(total_time, 6),
        "Total Test Duration (s)": round(plugin.total_duration, 6),
        "Failures": plugin.failures,
        "Coverage (%)": round(coverage_percent, 2),
        "Exit Code": exit_code,
        "Full report": report_str
    }

def main():
    # List the individual test identifiers from your property_testing.py file.
    test_names = [
        "property_testing.py",
        "combinatorial_testing.py",
        "data_flow_testing.py",
        "unit_testing.py",
    ]

    results = []
    for test in test_names:
        print("Running", test)
        result = run_single_test(test)
        results.append(result)

    csv_file = "property_test_metrics_mutants.csv"
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "Test",
            "Overall Execution Time (s)",
            "Total Test Duration (s)",
            "Failures",
            "Coverage (%)",
            "Exit Code",
            "Full report"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)

    print("Metrics collected and saved to", csv_file)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
