import pytest

def pytest_html_results_table_header(cells):
    cells.insert(3, "<th>Metrics Values</th>")

def pytest_html_results_table_row(report, cells):
    cells.insert(3, f"<td>{getattr(report, 'metric_val', '')}</td>")

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        report.metric_val = getattr(item, "metric_val", "")
