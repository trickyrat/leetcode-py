import coverage
import pytest
import sys
import datetime
import os


def run_tests_with_coverage():
    cov = coverage.Coverage(
        source=['./src'],
        omit=[
            'test/*',
            '*/__pycache__/*',
            '*/venv/*',
            '*/env/*',
            '*/virtualenv/*'
        ],
        config_file=True
    )

    print("Starting to collect coverage data...")
    cov.start()

    try:
        print("Running pytest tests...")
        exit_code = pytest.main([
            './test',
            '-v',
            '--tb=short'
        ])

    except Exception as e:
        print(f"Encountered an error while running tests: {e}")
        exit_code = 1

    finally:
        print("Stopping to collect coverage data...")
        cov.stop()
        cov.save()

    return exit_code


def generate_coverage_reports():
    cov = coverage.Coverage()
    cov.load()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("coverage_reports", exist_ok=True)

    reports = {}

    xml_file = f"coverage_reports/coverage_{timestamp}.xml"
    cov.xml_report(outfile=xml_file)
    reports['xml'] = xml_file

    json_file = f"coverage_reports/coverage_{timestamp}.json"
    cov.json_report(outfile=json_file)
    reports['json'] = json_file

    lcov_file = f"coverage_reports/coverage_{timestamp}.lcov"
    cov.lcov_report(outfile=lcov_file)
    reports['lcov'] = lcov_file

    html_dir = f"coverage_reports/htmlcov_{timestamp}"
    cov.html_report(directory=html_dir)
    reports['html'] = html_dir

    print("\n" + "=" * 50)
    print("Summary of coverage:")
    cov.report()

    print("\nReport generated:")
    for format_name, path in reports.items():
        print(f"  {format_name.upper()}: {path}")

    return reports


if __name__ == "__main__":
    exit_code = run_tests_with_coverage()

    if exit_code == 0:
        generate_coverage_reports()
    else:
        print("Failed to test, Skip coverage report generation")

    sys.exit(exit_code)