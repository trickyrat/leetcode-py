echo "activating virtual environment"
. ./.venv/bin/activate
echo "deactivating virtual environment"

echo "running tests"
coverage run -m pytest .
echo "tests complete"

echo "generating HTML test results"
coverage html -d ./test_results/html
echo "test results complete"

current_timestamp=$(date +"%Y%m%d_%H%M%S")

echo "generating XML test results"
coverage xml -o ./test_results/coverage_"${current_timestamp}".xml
echo "test results complete"

echo "generating JSON test results"
coverage json -o ./test_results/coverage_"${current_timestamp}".json
echo "test results complete"

echo "generating LCOV test results"
coverage lcov -o ./test_results/coverage_"${current_timestamp}".lcov
echo "test results complete"

echo "deactivating virtual environment"
deactivate
echo "virtual environment deactivated"