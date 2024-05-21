echo "activating virtual environment"
. ./.venv/bin/activate
echo "deactivating virtual environment"

echo "running tests"
coverage run -m pytest .
echo "tests complete"

echo "generating test results"
coverage html -d ./test_results
echo "test results complete"

echo "deactivating virtual environment"
deactivate
echo "virtual environment deactivated"