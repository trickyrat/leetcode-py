. ./.venv/bin/activate

coverage run -m pytest .

coverage html -d ./test_results

deactivate