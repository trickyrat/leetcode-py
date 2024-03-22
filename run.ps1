./.venv/Scripts/activate.ps1

coverage run -m pytest .

coverage html -d ./test_results