Write-Host "Activating virtual environment..."
./.venv/Scripts/activate.ps1
Write-Host "Virtual environment activated."

Write-Host "Running tests..."
coverage run -m pytest .
Write-Host  "Tests completed."

Write-Host "Generating test results..."
coverage html -d ./test_results
Write-Host "Test results generated."

Write-Host "Deactivating virtual environment..."
./.venv/Scripts/deactivate.ps1
Write-Host "Virtual environment deactivated."