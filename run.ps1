Write-Host "Activating virtual environment..."
./.venv/Scripts/activate.ps1
Write-Host "Virtual environment activated."

Write-Host "Running tests..."
coverage run -m pytest .
Write-Host  "Tests completed."

Write-Host "Generating HTML test results..."
coverage html -d .\test_results\html
Write-Host "Test results generated."

$current_timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "Generating XML test results..."
coverage xml -o .\test_results\coverage_$current_timestamp.xml
Write-Host "Test results generated."

Write-Host "Generating JSON test results..."
coverage json -o .\test_results\coverage_$current_timestamp.json
Write-Host "Test results generated."

Write-Host "Generating LCOV test results..."
coverage lcov -o .\test_results\coverage_$current_timestamp.lcov
Write-Host "Test results generated."

Write-Host "Deactivating virtual environment..."
./.venv/Scripts/deactivate.bat
Write-Host "Virtual environment deactivated."