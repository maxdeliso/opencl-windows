# vcpkg Setup Script for OpenCL Windows Project
# This script automates the one-time vcpkg setup

param(
    [string]$VcpkgPath = "C:\dev\vcpkg"
)

Write-Host "=== vcpkg Setup Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if vcpkg already exists
if (Test-Path "$VcpkgPath\vcpkg.exe") {
    Write-Host "✓ vcpkg found at: $VcpkgPath" -ForegroundColor Green
    $vcpkgExe = "$VcpkgPath\vcpkg.exe"
} else {
    Write-Host "vcpkg not found at: $VcpkgPath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install vcpkg first:" -ForegroundColor Yellow
    Write-Host "  1. git clone https://github.com/Microsoft/vcpkg.git $VcpkgPath" -ForegroundColor Gray
    Write-Host "  2. cd $VcpkgPath" -ForegroundColor Gray
    Write-Host "  3. .\bootstrap-vcpkg.bat" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Or provide the path to existing vcpkg:" -ForegroundColor Yellow
    Write-Host "  .\setup-vcpkg.ps1 -VcpkgPath C:\path\to\vcpkg" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "=== Integrating vcpkg with Visual Studio ===" -ForegroundColor Cyan
& $vcpkgExe integrate install
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ vcpkg integrated with Visual Studio" -ForegroundColor Green
} else {
    Write-Host "Warning: vcpkg integrate install returned error code $LASTEXITCODE" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Installing Project Dependencies ===" -ForegroundColor Cyan

# Check if vcpkg.json exists
if (-not (Test-Path "vcpkg.json")) {
    Write-Host "Error: vcpkg.json not found in current directory" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "Installing dependencies from vcpkg.json..." -ForegroundColor Cyan
& $vcpkgExe install
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "Warning: vcpkg install returned error code $LASTEXITCODE" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open main.sln in Visual Studio" -ForegroundColor White
Write-Host "  2. Build the solution (F7)" -ForegroundColor White
Write-Host "  3. vcpkg dependencies will be automatically resolved" -ForegroundColor White
Write-Host ""
Write-Host "For more information, see VCPKG_SETUP.md" -ForegroundColor Gray
