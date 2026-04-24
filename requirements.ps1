# requirements.ps1

$ErrorActionPreference = "Stop"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing ultralytics..."
python -m pip install ultralytics

Write-Host "Installing boxmot..."
python -m pip install boxmot

Write-Host "Installing rtmlib..."
python -m pip install rtmlib -i https://pypi.org/simple

Write-Host "Upgrading TensorRT (cu12)..."
python -m pip install --upgrade tensorrt-cu12

Write-Host "Upgrading all packages..."
python -m pip freeze > r.txt
((Get-Content r.txt) -replace '==', '>=') | Set-Content r.txt
python -m pip install --upgrade -r r.txt
Remove-Item r.txt

Write-Host "Uninstalling torch, torchvision, onnxruntime..."
python -m pip uninstall -y torch torchvision onnxruntime

Write-Host "Installing PyTorch (CUDA 12.8 index)..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing onnxruntime-gpu..."
python -m pip install onnxruntime-gpu

Write-Host "Done."