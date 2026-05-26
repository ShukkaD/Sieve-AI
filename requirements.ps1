# requirements.ps1

$ErrorActionPreference = "Stop"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing ultralytics..."
python -m pip install -U ultralytics

Write-Host "Installing boxmot..."
python -m pip install -U boxmot

Write-Host "Installing rtmlib..."
python -m pip install -U rtmlib -i https://pypi.org/simple

Write-Host "Installing TensorRT (CUDA 13)..."
python -m pip install --upgrade tensorrt-cu13

Write-Host "Upgrading all packages..."
python -m pip freeze > r.txt
((Get-Content r.txt) -replace '==', '>=') | Set-Content r.txt
python -m pip install --upgrade -r r.txt
Remove-Item r.txt

Write-Host "Uninstalling torch, torchvision, onnxruntime..."
python -m pip uninstall -y torch torchvision onnxruntime

Write-Host "Installing PyTorch (CUDA 13 index)..."
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu132

Write-Host "Installing onnxruntime-gpu..."
python -m pip install -U coloredlogs flatbuffers numpy packaging protobuf sympy
python -m pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ --upgrade onnxruntime-gpu

Write-Host "Finished installing requirements"