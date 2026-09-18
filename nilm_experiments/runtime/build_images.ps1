param([switch]$VerifyGpu)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Convert-ReportToUtf8([string]$Path) {
    $report = Get-Content -Raw $Path
    $null = $report | ConvertFrom-Json
    [System.IO.File]::WriteAllText((Join-Path $PWD.Path $Path), $report, [System.Text.UTF8Encoding]::new($false))
}
if (Test-Path runtime/requirements-lock.txt) {
    docker build -t nilm-runtime:torch2.5.1-cu118-v1 -f runtime/Dockerfile.offline .
} else {
    docker build -t nilm-runtime:torch2.5.1-cu118-v1 -f runtime/Dockerfile .
}
if ($LASTEXITCODE -ne 0) { throw 'Base image build failed' }
$tags = @('nilm-runtime:torch2.5.1-cu118-v1')
foreach ($model in @('NILMFormer','FCN','BERT4NILM','SGN')) {
    $modelDir = @{NILMFormer='third_party/nilmformer'; FCN='FCN'; BERT4NILM='third_party/bert4nilm'; SGN='third_party/nilmtk_contrib'}[$model]
    $tag = 'nilm-' + $model.ToLower() + ':torch2.5.1-cu118-v1'
    docker build -t $tag -f "$modelDir/runtime/Dockerfile" .
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $model" }
    New-Item -ItemType Directory -Force "$modelDir/runtime/verification" | Out-Null
    docker run --rm --mount "type=bind,source=$($PWD.Path),target=/workspace/nilm_experiments" $tag | Tee-Object "$modelDir/runtime/verification/docker-cpu.json"
    if ($LASTEXITCODE -ne 0) { throw "Runtime check failed: $model" }
    Convert-ReportToUtf8 "$modelDir/runtime/verification/docker-cpu.json"
    if ($VerifyGpu) {
        docker run --rm --gpus all --mount "type=bind,source=$($PWD.Path),target=/workspace/nilm_experiments" $tag python runtime/verify_runtime.py --model $model --device cuda | Tee-Object "$modelDir/runtime/verification/docker-gpu.json"
        if ($LASTEXITCODE -ne 0) { throw "GPU runtime check failed: $model" }
        Convert-ReportToUtf8 "$modelDir/runtime/verification/docker-gpu.json"
    }
    $tags += $tag
}
New-Item -ItemType Directory -Force runtime/images | Out-Null
if (Test-Path runtime/images/nilm-cu118-v1.tar) { throw 'Archive already exists; choose a new output name instead of overwriting' }
docker save -o runtime/images/nilm-cu118-v1.tar @tags
if ($LASTEXITCODE -ne 0) { throw 'Image export failed' }
Get-FileHash runtime/images/nilm-cu118-v1.tar -Algorithm SHA256 | Format-List
