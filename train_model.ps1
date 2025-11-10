# ================================
# train_model_v2.ps1 (PowerShell 5.1 compatible)
# .env 기반으로 goal/coherence 학습 실행
# 사용 예) .\train_model.ps1 -Mode all -BuildCoherenceIfMissing
# ================================

param(
  [ValidateSet('goal','coh','all')]
  [string]$Mode = 'goal',
  [switch]$BuildCoherenceIfMissing
)

# Util: Coalesce (첫 번째 유효 문자열 반환)
function Coalesce([string[]]$vals) {
  foreach ($v in $vals) {
    if ($null -ne $v -and $v.Trim().Length -gt 0) { return $v }
  }
  return $null
}

# 1) .env 자동 로드
function Load-DotEnv {
  param([string]$Path = ".env")
  if (-not (Test-Path $Path)) {
    Write-Warning "[WARN] .env not found. Using system env only."
    return
  }
  Write-Host "[INFO] Loading .env from $Path"
  Get-Content $Path | ForEach-Object {
    if ($_ -match '^\s*#' -or $_.Trim().Length -eq 0) { return }
    if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*)\s*$') {
      $name  = $matches[1].Trim()
      $value = $matches[2].Trim()
      if ($value.StartsWith('"') -and $value.EndsWith('"')) {
        $value = $value.Substring(1, $value.Length - 2)
      }
      [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
      Set-Item -Path ("Env:" + $name) -Value $value | Out-Null
    }
  }
  Write-Host "[INFO] .env loaded."
}

Load-DotEnv

# 2) 경로/디렉토리 유틸
function Ensure-Dir([string]$Dir) {
  if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Path $Dir | Out-Null }
}

function Test-CheckpointPresent([string]$Dir) {
  if (-not (Test-Path $Dir)) { return $false }
  $files = Get-ChildItem -Path $Dir -Recurse -File -ErrorAction SilentlyContinue
  return ($files.Count -gt 0)
}

# 3) 출력/데이터 경로
# Goal ckpt dir
$goalOut = Coalesce @($env:GOAL_CKPT_DIR, $env:GOAL_CKPT_DIR, "data/ckpt_goal")
Ensure-Dir $goalOut

# Coherence ckpt dir
$cohOut  = Coalesce @($env:COH_CKPT_DIR, $env:COH_CKPT_DIR, "data/ckpt_coherence")
Ensure-Dir $cohOut

# Datasets
$goalDs = Coalesce @($env:GOAL_DATASETS_DIR, "datasets")
$cohDs  = Coalesce @($env:COH_DATASETS_DIR,  "datasets/order_coherence")

# 4) 학습 함수
function Invoke-TrainGoal {
  $model = Coalesce @($env:MODEL_NAME_OR_PATH, $env:MODEL, "skt/kobert-base-v1")
  $epochs = Coalesce @($env:GOAL_EPOCHS, $env:EPOCHS, "5")
  $lr = Coalesce @($env:GOAL_LR, $env:LR, "2e-5")
  $maxlen = Coalesce @($env:GOAL_MAX_LEN, $env:MAX_LEN, "128")
  $wd = Coalesce @($env:GOAL_WEIGHT_DECAY, $env:WEIGHT_DECAY, "0.05")
  $warm = Coalesce @($env:GOAL_WARMUP_RATIO, $env:WARMUP_RATIO, "0.06")
  $fp16 = Coalesce @($env:FP16, "0")
  $gacc = Coalesce @($env:GRAD_ACCUM, "4")
  $drop = Coalesce @($env:GOAL_DROPOUT, $env:DROPOUT, "0.10")
  $ls   = Coalesce @($env:GOAL_LABEL_SMOOTH, $env:LABEL_SMOOTH, "0.05")

  Write-Host "----------------------------------------------------"
  Write-Host ("[GOAL] MODEL        = " + $model)
  Write-Host ("       EPOCHS       = " + $epochs)
  Write-Host ("       LR           = " + $lr)
  Write-Host ("       MAX_LEN      = " + $maxlen)
  Write-Host ("       WEIGHT_DECAY = " + $wd)
  Write-Host ("       WARMUP_RATIO = " + $warm)
  Write-Host ("       FP16         = " + $fp16)
  Write-Host ("       GRAD_ACCUM   = " + $gacc)
  Write-Host ("       DROPOUT      = " + $drop)
  Write-Host ("       LABEL_SMOOTH = " + $ls)
  Write-Host ("       OUT          = " + $goalOut)
  Write-Host ("       DATASETS_DIR = " + $goalDs)
  Write-Host "----------------------------------------------------"

  # 최소 필요 env 안전 설정
  $prevOut = $env:OUT;            $env:OUT = $goalOut
  $prevDs  = $env:DATASETS_DIR;   $env:DATASETS_DIR = $goalDs

  python app/train_goal.py
  $code = $LASTEXITCODE

  $env:OUT = $prevOut
  $env:DATASETS_DIR = $prevDs

  if ($code -ne 0) { Write-Host "[ERROR] train_goal.py failed ($code)"; exit $code }
}

function Invoke-TrainCoh {
  $model = Coalesce @($env:MODEL_NAME_OR_PATH, $env:MODEL, "skt/kobert-base-v1")
  $epochs = Coalesce @($env:COH_EPOCHS, $env:EPOCHS, "6")
  $lr = Coalesce @($env:COH_LR, $env:LR, "3e-5")
  $maxlen = Coalesce @($env:COH_MAX_LEN, $env:MAX_LEN, "384")
  $wd = Coalesce @($env:COH_WEIGHT_DECAY, $env:WEIGHT_DECAY, "0.05")
  $warm = Coalesce @($env:COH_WARMUP_RATIO, $env:WARMUP_RATIO, "0.10")
  $fp16 = Coalesce @($env:FP16, "0")
  $gacc = Coalesce @($env:GRAD_ACCUM, "4")
  $drop = Coalesce @($env:COH_DROPOUT, $env:DROPOUT, "0.10")
  $ls   = Coalesce @($env:COH_LABEL_SMOOTH, $env:LABEL_SMOOTH, "0.10")

  Write-Host "----------------------------------------------------"
  Write-Host ("[COH ] MODEL        = " + $model)
  Write-Host ("       EPOCHS       = " + $epochs)
  Write-Host ("       LR           = " + $lr)
  Write-Host ("       MAX_LEN      = " + $maxlen)
  Write-Host ("       WEIGHT_DECAY = " + $wd)
  Write-Host ("       WARMUP_RATIO = " + $warm)
  Write-Host ("       FP16         = " + $fp16)
  Write-Host ("       GRAD_ACCUM   = " + $gacc)
  Write-Host ("       DROPOUT      = " + $drop)
  Write-Host ("       LABEL_SMOOTH = " + $ls)
  Write-Host ("       OUT          = " + $cohOut)
  Write-Host ("       DATASETS_DIR = " + $cohDs)
  Write-Host "----------------------------------------------------"

  $prevOut = $env:OUT;            $env:OUT = $cohOut
  $prevDs  = $env:DATASETS_DIR;   $env:DATASETS_DIR = $cohDs

  python app/train_coherence_v2.py
  $code = $LASTEXITCODE

  $env:OUT = $prevOut
  $env:DATASETS_DIR = $prevDs

  if ($code -ne 0) { Write-Host "[ERROR] train_coherence_v2.py failed ($code)"; exit $code }
}

# 5) 실행 플로우
switch ($Mode) {
  'goal' {
    Invoke-TrainGoal
    if ($BuildCoherenceIfMissing) {
      if (-not (Test-CheckpointPresent -Dir $cohOut)) {
        Write-Host "[INFO] Coherence checkpoint missing → building it now."
        Invoke-TrainCoh
      } else {
        Write-Host "[INFO] Coherence checkpoint found at '$cohOut' → skipping."
      }
    }
  }
  'coh'  { Invoke-TrainCoh }
  'all'  { Invoke-TrainGoal; Invoke-TrainCoh }
}

Write-Host "[DONE] Mode '$Mode' finished."
