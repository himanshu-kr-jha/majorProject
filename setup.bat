@echo off
setlocal EnableDelayedExpansion
:: =============================================================================
:: setup.bat — Gait-YOLO: one-command environment setup + dataset download + training
::
:: Usage:
::   setup.bat                  :: full setup + download + train
::   setup.bat --skip-train     :: setup + download only
::   setup.bat --skip-download  :: skip downloads (datasets already present)
::
:: Requires : Python 3.10+, curl (Windows 10 1803+), tar (Windows 10+)
:: Optional : nvidia-smi (auto-detected for GPU PyTorch install)
::
:: Kaggle auth: reads %USERPROFILE%\.kaggle\kaggle.json automatically.
:: If missing, you will be prompted for credentials.
:: Get your token: https://www.kaggle.com/settings -> API -> Create New Token
::
:: UCF-Crime slug: defaults to odins0n/ucf-crime-dataset
:: Override:  set UCF_KAGGLE_SLUG=owner/dataset-name && setup.bat
:: =============================================================================

:: ── Parse flags ───────────────────────────────────────────────────────────────
set SKIP_TRAIN=0
set SKIP_DOWNLOAD=0
for %%A in (%*) do (
  if "%%A"=="--skip-train"    set SKIP_TRAIN=1
  if "%%A"=="--skip-download" set SKIP_DOWNLOAD=1
)

:: ── Log setup ─────────────────────────────────────────────────────────────────
if not exist logs mkdir logs
if not exist results mkdir results
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
set LOG=logs\setup_%DT:~0,8%_%DT:~8,6%.log

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║          Gait-YOLO — One-Command Setup               ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo [INFO]  Full log: %LOG%

:: =============================================================================
:: 1. Prerequisites
:: =============================================================================
echo.
echo ━━━  1/8  Checking prerequisites  ━━━

where python >nul 2>&1
if errorlevel 1 (
  echo [WARN]  python not found — install from https://python.org
) else (
  echo [ OK ]  python found
)

where pip >nul 2>&1
if errorlevel 1 (
  echo [WARN]  pip not found — run: python -m ensurepip --upgrade
) else (
  echo [ OK ]  pip found
)

where curl >nul 2>&1
if errorlevel 1 (
  echo [WARN]  curl not found — upgrade to Windows 10 1803+
) else (
  echo [ OK ]  curl found
)

where tar >nul 2>&1
if errorlevel 1 (
  echo [WARN]  tar not found — upgrade to Windows 10+
) else (
  echo [ OK ]  tar found
)

:: Check Python version
for /f "tokens=*" %%V in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set PY_VER=%%V
echo [ OK ]  Python %PY_VER%

:: Check GPU
set GPU_AVAILABLE=0
set CUDA_VER=0

:: Try direct path first (most reliable)
set NVSMI_PATH="C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"

if exist %NVSMI_PATH% (
    echo [INFO]  Using NVIDIA-SMI from default path

    for /f "tokens=*" %%G in ('%NVSMI_PATH% --query-gpu^=name --format^=csv^,noheader 2^>nul') do set GPU_NAME=%%G
    for /f "tokens=3" %%C in ('%NVSMI_PATH% 2^>nul ^| findstr /C:"CUDA Version"') do set CUDA_VER=%%C

    echo [ OK ]  GPU: !GPU_NAME!  (CUDA !CUDA_VER!)
    set GPU_AVAILABLE=1

) else (

    :: fallback to PATH
    where nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo [INFO]  Using NVIDIA-SMI from PATH

        for /f "tokens=*" %%G in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do set GPU_NAME=%%G
        for /f "tokens=3" %%C in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do set CUDA_VER=%%C

        echo [ OK ]  GPU: !GPU_NAME!  (CUDA !CUDA_VER!)
        set GPU_AVAILABLE=1

    ) else (
        echo [WARN]  nvidia-smi not found — CPU-only PyTorch will be installed
    )
)

echo.>> %LOG% 2>&1
echo Prerequisites checked >> %LOG% 2>&1

:: =============================================================================
:: 2. Virtual environment
:: =============================================================================
echo.
echo ━━━  2/8  Python virtual environment  ━━━

if not exist venv (
  echo [INFO]  Creating venv\ ...
  python -m venv venv >> %LOG% 2>&1
  echo [ OK ]  venv created
) else (
  echo [ OK ]  venv\ exists — reusing
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel --quiet >> %LOG% 2>&1
echo [ OK ]  pip/setuptools upgraded

:: =============================================================================
:: 3. PyTorch (CUDA-aware)
:: =============================================================================
echo.
echo ━━━  3/8  Installing PyTorch  ━━━

python -c "import torch" >nul 2>&1
if not errorlevel 1 (
  for /f "tokens=*" %%T in ('python -c "import torch; print(torch.__version__)"') do echo [ OK ]  PyTorch %%T already installed
  goto :pytorch_done
)

:: Parse CUDA major version
for /f "tokens=1 delims=." %%M in ("!CUDA_VER!") do set CUDA_MAJOR=%%M
set /a CUDA_MAJOR_INT=!CUDA_MAJOR! 2>nul

if "!GPU_AVAILABLE!"=="1" (
  if !CUDA_MAJOR_INT! GEQ 12 (
    echo [INFO]  CUDA 12.x detected — installing torch+cu121 ...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet >> %LOG% 2>&1
  ) else if !CUDA_MAJOR_INT! EQU 11 (
    echo [INFO]  CUDA 11.x detected — installing torch+cu118 ...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet >> %LOG% 2>&1
  ) else (
    echo [INFO]  Unknown CUDA — installing CPU torch ...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet >> %LOG% 2>&1
  )
) else (
  echo [INFO]  No GPU — installing CPU torch ...
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet >> %LOG% 2>&1
)
echo [ OK ]  PyTorch installed

:pytorch_done
python -c "import torch; print('  version :', torch.__version__); print('  cuda    :', torch.cuda.is_available()); t=torch.cuda.is_available() and print('  gpu     :', torch.cuda.get_device_name(0))"

:: =============================================================================
:: 4. Project dependencies
:: =============================================================================
echo.
echo ━━━  4/8  Installing project dependencies  ━━━
pip install -r requirements.txt --quiet >> %LOG% 2>&1
echo [ OK ]  requirements.txt installed

:: =============================================================================
:: 5. Kaggle credentials
:: =============================================================================
echo.
echo ━━━  5/8  Kaggle API credentials  ━━━

set KAGGLE_JSON=%USERPROFILE%\.kaggle\kaggle.json
if exist "%KAGGLE_JSON%" (
  for /f "tokens=*" %%U in ('python -c "import json; d=json.load(open(r'%KAGGLE_JSON%')); print(d['username'])"') do set KAGGLE_USERNAME=%%U
  for /f "tokens=*" %%K in ('python -c "import json; d=json.load(open(r'%KAGGLE_JSON%')); print(d['key'])"') do set KAGGLE_KEY=%%K
  echo [ OK ]  Loaded from %KAGGLE_JSON% (user: !KAGGLE_USERNAME!)
) else (
  echo [WARN]  %KAGGLE_JSON% not found
  echo         Get token at: https://www.kaggle.com/settings -^> API -^> Create New Token
  set /p KAGGLE_USERNAME=  Kaggle username:
  set /p KAGGLE_KEY=  Kaggle API key:
  if not exist "%USERPROFILE%\.kaggle" mkdir "%USERPROFILE%\.kaggle"
  python -c "import json; f=open(r'%KAGGLE_JSON%','w'); json.dump({'username':'!KAGGLE_USERNAME!','key':'!KAGGLE_KEY!'},f); f.close()"
  echo [ OK ]  Credentials saved to %KAGGLE_JSON%
)

:: Validate credentials
for /f "tokens=*" %%H in ('curl -s -o nul -w "%%{http_code}" -u "!KAGGLE_USERNAME!:!KAGGLE_KEY!" "https://www.kaggle.com/api/v1/datasets/list?search=test&page=1" 2^>nul') do set HTTP_CODE=%%H
if "!HTTP_CODE!"=="200" (
  echo [ OK ]  Kaggle credentials valid
) else (
  echo [WARN]  Kaggle API returned HTTP !HTTP_CODE! — credentials may be wrong
)

:: =============================================================================
:: 6. Dataset downloads
:: =============================================================================
echo.
echo ━━━  6/8  Downloading datasets  ━━━

if "!SKIP_DOWNLOAD!"=="0" (
  if not exist datasets\tmp mkdir datasets\tmp

  :: Helper macro via goto
  goto :skip_dl_def

  :kaggle_dl
    :: %~1=NAME  %~2=SLUG  %~3=DEST  %~4=MARKER
    if exist "%~4" (
      echo [ OK ]  %~1 already present — skipping
      goto :eof
    )
    set DL_ZIP=datasets\tmp\%~1.zip
    echo [INFO]  Downloading %~1 (slug: %~2) ...
    curl -L --progress-bar -u "!KAGGLE_USERNAME!:!KAGGLE_KEY!" -o "!DL_ZIP!" "https://www.kaggle.com/api/v1/datasets/download/%~2"
    if not errorlevel 1 (
      echo [INFO]  Extracting to %~3 ...
      if not exist "%~3" mkdir "%~3"
      tar -xf "!DL_ZIP!" -C "%~3"
      del /f /q "!DL_ZIP!" >nul 2>&1
      echo [ OK ]  %~1 extracted
    ) else (
      echo [WARN]  Download failed for %~1 — check slug or credentials. Skipping.
      del /f /q "!DL_ZIP!" >nul 2>&1
    )
    goto :eof

  :skip_dl_def

  call :kaggle_dl "guns-knives" "kruthisb999/guns-and-knifes-detection-in-cctv-videos" "datasets\guns-knives" "datasets\guns-knives\combined_gunsnknifes\data.yaml"

  call :kaggle_dl "casia-b" "trnquanghuyn/casia-b" "datasets\casia-b" "datasets\casia-b\output"

  if not defined UCF_KAGGLE_SLUG set UCF_KAGGLE_SLUG=odins0n/ucf-crime-dataset
  echo [INFO]  UCF-Crime slug: !UCF_KAGGLE_SLUG!
  echo [INFO]  (Override: set UCF_KAGGLE_SLUG=owner/slug ^&^& setup.bat)
  call :kaggle_dl "ucf-crime" "!UCF_KAGGLE_SLUG!" "datasets\anomalydetectiondatasetucf" "datasets\anomalydetectiondatasetucf\Anomaly_Test.txt"

  echo [ OK ]  Dataset downloads complete
) else (
  echo [INFO]  Skipping downloads (--skip-download flag)
)

:: =============================================================================
:: 7. Training pipeline
:: =============================================================================
echo.
echo ━━━  7/8  Training pipeline  ━━━

if "!SKIP_TRAIN!"=="0" (

  :: Step A: VideoMAE HF eval
  echo [INFO]  Running: python scripts\run_videomae_hf_eval.py
  python scripts\run_videomae_hf_eval.py >> logs\videomae_hf_eval.log 2>&1
  if not errorlevel 1 (echo [ OK ]  videomae_hf_eval done) else (echo [WARN]  videomae_hf_eval had errors — check logs\videomae_hf_eval.log)

  :: Step B: retrain gait autoencoder (normal-only)
  echo [INFO]  Running: python scripts\train_gait.py
  python scripts\train_gait.py >> logs\train_gait.log 2>&1
  if not errorlevel 1 (echo [ OK ]  train_gait done) else (echo [WARN]  train_gait had errors — check logs\train_gait.log)

  :: Step C: fine-tune YOLO with yolov8s
  echo [INFO]  Running: python scripts\train_yolo.py
  python scripts\train_yolo.py >> logs\train_yolo.log 2>&1
  if not errorlevel 1 (echo [ OK ]  train_yolo done) else (echo [WARN]  train_yolo had errors — check logs\train_yolo.log)

  :: =============================================================================
  :: 8. Evaluation pipeline
  :: =============================================================================
  echo.
  echo ━━━  8/8  Evaluation pipeline  ━━━

  :: Patch eval scripts to new checkpoint names if produced
  if exist "models\casib-b\best_gait_v2.pth" (
    python -c "import re,pathlib; p=pathlib.Path('scripts/run_gait_eval.py'); p.write_text(re.sub(r'best_transformer_gait\.pth','best_gait_v2.pth',p.read_text()))"
    echo [ OK ]  run_gait_eval.py patched to best_gait_v2.pth
  )
  if exist "models\knifes&pistol\best_v2.pt" (
    python -c "import re,pathlib; p=pathlib.Path('scripts/run_yolo_eval.py'); p.write_text(re.sub(r'\bbest\.pt\b','best_v2.pt',p.read_text()))"
    echo [ OK ]  run_yolo_eval.py patched to best_v2.pt
  )

  echo [INFO]  Running: python scripts\run_yolo_eval.py
  python scripts\run_yolo_eval.py >> logs\eval_yolo.log 2>&1
  if not errorlevel 1 (echo [ OK ]  eval_yolo done) else (echo [WARN]  eval_yolo had errors — check logs\eval_yolo.log)

  echo [INFO]  Running: python scripts\run_gait_eval.py
  python scripts\run_gait_eval.py >> logs\eval_gait.log 2>&1
  if not errorlevel 1 (echo [ OK ]  eval_gait done) else (echo [WARN]  eval_gait had errors — check logs\eval_gait.log)

  echo [INFO]  Running: python scripts\run_videomae_hf_eval.py
  python scripts\run_videomae_hf_eval.py >> logs\eval_videomae.log 2>&1
  if not errorlevel 1 (echo [ OK ]  eval_videomae done) else (echo [WARN]  eval_videomae had errors — check logs\eval_videomae.log)

  echo [INFO]  Running: python scripts\run_full_real_eval.py
  python scripts\run_full_real_eval.py >> logs\eval_fusion.log 2>&1
  if not errorlevel 1 (echo [ OK ]  eval_fusion done) else (echo [WARN]  eval_fusion had errors — check logs\eval_fusion.log)

  echo [INFO]  Running: python src\experiments\ablation.py --real
  python src\experiments\ablation.py --real >> logs\ablation.log 2>&1
  if not errorlevel 1 (echo [ OK ]  ablation done) else (echo [WARN]  ablation had errors — check logs\ablation.log)

  :: Print results summary
  echo.
  echo [INFO]  Results summary:
  python -c "import json,glob,os; [print(f'  {os.path.basename(jf):<42} F1={d.get(\"f1\",d.get(\"fusion_f1\",\"—\")):.4f}' if isinstance(d.get('f1',d.get('fusion_f1')),float) else f'  {os.path.basename(jf):<42} F1={d.get(\"f1\",d.get(\"fusion_f1\",chr(8212)))}') for jf in sorted(glob.glob('results/*.json')) for d in [json.load(open(jf))] if True]" 2>nul

) else (
  echo [INFO]  Skipping training (--skip-train)
)

:: =============================================================================
echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║               Setup complete!                        ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo   Log        : %LOG%
echo   Results    : results\
echo   Checkpoints: models\casib-b\best_gait_v2.pth
echo                models\knifes^&pistol\best_v2.pt
echo.
echo   To activate the environment later:
echo     venv\Scripts\activate.bat
echo.

endlocal
