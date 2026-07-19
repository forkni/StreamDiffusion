"""
StreamDiffusion Nsight Compute (ncu) profiling launcher.

Captures per-kernel performance metrics from StreamDiffusion's TRT inference path.

── Quick start ─────────────────────────────────────────────────────────────────
Production engine (recommended — reuses td_config.yaml cached engine):
    venv/Scripts/python scripts/profiling/profile_ncu.py --config StreamDiffusionTD/td_config.yaml --set roofline --kernel-regex "sm89_xmma_gemm_e4m3" --launch-count 50

Basic metrics (2-3× overhead), first 50 kernels:
    venv/Scripts/python scripts/profiling/profile_ncu.py --config StreamDiffusionTD/td_config.yaml --set basic

Full metrics (20-50× overhead):
    venv/Scripts/python scripts/profiling/profile_ncu.py --config StreamDiffusionTD/td_config.yaml --set full

Fallback (single.py, non-production engine):
    venv/Scripts/python scripts/profiling/profile_ncu.py --target benchmark --set roofline

── Output ──────────────────────────────────────────────────────────────────────
  logs/ncu_<target>_<set>_<TS>.ncu-rep   — open in Nsight Compute UI
  logs/ncu_<target>_<set>_<TS>.csv       — (with --csv) kernel summary table

── Overhead factors ────────────────────────────────────────────────────────────
  basic:          2-3×   (arithmetic throughput, memory bandwidth, occupancy)
  full:          20-50×  (all hardware counters, multi-pass)
  roofline:       3-5×   (achievable roofline — adds SM throughput counters)
  memoryworkload: 5-10×  (L1/L2/DRAM access patterns)
  source:        10-30×  (source-level annotation, needs -lineinfo in compilation)

── Notes ────────────────────────────────────────────────────────────────────────
  - ncu attaches to the target process; TRT CUDA graphs must be disabled.
    Set STREAMDIFFUSION_PROFILE_TRT=1 — this disables CUDA graphs automatically
    (TRTProfiler IProfiler hooks are incompatible with graph replay).
  - --launch-skip N: skip the first N kernel launches (skip warmup/compilation).
  - --launch-count N: capture the next N launches after the skip.
  - Use --dry-run to see the exact ncu command without executing.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="StreamDiffusion Nsight Compute launcher")
parser.add_argument(
    "--target",
    default="benchmark",
    choices=["benchmark", "infer"],
    help="Target script: benchmark (single.py, 1 iter) or infer (minimal 1-frame driver)",
)
parser.add_argument(
    "--set",
    dest="metric_set",
    default="basic",
    choices=["basic", "full", "roofline", "memoryworkload", "source"],
    help="ncu metric preset (default: basic)",
)
parser.add_argument(
    "--kernel-regex",
    default="",
    help="Filter captured kernels by name regex (empty = all kernels)",
)
parser.add_argument(
    "--launch-skip",
    type=int,
    default=0,
    help="Skip first N kernel launches (default: 0)",
)
parser.add_argument(
    "--launch-count",
    type=int,
    default=50,
    help="Capture N kernel launches after skip (default: 50)",
)
parser.add_argument(
    "--csv",
    action="store_true",
    help="After capture, export details as CSV to logs/",
)
parser.add_argument(
    "--config",
    default="",
    metavar="PATH",
    help="YAML/JSON config file (e.g. StreamDiffusionTD/td_config.yaml). "
    "When provided, targets profile_nsys.py --target benchmark --config <path> "
    "instead of single.py — reuses the cached production engine.",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the ncu command without executing",
)
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
_PYTHON = sys.executable

os.makedirs(_LOGS_DIR, exist_ok=True)

# ── Locate ncu ─────────────────────────────────────────────────────────────────
# NOTE: We resolve the raw .exe path to bypass the ncu.bat shim, which
# interprets '|' in --kernel-name as a pipe and corrupts the regex.
_NCU_CANDIDATES = [
    os.environ.get("NCU", ""),
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\target\windows-desktop-win7-x64\ncu.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.1\target\windows-desktop-win7-x64\ncu.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\target\windows-desktop-win7-x64\ncu.exe",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\ncu.exe",
    "ncu",
]
_NCU = next((p for p in _NCU_CANDIDATES if p and shutil.which(p)), None)
if _NCU is None:
    sys.exit("[profile_ncu] ERROR: ncu not found. Set NCU=<path-to-ncu.exe> or install Nsight Compute.")
print(f"[profile_ncu] ncu: {_NCU}")

# ── Target command ─────────────────────────────────────────────────────────────
if args.config:
    # Production path: reuses the cached engine from td_config.yaml via profile_nsys.py.
    # This is the correct target for profiling the actual deployed engine.
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_PROJECT_ROOT, args.config)
    target_cmd = [
        _PYTHON,
        os.path.join(_PROJECT_ROOT, "scripts", "profiling", "profile_nsys.py"),
        "--target",
        "benchmark",
        "--config",
        cfg_path,
    ]
    _target_label = f"config_{os.path.splitext(os.path.basename(args.config))[0]}"
else:
    _TARGETS = {
        "benchmark": [
            _PYTHON,
            os.path.join(_PROJECT_ROOT, "examples", "benchmark", "single.py"),
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--acceleration",
            "tensorrt",
        ],
        "infer": [
            _PYTHON,
            os.path.join(_PROJECT_ROOT, "examples", "benchmark", "single.py"),
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--acceleration",
            "tensorrt",
        ],
    }
    target_cmd = _TARGETS[args.target]
    _target_label = args.target

# ── Output paths ───────────────────────────────────────────────────────────────
rep_name = f"ncu_{_target_label}_{args.metric_set}_{_TIMESTAMP}"
rep_path = os.path.join(_LOGS_DIR, rep_name + ".ncu-rep")
csv_path = os.path.join(_LOGS_DIR, rep_name + ".csv")

# ── Build ncu command ──────────────────────────────────────────────────────────
ncu_cmd = [
    str(_NCU),
    "--target-processes",
    "all",
    "--set",
    args.metric_set,
    "--kernel-name-base",
    "demangled",
    "--launch-skip",
    str(args.launch_skip),
    "--launch-count",
    str(args.launch_count),
    "--import-source",
    "yes",
    "--source-folders",
    os.path.join(_PROJECT_ROOT, "src", "streamdiffusion"),
    "--export",
    rep_path,
]
if args.kernel_regex:
    ncu_cmd += ["--kernel-name", args.kernel_regex]

ncu_cmd += target_cmd

# ── Environment ────────────────────────────────────────────────────────────────
proc_env = dict(os.environ)
proc_env["CUDA_LAUNCH_BLOCKING"] = "1"  # required for accurate per-kernel profiling
proc_env["GPU_PROFILER"] = "1"

print(f"\n[profile_ncu] target:      {args.target}")
print(f"[profile_ncu] metric set:  {args.metric_set}")
print(f"[profile_ncu] launch skip: {args.launch_skip}  count: {args.launch_count}")
print(f"[profile_ncu] output:      {rep_path}")
print(f"\n[profile_ncu] Command:\n  {' '.join(ncu_cmd)}\n")

if args.dry_run:
    print("[profile_ncu] --dry-run: exiting without executing.")
    sys.exit(0)

# ── Run ncu (stream output + early-abort on known fatal errors) ───────────────
# ncu often keeps the target process running even after it can't collect metrics
# (e.g. ERR_NVGPUCTRPERM), so a 30-second TRT workload would otherwise complete
# only to produce no .ncu-rep. Stream stderr/stdout, kill the proc on first match.
_FATAL_NCU_PATTERNS = [
    (
        "ERR_NVGPUCTRPERM",
        "GPU performance counter access denied. Either run this terminal as "
        "Administrator OR open NVIDIA Control Panel -> Desktop menu -> Enable "
        "Developer Settings -> Developer -> Manage GPU Performance Counters -> "
        "'Allow access to the GPU performance counters to all users'. "
        "See https://developer.nvidia.com/ERR_NVGPUCTRPERM",
    ),
]

proc = subprocess.Popen(
    ncu_cmd,
    env=proc_env,
    cwd=_PROJECT_ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
abort_msg = None
assert proc.stdout is not None
for line in iter(proc.stdout.readline, ""):
    sys.stdout.write(line)
    sys.stdout.flush()
    for pat, msg in _FATAL_NCU_PATTERNS:
        if pat in line:
            abort_msg = msg
            break
    if abort_msg:
        break

if abort_msg:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
else:
    proc.wait()

if abort_msg:
    print(f"\n[profile_ncu] ABORTED: {abort_msg}", file=sys.stderr)
    sys.exit(2)

print(f"\n[profile_ncu] ncu exited (code {proc.returncode})")

# Post-run sanity check: ncu can exit 0 even when no report is produced
# (e.g. zero matched kernels under --kernel-name). Surface that explicitly.
if not os.path.exists(rep_path):
    print(
        f"[profile_ncu] WARNING: expected report at {rep_path} but file is missing.\n"
        "  Likely causes: --kernel-name regex matched zero kernels, or ncu "
        "encountered an error not in _FATAL_NCU_PATTERNS. Inspect the output above "
        "for ==ERROR== / ==WARNING== lines.",
        file=sys.stderr,
    )
    sys.exit(2)

print(f"[profile_ncu] Report -> {rep_path}")
print(f"[profile_ncu] Open with: Nsight Compute UI -> File -> Open -> {rep_name}.ncu-rep")

# ── Optional CSV export ────────────────────────────────────────────────────────
if args.csv and os.path.exists(rep_path):
    csv_cmd = [str(_NCU), "--csv", "--page", "details", "--import", rep_path]
    print(f"\n[profile_ncu] Exporting CSV -> {csv_path}")
    with open(csv_path, "w") as csv_fh:
        subprocess.run(csv_cmd, stdout=csv_fh, env=proc_env)
    print(f"[profile_ncu] CSV -> {csv_path}")
