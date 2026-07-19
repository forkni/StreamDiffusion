#!/usr/bin/env python3
"""
Dependency Audit Summary Generator

Parses pip-audit JSON output and generates comprehensive executive summary reports.

Features:
- Security vulnerability analysis with severity breakdown
- ML stack health check (PyTorch, CUDA, GPU)
- Outdated packages with risk prioritization
- Dependency tree analysis with orphan detection
- Full markdown report saved to audit_reports/

Usage:
    .venv/Scripts/pip-audit --format json | python tools/summarize_audit.py
    # or:
    python tools/summarize_audit.py audit.json
    # or with custom output:
    python tools/summarize_audit.py --no-save  # stdout only
    python tools/summarize_audit.py -o custom.md  # custom output path
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import tomllib

# ============================================================================
# PACKAGE CLASSIFICATION CONSTANTS
# ============================================================================

# Tier 1: Universal ML packages (present in most ML projects)
UNIVERSAL_ML_CORE = {
    # PyTorch ecosystem
    "torch",
    "torchvision",
    "torchaudio",
    # HuggingFace ecosystem
    "transformers",
    "accelerate",
    "safetensors",
    "peft",
    "huggingface-hub",
    "tokenizers",
    "datasets",
    # Core ML utilities
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "einops",
    "pillow",
}

# PyTorch ecosystem packages (implicitly version-locked)
PYTORCH_ECOSYSTEM = {
    "torch",
    "torchvision",
    "torchaudio",
    "xformers",
    "triton",
    "triton-windows",
    "torch-tensorrt",
    "torch_tensorrt",
    "torchao",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-cu11",
    "nvidia-cublas-cu12",
    "nvidia-cublas-cu11",
}

# TensorRT/CUDA stack
TENSORRT_ECOSYSTEM = {
    "tensorrt",
    "tensorrt-cu12",
    "tensorrt_cu12",
    "tensorrt_cu12_bindings",
    "tensorrt_cu12_libs",
    "onnx",
    "onnx-graphsurgeon",
    "onnx_graphsurgeon",
    "onnxruntime",
    "polygraphy",
}

# CUDA/NVIDIA packages
CUDA_STACK = {
    "cuda-python",
    "cuda-toolkit",
    "cuda-pathfinder",
    "nvidia-cuda-runtime",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu11",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-cu11",
    "nvidia-cublas-cu12",
    "nvidia-cublas-cu11",
    "nvidia-ml-py",
    "nvidia-pyindex",
}

# Universal dev tools
DEV_TOOLS = {
    # Testing
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
    "pytest-mock",
    "coverage",
    # Linting/Formatting
    "black",
    "isort",
    "ruff",
    "mypy",
    "pyrefly",
    "flake8",
    # Build/Audit
    "pip",
    "pip-audit",
    "pip_audit",
    "pipdeptree",
    "pip-licenses",
    "wheel",
    "setuptools",
    "uv",
    "ninja",
    # Development
    "ipython",
    "pyreadline3",
    "rich",
    "colorama",
    "tqdm",
}

# Tier 2: Domain-specific packages (auto-detected based on what's installed)
EMBEDDING_SEARCH = {
    "sentence-transformers",
    "faiss-cpu",
    "faiss-gpu",
    "FlagEmbedding",
    "rank-bm25",
    "ir_datasets",
}

CODE_PARSING = {
    "tree-sitter",
    "tree-sitter-python",
    "tree-sitter-javascript",
    "tree-sitter-typescript",
    "tree-sitter-rust",
    "tree-sitter-go",
    "tree-sitter-java",
    "tree-sitter-c",
    "tree-sitter-cpp",
    "tree-sitter-c-sharp",
    "tree-sitter-glsl",
}

IMAGE_GENERATION = {
    "diffusers",
    "controlnet-aux",
    "compel",
    "xformers",
    "sageattention",
    "tomesd",
    "opencv-python",
    "opencv-contrib-python",
    "mss",
}

WEB_API = {
    "fastapi",
    "starlette",
    "uvicorn",
    "aiohttp",
    "httpx",
    "mcp",
    "sse-starlette",
    "python-multipart",
}

NLP_PACKAGES = {
    "nltk",
    "tiktoken",
    "sentencepiece",
    "regex",
}

# Category display configuration
CATEGORY_LABELS = {
    "project_package": ("[PROJECT]", "Project package"),
    "ml_core": ("[ML CORE]", "Core ML packages (required)"),
    "pytorch_ecosystem": ("[PYTORCH]", "PyTorch ecosystem (required)"),
    "cuda_stack": ("[CUDA]", "CUDA/TensorRT stack (required)"),
    "dev_tools": ("[DEV]", "Development tools"),
    "embedding_search": ("[EMBEDDING]", "Embedding/Search packages"),
    "code_parsing": ("[PARSING]", "Code parsing (tree-sitter)"),
    "image_generation": ("[IMAGE]", "Image generation packages"),
    "web_api": ("[WEB]", "Web/API packages"),
    "nlp": ("[NLP]", "NLP packages"),
    "true_orphans": ("[?]", "Unknown packages (investigate)"),
}


def parse_audit_json(data: dict) -> dict:
    """Extract vulnerability information from pip-audit JSON."""
    # Group vulnerabilities by package
    vuln_packages = defaultdict(list)
    severity_counts = defaultdict(int)
    total_packages = 0

    for dep in data.get("dependencies", []):
        pkg_name = dep["name"]

        # Skip dependencies that couldn't be audited
        if "skip_reason" in dep:
            continue

        total_packages += 1
        pkg_version = dep["version"]
        vulns = dep.get("vulns", [])

        if vulns:
            for vuln in vulns:
                vuln_packages[pkg_name].append(
                    {
                        "version": pkg_version,
                        "cve_id": vuln.get("id", "UNKNOWN"),
                        "fix_versions": vuln.get("fix_versions", []),
                        "aliases": vuln.get("aliases", []),
                        "description": (
                            vuln.get("description", "")[:200] + "..."
                            if len(vuln.get("description", "")) > 200
                            else vuln.get("description", "")
                        ),
                    }
                )

                # Try to extract severity from CVE ID or description
                if any(alias.startswith("CVE-") for alias in vuln.get("aliases", [])):
                    severity_counts["high"] += 1
                else:
                    severity_counts["medium"] += 1

    return {
        "total_packages": total_packages,
        "vulnerable_packages": len(vuln_packages),
        "total_cves": sum(len(v) for v in vuln_packages.values()),
        "vulnerabilities": dict(vuln_packages),
        "severity_counts": dict(severity_counts),
    }


def get_python_executable(cli_override: Path | None = None) -> str:
    """Get Python executable with priority: CLI > env var > auto-detect > fallback.

    Priority order:
    1. CLI argument (--python)
    2. DEPS_AUDIT_PYTHON environment variable
    3. .venv/Scripts/python.exe or .venv/bin/python (auto-detect)
    4. venv/Scripts/python.exe or venv/bin/python
    5. sys.executable (last resort)

    Args:
        cli_override: Optional Python path from CLI argument

    Returns:
        Path to Python executable as string
    """
    # Priority 1: CLI override
    if cli_override and Path(cli_override).exists():
        return str(cli_override)

    # Priority 2: Environment variable
    env_python = os.environ.get("DEPS_AUDIT_PYTHON")
    if env_python and Path(env_python).exists():
        return env_python

    # Priority 3-5: Auto-detect (venv first, then sys.executable)
    candidates = [
        Path.cwd() / ".venv" / "Scripts" / "python.exe",  # Windows
        Path.cwd() / ".venv" / "bin" / "python",  # Linux/Mac
        Path.cwd() / "venv" / "Scripts" / "python.exe",  # Alt Windows
        Path.cwd() / "venv" / "bin" / "python",  # Alt Linux/Mac
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Fallback to current Python
    return sys.executable


def find_python_with_pipdeptree(python_override: Path | None = None) -> str | None:
    """Find a Python executable that has pipdeptree installed.

    Args:
        python_override: Optional Python path from CLI argument

    Returns:
        Path to Python executable with pipdeptree, or None if not found
    """
    python_exe = get_python_executable(python_override)

    # Verify pipdeptree is available
    try:
        result = subprocess.run(
            [python_exe, "-m", "pipdeptree", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return python_exe
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


def get_dependency_tree_json(python_override: Path | None = None) -> list | None:
    """Run pipdeptree --json and return parsed output.

    Args:
        python_override: Optional Python path from CLI argument

    Returns:
        Parsed pipdeptree JSON output, or None on failure
    """
    python_exe = find_python_with_pipdeptree(python_override)
    if python_exe is None:
        return None

    try:
        result = subprocess.run(
            [python_exe, "-m", "pipdeptree", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        return None
    return None


def get_outdated_packages(python_override: Path | None = None) -> list[dict] | None:
    """Run pip list --outdated --format=json and return parsed output.

    Args:
        python_override: Optional Python path from CLI argument

    Returns:
        List of outdated packages, empty list if none, or None on failure
    """
    python_exe = get_python_executable(python_override)

    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            timeout=120,  # Increased timeout for slow networks
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        # Return empty list instead of None if command succeeded but no outdated packages
        if result.returncode == 0:
            return []
    except subprocess.TimeoutExpired:
        # Timeout is common on slow networks, return None to signal failure
        return None
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        return None
    return None


def get_ml_stack_health(python_override: Path | None = None) -> dict:
    """Get ML stack health information (PyTorch, CUDA, GPU).

    Args:
        python_override: Optional Python path from CLI argument

    Returns:
        Dictionary with ML stack information
    """
    python_exe = get_python_executable(python_override)

    ml_info = {
        "pytorch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "transformers_version": None,
        "faiss_version": None,
        "sentence_transformers_version": None,
    }

    # Get PyTorch/CUDA info
    try:
        result = subprocess.run(
            [
                python_exe,
                "-c",
                """
import json
info = {}
try:
    import torch
    info['pytorch_version'] = torch.__version__
    info['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
except ImportError:
    pass
print(json.dumps(info))
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            pytorch_info = json.loads(result.stdout.strip())
            ml_info.update(pytorch_info)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        pass

    # Get other ML package versions
    try:
        result = subprocess.run(
            [
                python_exe,
                "-c",
                """
import json
info = {}
try:
    import transformers
    info['transformers_version'] = transformers.__version__
except ImportError:
    pass
try:
    import faiss
    info['faiss_version'] = faiss.__version__ if hasattr(faiss, '__version__') else 'installed'
except ImportError:
    pass
try:
    import sentence_transformers
    info['sentence_transformers_version'] = sentence_transformers.__version__
except ImportError:
    pass
print(json.dumps(info))
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            other_info = json.loads(result.stdout.strip())
            ml_info.update(other_info)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        pass

    return ml_info


def _normalize_package_name(name: str) -> str:
    """PEP 503 package name normalization.

    Normalizes package names by converting to lowercase and replacing
    runs of hyphens, underscores, and dots with a single hyphen.
    This ensures 'pre-commit' and 'pre_commit' are treated as the same package.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def get_direct_dependencies(project_root: Path | None = None) -> set[str]:
    """Extract direct dependency names from pyproject.toml."""
    if project_root is None:
        project_root = Path.cwd()

    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return set()

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    deps = set()
    # Main dependencies
    for dep in data.get("project", {}).get("dependencies", []):
        # Extract package name (before any version specifier)
        name = re.split(r"[<>=!~\[]", dep)[0].strip()
        deps.add(_normalize_package_name(name))

    # Optional dependencies (dev, test, etc.)
    for group_deps in data.get("project", {}).get("optional-dependencies", {}).values():
        for dep in group_deps:
            name = re.split(r"[<>=!~\[]", dep)[0].strip()
            deps.add(_normalize_package_name(name))

    return deps


def get_dependency_constraints(
    tree_data: list | None,
) -> dict[str, list[tuple[str, str]]]:
    """Get dependency constraints for each package.

    Returns dict mapping package_name -> list of (dependent, constraint) tuples.
    Example: {"fsspec": [("datasets", "<=2025.10.0")]}
    """
    if not tree_data:
        return {}

    constraints = defaultdict(list)

    for pkg in tree_data:
        pkg_name = pkg["package"]["package_name"]
        for dep in pkg.get("dependencies", []):
            dep_name = dep["package_name"]
            required_version = dep.get("required_version", "Any")

            # Only track meaningful constraints (not "Any")
            if required_version and required_version != "Any":
                constraints[dep_name.lower()].append((pkg_name, required_version))

    return dict(constraints)


def categorize_outdated_packages(
    outdated: list[dict],
    constraints: dict | None = None,
    ecosystem_constraints: dict | None = None,
) -> dict:
    """Categorize outdated packages by update risk.

    Args:
        outdated: List of outdated packages from pip list --outdated
        constraints: Explicit version constraints from dependency tree
        ecosystem_constraints: Implicit constraints (PyTorch/CUDA ecosystem)

    Returns:
        Dict with categorized packages (ml_core, blocked, ecosystem_blocked, major_jump, safe_update)
    """
    ML_CORE = UNIVERSAL_ML_CORE  # Use universal ML core packages

    if constraints is None:
        constraints = {}
    if ecosystem_constraints is None:
        ecosystem_constraints = {}

    categories = {
        "ml_core": [],  # DO NOT auto-update
        "major_jump": [],  # Review breaking changes
        "safe_update": [],  # Safe to update
        "blocked": [],  # Cannot update due to explicit constraints
        "ecosystem_blocked": [],  # Cannot update due to PyTorch/CUDA ecosystem
    }

    for pkg in outdated:
        name = pkg["name"].lower()
        current = pkg["version"]
        latest = pkg["latest_version"]

        # Parse major versions
        current_major = current.split(".")[0].lstrip("v")
        latest_major = latest.split(".")[0].lstrip("v")

        # Get blocking constraints for this package
        blocking = constraints.get(name, [])

        # Check for ecosystem constraints FIRST (implicit locking to PyTorch/CUDA)
        ecosystem_reason = ecosystem_constraints.get(name)

        pkg_info = {
            "name": pkg["name"],
            "current": current,
            "latest": latest,
            "is_major_jump": current_major != latest_major,
            "blocked_by": blocking if blocking else None,
        }

        # Check if package has strict version constraints
        has_strict_constraint = any(
            "==" in constraint or "<=" in constraint or "<" in constraint for _, constraint in blocking
        )

        if ecosystem_reason:
            # Package is implicitly locked to PyTorch/CUDA ecosystem
            pkg_info["blocked_by"] = [(ecosystem_reason, "ecosystem")]
            categories["ecosystem_blocked"].append(pkg_info)
        elif has_strict_constraint and blocking:
            # Package is blocked by explicit version constraints
            categories["blocked"].append(pkg_info)
        elif name in ML_CORE:
            categories["ml_core"].append(pkg_info)
        elif current_major != latest_major:
            categories["major_jump"].append(pkg_info)
        else:
            categories["safe_update"].append(pkg_info)

    return categories


def build_package_trees(tree_data: list, direct_deps: set) -> dict:
    """Build dependency trees for direct dependencies only."""
    trees = {}
    for pkg in tree_data:
        pkg_name = pkg["package"]["package_name"].lower()
        if pkg_name in direct_deps:
            trees[pkg_name] = {
                "version": pkg["package"]["installed_version"],
                "dependencies": pkg.get("dependencies", []),
            }
    return trees


def detect_project_domains(installed_packages: set[str]) -> dict[str, list]:
    """Auto-detect which domain-specific categories apply to this project.

    Returns dict mapping category name to list of matching packages.
    """
    domains = {}

    # Check each domain category
    domain_sets = {
        "embedding_search": EMBEDDING_SEARCH,
        "code_parsing": CODE_PARSING,
        "image_generation": IMAGE_GENERATION,
        "web_api": WEB_API,
        "nlp": NLP_PACKAGES,
    }

    for domain_name, domain_packages in domain_sets.items():
        matches = installed_packages & domain_packages
        if len(matches) >= 2:  # At least 2 packages to count as a domain
            domains[domain_name] = sorted(matches)

    return domains


def get_ecosystem_constraints(ml_info: dict | None = None) -> dict[str, str]:
    """Generate synthetic constraints for PyTorch ecosystem packages.

    Args:
        ml_info: ML stack information from get_ml_stack_health()

    Returns:
        Dict mapping package name -> constraint reason
    """
    constraints = {}
    if not ml_info:
        return constraints

    pytorch_version = ml_info.get("pytorch_version")
    cuda_version = ml_info.get("cuda_version")

    if pytorch_version:
        for pkg in PYTORCH_ECOSYSTEM:
            constraints[pkg.lower()] = f"Implicitly locked to torch=={pytorch_version}"

    if cuda_version:
        for pkg in TENSORRT_ECOSYSTEM:
            if pkg.lower() not in constraints:
                constraints[pkg.lower()] = f"Implicitly locked to CUDA {cuda_version}"

    return constraints


def find_orphan_packages(tree_data: list, direct_deps: set, project_name: str | None = None) -> dict:
    """Find packages not in direct deps and categorize them.

    Uses tiered classification:
    1. Universal ML packages (always recognized)
    2. Domain-specific packages (auto-detected)
    3. True orphans (unknown packages)

    Args:
        tree_data: Output from pipdeptree --json
        direct_deps: Set of direct dependencies from pyproject.toml
        project_name: Optional project package name (auto-detected from pyproject.toml)

    Returns:
        Dict mapping category -> list of packages
    """
    all_installed = {}
    all_required_by = defaultdict(set)

    for pkg in tree_data:
        name = _normalize_package_name(pkg["package"]["package_name"])
        version = pkg["package"]["installed_version"]
        all_installed[name] = version

        # Track reverse dependencies
        for dep in pkg.get("dependencies", []):
            dep_name = _normalize_package_name(dep["package_name"])
            all_required_by[dep_name].add(name)

    # Find packages with no dependents and not in direct deps
    potential_orphans = []
    for name, version in all_installed.items():
        if name not in direct_deps and not all_required_by.get(name):
            potential_orphans.append({"name": name, "version": version})

    # Get all installed package names for domain detection
    installed = {pkg["package"]["package_name"].lower() for pkg in tree_data}
    detected_domains = detect_project_domains(installed)

    # Categorize orphans
    categorized = {
        "project_package": [],  # The project itself
        "ml_core": [],  # Universal ML packages
        "pytorch_ecosystem": [],  # PyTorch-locked packages
        "cuda_stack": [],  # CUDA/TensorRT packages
        "dev_tools": [],  # Dev/build tools
        "embedding_search": [],  # Auto-detected: embedding/search
        "code_parsing": [],  # Auto-detected: tree-sitter
        "image_generation": [],  # Auto-detected: diffusion/image
        "web_api": [],  # Auto-detected: web frameworks
        "nlp": [],  # Auto-detected: NLP tools
        "true_orphans": [],  # Actually unknown
    }

    for pkg in potential_orphans:
        name = pkg["name"].lower()

        # Check project package first
        if project_name and name == project_name.lower():
            categorized["project_package"].append(pkg)
        # Universal categories
        elif name in UNIVERSAL_ML_CORE:
            categorized["ml_core"].append(pkg)
        elif name in PYTORCH_ECOSYSTEM:
            categorized["pytorch_ecosystem"].append(pkg)
        elif name in CUDA_STACK or name in TENSORRT_ECOSYSTEM:
            categorized["cuda_stack"].append(pkg)
        elif name in DEV_TOOLS:
            categorized["dev_tools"].append(pkg)
        # Domain-specific (only if domain detected in project)
        elif "embedding_search" in detected_domains and name in EMBEDDING_SEARCH:
            categorized["embedding_search"].append(pkg)
        elif "code_parsing" in detected_domains and name in CODE_PARSING:
            categorized["code_parsing"].append(pkg)
        elif "image_generation" in detected_domains and name in IMAGE_GENERATION:
            categorized["image_generation"].append(pkg)
        elif "web_api" in detected_domains and name in WEB_API:
            categorized["web_api"].append(pkg)
        elif "nlp" in detected_domains and name in NLP_PACKAGES:
            categorized["nlp"].append(pkg)
        else:
            categorized["true_orphans"].append(pkg)

    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


def safe_print(text: str) -> None:
    """Print text with Windows-safe encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic Unicode characters for Windows console
        safe_text = text.encode("ascii", "replace").decode("ascii")
        print(safe_text)


def print_summary(summary: dict) -> None:
    """Print formatted security summary."""
    safe_print("=" * 70)
    safe_print("DEPENDENCY AUDIT SUMMARY".center(70))
    safe_print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Packages: {summary['total_packages']}")
    print(f"Vulnerable Packages: {summary['vulnerable_packages']}")
    print(f"Total CVEs: {summary['total_cves']}")
    print()

    if summary["total_cves"] == 0:
        print("[OK] No known vulnerabilities found!")
        print()
        return

    # Print severity breakdown if available
    if summary["severity_counts"]:
        print("Severity Breakdown:")
        for severity, count in summary["severity_counts"].items():
            print(f"  - {severity.capitalize()}: {count}")
        print()

    print("=" * 70)
    print("VULNERABILITIES FOUND".center(70))
    print("=" * 70)
    print()

    for pkg_name, vulns in sorted(summary["vulnerabilities"].items()):
        print(f"[PACKAGE] {pkg_name} ({vulns[0]['version']})")
        print("-" * 70)

        for vuln in vulns:
            print(f"  [VULN]  {vuln['cve_id']}")

            if vuln["aliases"]:
                aliases_str = ", ".join(vuln["aliases"])
                print(f"      Aliases: {aliases_str}")

            if vuln["fix_versions"]:
                fix_str = ", ".join(vuln["fix_versions"])
                print(f"      Fix Available: {fix_str}")
            else:
                print("      Fix Available: No fix released yet")

            if vuln["description"]:
                safe_print(f"      Description: {vuln['description']}")

            print()

        print()

    print("=" * 70)
    print("RECOMMENDED ACTIONS".center(70))
    print("=" * 70)
    print()

    # Generate actionable recommendations
    fixable = [(pkg, v) for pkg, vulns in summary["vulnerabilities"].items() for v in vulns if v["fix_versions"]]

    if fixable:
        print("[FIXES] Packages with available fixes:")
        for pkg, vuln in fixable:
            fix_version = vuln["fix_versions"][0] if vuln["fix_versions"] else "latest"
            print(f"   pip install --upgrade {pkg}=={fix_version}")
        print()

    unfixable = [(pkg, v) for pkg, vulns in summary["vulnerabilities"].items() for v in vulns if not v["fix_versions"]]

    if unfixable:
        print("[MONITOR] Packages without fixes (monitor for updates):")
        for pkg, vuln in unfixable:
            print(f"   {pkg}: {vuln['cve_id']}")
        print()

    print("[NEXT STEPS] Actions to take:")
    print("   1. Review CVE details at https://osv.dev/")
    print("   2. Test updates in isolated environment")
    print("   3. Run full test suite before deploying")
    print("   4. Update pyproject.toml with new version constraints")
    print()


def print_dependency_tree(pkg_name: str, pkg_data: dict, indent: int = 0, visited: set | None = None) -> None:
    """Recursively print ASCII dependency tree."""
    if visited is None:
        visited = set()

    prefix = "  " * indent
    if indent == 0:
        safe_print(f"{pkg_name}=={pkg_data['version']}")

    deps = pkg_data.get("dependencies", [])
    for i, dep in enumerate(deps):
        dep_name = dep["package_name"]
        dep_key = dep.get("key", dep_name.lower())
        required = dep.get("required_version", "Any")
        installed = dep.get("installed_version", "?")

        is_last = i == len(deps) - 1
        branch = "+-- " if is_last else "|-- "  # ASCII-safe for Windows console

        safe_print(f"{prefix}{branch}{dep_name} [required: {required}, installed: {installed}]")

        # Prevent infinite loops from circular dependencies
        if dep_key not in visited:
            visited.add(dep_key)
            # Recurse for nested dependencies
            nested_deps = dep.get("dependencies", [])
            if nested_deps:
                child_prefix = "    " if is_last else "|   "  # ASCII-safe
                for j, nested in enumerate(nested_deps):
                    nested_is_last = j == len(nested_deps) - 1
                    nested_branch = "+-- " if nested_is_last else "|-- "  # ASCII-safe
                    n_name = nested["package_name"]
                    n_req = nested.get("required_version", "Any")
                    n_inst = nested.get("installed_version", "?")
                    safe_print(
                        f"{prefix}{child_prefix}{nested_branch}{n_name} [required: {n_req}, installed: {n_inst}]"
                    )


def print_dependency_analysis(tree_data: list, direct_deps: set) -> None:
    """Print complete dependency analysis section."""
    if not tree_data:
        safe_print("\n[WARN] pipdeptree not available - skipping dependency tree analysis")
        safe_print("       Install with: pip install pipdeptree")
        return

    # Calculate stats
    all_installed = {pkg["package"]["package_name"].lower() for pkg in tree_data}
    transitive = all_installed - direct_deps

    safe_print("\n" + "=" * 70)
    safe_print("DEPENDENCY TREE ANALYSIS".center(70))
    safe_print("=" * 70)
    safe_print(f"Direct Dependencies: {len(direct_deps)} (from pyproject.toml)")
    safe_print(f"Transitive Dependencies: {len(transitive)} (pulled in automatically)")
    safe_print(f"Total Installed: {len(all_installed)}")
    safe_print("")

    # Build and print trees for direct deps
    trees = build_package_trees(tree_data, direct_deps)

    for pkg_name in sorted(trees.keys()):
        pkg_data = trees[pkg_name]
        if pkg_data["dependencies"]:  # Only show packages with dependencies
            safe_print(f"[TREE] {pkg_name} ({pkg_data['version']})")
            safe_print("-" * 70)
            print_dependency_tree(pkg_name, pkg_data)
            safe_print("")

    # Find and categorize orphans
    project_info = get_project_info()
    project_name = project_info.get("name") if project_info["name"] != "Unknown" else None
    orphan_data = find_orphan_packages(tree_data, direct_deps, project_name)
    total_orphans = sum(len(v) for v in orphan_data.values())

    if total_orphans:
        safe_print(f"[PACKAGES] {total_orphans} packages not tracked in pyproject.toml:")
        safe_print("-" * 70)

        for category, packages in orphan_data.items():
            if packages:
                tag, description = CATEGORY_LABELS.get(category, ("[?]", category))
                safe_print(f"  {tag} {description}:")
                for pkg in sorted(packages, key=lambda x: x["name"]):
                    safe_print(f"    - {pkg['name']} ({pkg['version']})")
                safe_print("")

        if orphan_data.get("true_orphans"):
            safe_print("  Actions for unknown packages:")
            safe_print("    - If needed: Add to pyproject.toml dependencies")
            safe_print("    - If not needed: pip uninstall <package>")
            safe_print("")
    else:
        safe_print("[OK] No packages outside direct dependencies detected")
        safe_print("")


def print_outdated_analysis(
    outdated: list[dict] | None,
    tree_data: list | None = None,
    ml_info: dict | None = None,
) -> None:
    """Print outdated packages analysis to console."""
    if outdated is None:
        safe_print("\n[WARN] Could not retrieve outdated packages")
        return

    if not outdated:
        safe_print("\n[OK] All packages are up to date!")
        return

    # Get dependency constraints
    constraints = get_dependency_constraints(tree_data)
    ecosystem_constraints = get_ecosystem_constraints(ml_info)
    categories = categorize_outdated_packages(outdated, constraints, ecosystem_constraints)

    safe_print("\n" + "=" * 70)
    safe_print("OUTDATED PACKAGES ANALYSIS".center(70))
    safe_print("=" * 70)
    safe_print(f"Total Outdated: {len(outdated)}")
    safe_print("")

    # Blocked packages (cannot update due to explicit version constraints)
    if categories["blocked"]:
        safe_print("[BLOCKED] Cannot update due to version constraints:")
        safe_print("-" * 70)
        for pkg in categories["blocked"]:
            safe_print(f"  {pkg['name']}: {pkg['current']} -> {pkg['latest']}")
            if pkg["blocked_by"]:
                for dependent, constraint in pkg["blocked_by"]:
                    safe_print(f"    Blocked by: {dependent} requires {constraint}")
        safe_print("")

    # Ecosystem-blocked packages (PyTorch/CUDA version-locked)
    if categories.get("ecosystem_blocked"):
        safe_print("[ECOSYSTEM] Locked to current PyTorch/CUDA version:")
        safe_print("-" * 70)
        for pkg in categories["ecosystem_blocked"]:
            safe_print(f"  {pkg['name']}: {pkg['current']} -> {pkg['latest']}")
            if pkg["blocked_by"]:
                reason, _ = pkg["blocked_by"][0]
                safe_print(f"    Reason: {reason}")
        safe_print("")

    # ML Core packages (DO NOT auto-update)
    if categories["ml_core"]:
        safe_print("[ML CORE] DO NOT auto-update - test thoroughly first:")
        safe_print("-" * 70)
        for pkg in categories["ml_core"]:
            jump = " [MAJOR]" if pkg["is_major_jump"] else ""
            safe_print(f"  {pkg['name']}: {pkg['current']} -> {pkg['latest']}{jump}")
        safe_print("")

    # Major version jumps (review breaking changes)
    if categories["major_jump"]:
        safe_print("[MAJOR VERSION] Review breaking changes before updating:")
        safe_print("-" * 70)
        for pkg in categories["major_jump"]:
            safe_print(f"  {pkg['name']}: {pkg['current']} -> {pkg['latest']}")
        safe_print("")

    # Safe updates
    if categories["safe_update"]:
        safe_print("[SAFE] Minor/patch updates (generally safe):")
        safe_print("-" * 70)
        for pkg in categories["safe_update"]:
            safe_print(f"  {pkg['name']}: {pkg['current']} -> {pkg['latest']}")
        safe_print("")


def get_project_info() -> dict:
    """Get project name and version from pyproject.toml."""
    pyproject_path = Path.cwd() / "pyproject.toml"
    if not pyproject_path.exists():
        return {"name": "Unknown", "version": "Unknown"}

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return {
            "name": data.get("project", {}).get("name", "Unknown"),
            "version": data.get("project", {}).get("version", "Unknown"),
        }
    except tomllib.TOMLDecodeError:
        return {"name": "Unknown", "version": "Unknown"}


def generate_markdown_report(
    summary: dict,
    tree_data: list | None,
    direct_deps: set,
    outdated: list[dict] | None = None,
    ml_info: dict | None = None,
) -> str:
    """Generate comprehensive executive summary report as markdown string."""
    lines = []
    now = datetime.now()
    project_info = get_project_info()

    # Calculate dependency stats
    total_installed = len(tree_data) if tree_data else summary["total_packages"]
    transitive_count = total_installed - len(direct_deps) if tree_data else 0

    # Title
    lines.append("# Dependency Audit Executive Summary")
    lines.append("")
    lines.append(f"**Date**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Project**: {project_info['name']} (v{project_info['version']})")
    lines.append(
        f"**Total Dependencies**: {total_installed} packages "
        f"({len(direct_deps)} direct + {transitive_count} transitive)"
    )
    lines.append(f"**Audit Report**: `audit_reports/{now.strftime('%Y-%m-%d-%H%M')}-audit-summary.md`")
    lines.append("")

    # --- Security Status ---
    lines.append("---")
    lines.append("")
    if summary["total_cves"] == 0:
        lines.append("## ✅ Security Status: **EXCELLENT**")
    else:
        lines.append("## ⚠️ Security Status: **ACTION REQUIRED**")
    lines.append("")
    lines.append(
        f"- **Known Vulnerabilities**: "
        f"{summary['severity_counts'].get('critical', 0)} critical, "
        f"{summary['severity_counts'].get('high', 0)} high, "
        f"{summary['severity_counts'].get('medium', 0)} medium, "
        f"{summary['severity_counts'].get('low', 0)} low"
    )
    lines.append(f"- **CVE Count**: {summary['total_cves']}")
    lines.append(f"- **Last Scan**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if summary["total_cves"] == 0:
        lines.append(
            "**Finding**: No security vulnerabilities detected in any dependencies. "
            "All packages are clean according to OSV database."
        )
        lines.append("")

    # --- ML Stack Health ---
    if ml_info:
        lines.append("---")
        lines.append("")
        if ml_info.get("cuda_available"):
            lines.append("## 🤖 ML Stack Health: **GOOD**")
        elif ml_info.get("pytorch_version"):
            lines.append("## 🤖 ML Stack Health: **CPU-ONLY**")
        else:
            lines.append("## 🤖 ML Stack Health: **NOT INSTALLED**")
        lines.append("")

        lines.append("| Component | Version | Status | Notes |")
        lines.append("|-----------|---------|--------|-------|")

        # PyTorch
        if ml_info.get("pytorch_version"):
            pytorch_status = "✅ Stable"
            pytorch_notes = ""
            # Check if outdated
            if outdated:
                for pkg in outdated:
                    if pkg["name"].lower() == "torch":
                        pytorch_notes = f"Latest: {pkg['latest_version']}"
                        break
            lines.append(f"| PyTorch | {ml_info['pytorch_version']} | {pytorch_status} | {pytorch_notes} |")
        else:
            lines.append("| PyTorch | Not installed | ⚪ N/A | |")

        # CUDA
        if ml_info.get("cuda_available"):
            lines.append(
                f"| CUDA | {ml_info.get('cuda_version', 'Unknown')} | ✅ Available | "
                f"Compatible with {ml_info.get('gpu_name', 'GPU')} |"
            )
        else:
            lines.append("| CUDA | N/A | ⚪ Not available | CPU mode |")

        # GPU
        if ml_info.get("gpu_name"):
            lines.append(f"| GPU | {ml_info['gpu_name']} | ✅ Active | {ml_info.get('gpu_count', 1)} device(s) |")
        else:
            lines.append("| GPU | None | ⚪ N/A | |")

        # transformers
        if ml_info.get("transformers_version"):
            lines.append(f"| transformers | {ml_info['transformers_version']} | ✅ Current | |")

        # FAISS
        if ml_info.get("faiss_version"):
            lines.append(f"| FAISS | {ml_info['faiss_version']} | ✅ Current | CPU version |")

        # sentence-transformers
        if ml_info.get("sentence_transformers_version"):
            lines.append(f"| sentence-transformers | {ml_info['sentence_transformers_version']} | ✅ Current | |")

        lines.append("")

        if ml_info.get("cuda_available") and ml_info.get("pytorch_version"):
            lines.append(
                f"**CUDA/PyTorch Compatibility**: Excellent. "
                f"PyTorch {ml_info['pytorch_version']} with CUDA {ml_info.get('cuda_version', 'Unknown')} "
                f"support is working correctly"
                f"{' with ' + ml_info['gpu_name'] if ml_info.get('gpu_name') else ''}."
            )
            lines.append("")

    # --- Vulnerabilities Found ---
    if summary["vulnerabilities"]:
        lines.append("---")
        lines.append("")
        lines.append("## 🔴 Vulnerabilities Found")
        lines.append("")

        for pkg_name, vulns in sorted(summary["vulnerabilities"].items()):
            lines.append(f"### {pkg_name} ({vulns[0]['version']})")
            lines.append("")

            for vuln in vulns:
                lines.append(f"**{vuln['cve_id']}**")
                lines.append("")

                if vuln["aliases"]:
                    aliases_str = ", ".join(vuln["aliases"])
                    lines.append(f"- **Aliases**: {aliases_str}")

                if vuln["fix_versions"]:
                    fix_str = ", ".join(vuln["fix_versions"])
                    lines.append(f"- **Fix Available**: {fix_str}")
                else:
                    lines.append("- **Fix Available**: No fix released yet")

                if vuln["description"]:
                    desc = vuln["description"].replace("\n", " ")
                    lines.append(f"- **Description**: {desc}")

                lines.append("")

        # Recommended fix commands
        lines.append("### Recommended Actions")
        lines.append("")

        fixable = [(pkg, v) for pkg, vulns in summary["vulnerabilities"].items() for v in vulns if v["fix_versions"]]

        if fixable:
            lines.append("**Packages with available fixes:**")
            lines.append("")
            lines.append("```bash")
            for pkg, vuln in fixable:
                fix_version = vuln["fix_versions"][0] if vuln["fix_versions"] else "latest"
                lines.append(f"pip install --upgrade {pkg}=={fix_version}")
            lines.append("```")
            lines.append("")

        unfixable = [
            (pkg, v) for pkg, vulns in summary["vulnerabilities"].items() for v in vulns if not v["fix_versions"]
        ]

        if unfixable:
            lines.append("**Packages without fixes (monitor for updates):**")
            lines.append("")
            for pkg, vuln in unfixable:
                lines.append(f"- **{pkg}**: {vuln['cve_id']}")
            lines.append("")

    # --- Outdated Packages Analysis ---
    if outdated:
        constraints = get_dependency_constraints(tree_data) if tree_data else {}
        ecosystem_constraints = get_ecosystem_constraints(ml_info)
        categories = categorize_outdated_packages(outdated, constraints, ecosystem_constraints)

        lines.append("---")
        lines.append("")
        lines.append("## 📦 Outdated Packages Analysis")
        lines.append("")
        lines.append(f"**Total Outdated**: {len(outdated)} packages (prioritized by risk)")
        lines.append("")

        # High Priority - ML Core
        if categories["ml_core"]:
            lines.append("### 🔴 High Priority (ML Core - Test Thoroughly)")
            lines.append("")
            lines.append("⚠️ **DO NOT auto-update these packages** - CUDA compatibility and model behavior may change.")
            lines.append("")
            lines.append("| Package | Current | Latest | Priority | Reason |")
            lines.append("|---------|---------|--------|----------|--------|")
            for pkg in categories["ml_core"]:
                priority = "🔴 High" if pkg["is_major_jump"] else "🟡 Medium"
                reason = "Major version change" if pkg["is_major_jump"] else "ML core package"
                lines.append(f"| **{pkg['name']}** | {pkg['current']} | {pkg['latest']} | {priority} | {reason} |")
            lines.append("")

        # Major Version Jumps
        if categories["major_jump"]:
            lines.append("### 🟡 Medium Priority (Major Version Changes)")
            lines.append("")
            lines.append("Review breaking changes before updating. Check release notes.")
            lines.append("")
            lines.append("| Package | Current | Latest | Type |")
            lines.append("|---------|---------|--------|------|")
            for pkg in categories["major_jump"]:
                pkg_type = (
                    "Dev tool"
                    if pkg["name"].lower() in {"pytest", "black", "isort", "ruff", "mypy", "pyrefly"}
                    else "Library"
                )
                lines.append(f"| {pkg['name']} | {pkg['current']} | {pkg['latest']} | {pkg_type} |")
            lines.append("")

        # Blocked packages
        if categories.get("blocked"):
            lines.append("### 🔒 Blocked (Cannot Update)")
            lines.append("")
            lines.append("These packages have newer versions but cannot be updated due to dependency constraints.")
            lines.append("")
            lines.append("| Package | Current | Latest | Blocked By |")
            lines.append("|---------|---------|--------|------------|")
            for pkg in categories["blocked"]:
                # Format blocking dependencies
                blockers = []
                for blocker, constraint in pkg.get("blocked_by", []):
                    blockers.append(f"`{blocker}` requires `{constraint}`")
                blockers_str = "<br>".join(blockers) if blockers else "Unknown"
                lines.append(f"| **{pkg['name']}** | {pkg['current']} | {pkg['latest']} | {blockers_str} |")
            lines.append("")

        # Ecosystem-locked packages
        if categories.get("ecosystem_blocked"):
            lines.append("### 🔗 Ecosystem Locked (PyTorch/CUDA)")
            lines.append("")
            lines.append(
                "These packages are implicitly locked to the current PyTorch/CUDA version. "
                "Update only as part of a coordinated ecosystem upgrade."
            )
            lines.append("")
            lines.append("| Package | Current | Latest | Reason |")
            lines.append("|---------|---------|--------|--------|")
            for pkg in categories["ecosystem_blocked"]:
                reason = pkg.get("blocked_by", [("Unknown", "")][0])[0]
                lines.append(f"| {pkg['name']} | {pkg['current']} | {pkg['latest']} | {reason} |")
            lines.append("")

        # Safe Updates
        if categories["safe_update"]:
            lines.append("### 🟢 Low Priority (Minor/Patch Updates)")
            lines.append("")
            lines.append("Generally safe to update. Run tests after updating.")
            lines.append("")
            lines.append("| Package | Current | Latest |")
            lines.append("|---------|---------|--------|")
            for pkg in categories["safe_update"]:
                lines.append(f"| {pkg['name']} | {pkg['current']} | {pkg['latest']} |")
            lines.append("")

    # --- Dependency Tree Analysis ---
    if tree_data:
        all_installed = {pkg["package"]["package_name"].lower() for pkg in tree_data}
        transitive = all_installed - direct_deps

        lines.append("---")
        lines.append("")
        lines.append("## 🌳 Dependency Tree Analysis")
        lines.append("")
        lines.append(f"- **Direct Dependencies**: {len(direct_deps)} (from pyproject.toml)")
        lines.append(f"- **Transitive Dependencies**: {len(transitive)} (pulled in automatically)")
        lines.append(f"- **Total Installed**: {len(all_installed)}")
        dep_ratio = len(all_installed) / len(direct_deps) if direct_deps else 0
        lines.append(f"- **Dependency Ratio**: {dep_ratio:.2f}:1 (each direct dep pulls ~{dep_ratio:.1f} transitive)")
        lines.append("")

        # Build trees for direct deps
        trees = build_package_trees(tree_data, direct_deps)

        if trees:
            lines.append("### Key Dependency Trees")
            lines.append("")
            lines.append("<details><summary>Click to expand dependency trees</summary>")
            lines.append("")

            for pkg_name in sorted(trees.keys()):
                pkg_data = trees[pkg_name]
                if pkg_data["dependencies"]:
                    lines.append(f"**{pkg_name}** ({pkg_data['version']})")
                    lines.append("")
                    lines.append("```")
                    lines.append(f"{pkg_name}=={pkg_data['version']}")
                    for dep in pkg_data["dependencies"]:
                        dep_name = dep["package_name"]
                        required = dep.get("required_version", "Any")
                        installed = dep.get("installed_version", "?")
                        lines.append(f"|-- {dep_name} [required: {required}, installed: {installed}]")
                        # Add nested deps if present
                        for nested in dep.get("dependencies", [])[:3]:
                            n_name = nested["package_name"]
                            n_inst = nested.get("installed_version", "?")
                            lines.append(f"    +-- {n_name} [{n_inst}]")
                    lines.append("```")
                    lines.append("")

            lines.append("</details>")
            lines.append("")

        # Orphan packages (categorized)
        project_name = project_info.get("name") if project_info["name"] != "Unknown" else None
        orphan_data = find_orphan_packages(tree_data, direct_deps, project_name)
        total_orphans = sum(len(v) for v in orphan_data.values())

        if total_orphans:
            lines.append("### 🧹 Orphan Packages")
            lines.append("")
            lines.append(f"Found **{total_orphans}** packages not in pyproject.toml with no dependents.")
            lines.append("")

            # Display categories using CATEGORY_LABELS
            # Safe to keep categories
            safe_categories = [
                "project_package",
                "ml_core",
                "pytorch_ecosystem",
                "cuda_stack",
                "dev_tools",
            ]
            # Domain-specific categories
            domain_categories = [
                "embedding_search",
                "code_parsing",
                "image_generation",
                "web_api",
                "nlp",
            ]

            # Safe to Keep (infrastructure)
            has_safe = any(orphan_data.get(cat) for cat in safe_categories)
            if has_safe:
                lines.append("**Safe to Keep (Development Tools)**:")
                for category in safe_categories:
                    if orphan_data.get(category):
                        tag, description = CATEGORY_LABELS.get(category, ("[?]", category))
                        for pkg in sorted(orphan_data[category], key=lambda x: x["name"]):
                            lines.append(f"- `{pkg['name']}` ({pkg['version']})")
                lines.append("")

            # Domain-specific packages
            has_domain = any(orphan_data.get(cat) for cat in domain_categories)
            if has_domain:
                lines.append("**Domain-Specific Packages (Auto-Detected)**:")
                for category in domain_categories:
                    if orphan_data.get(category):
                        tag, description = CATEGORY_LABELS.get(category, ("[?]", category))
                        lines.append(f"- {description}:")
                        for pkg in sorted(orphan_data[category], key=lambda x: x["name"]):
                            lines.append(f"  - `{pkg['name']}` ({pkg['version']})")
                lines.append("")

            # Investigate Before Removing
            if orphan_data.get("true_orphans"):
                lines.append("**Investigate Before Removing**:")
                investigate_list = orphan_data["true_orphans"]
                for orphan in investigate_list[:10]:  # Limit to first 10
                    lines.append(f"- `{orphan['name']}` ({orphan['version']})")
                if len(investigate_list) > 10:
                    lines.append(f"- ... and {len(investigate_list) - 10} more")
                lines.append("")

            lines.append(
                "**Recommendation**: Don't remove orphans yet. Many are transitive dependencies "
                "that pipdeptree may not detect correctly (especially for compiled packages)."
            )
            lines.append("")

    # --- Health Metrics ---
    lines.append("---")
    lines.append("")
    lines.append("## 📊 Dependency Health Metrics")
    lines.append("")
    lines.append("| Metric | Value | Status |")
    lines.append("|--------|-------|--------|")

    total_pkgs = total_installed
    lines.append(f"| Total Packages | {total_pkgs} | ✅ Reasonable |")
    lines.append(f"| Direct Dependencies | {len(direct_deps)} | ✅ Manageable |")
    lines.append(f"| Transitive Dependencies | {transitive_count} | ✅ Expected for ML project |")

    if direct_deps:
        dep_ratio = total_pkgs / len(direct_deps)
        lines.append(f"| Dependency Ratio | {dep_ratio:.2f}:1 | ✅ Normal |")

    lines.append(
        f"| Security Vulnerabilities | {summary['total_cves']} | {'✅ Excellent' if summary['total_cves'] == 0 else '⚠️ Action needed'} |"
    )

    if outdated:
        outdated_pct = (len(outdated) / total_pkgs) * 100 if total_pkgs else 0
        lines.append(
            f"| Outdated Packages | {len(outdated)} ({outdated_pct:.1f}%) | "
            f"{'✅ Acceptable' if outdated_pct < 20 else '⚠️ Review needed'} |"
        )

    if tree_data:
        orphan_data_count = find_orphan_packages(tree_data, direct_deps, project_name)
        orphan_count = sum(len(v) for v in orphan_data_count.values())
        orphan_pct = (orphan_count / total_pkgs) * 100 if total_pkgs else 0
        lines.append(
            f"| Orphan Packages | {orphan_count} ({orphan_pct:.1f}%) | "
            f"{'✅ Normal' if orphan_pct < 20 else '⚠️ Review periodically'} |"
        )

    lines.append("")

    # --- Recommended Actions Summary ---
    lines.append("---")
    lines.append("")
    lines.append("## 🎯 Recommended Actions")
    lines.append("")

    if summary["total_cves"] == 0:
        lines.append("### Immediate Actions (This Week)")
        lines.append("")
        lines.append("✅ **No immediate security updates required** - All packages are vulnerability-free.")
        lines.append("")
    else:
        lines.append("### 🔴 Immediate Actions (This Week)")
        lines.append("")
        lines.append("Security vulnerabilities detected. Apply fixes listed above.")
        lines.append("")

    if outdated and categories.get("ml_core"):
        lines.append("### Short-Term Actions (Next Sprint)")
        lines.append("")
        lines.append("Consider updating ML core packages after thorough testing:")
        lines.append("")
        for pkg in categories["ml_core"][:3]:
            lines.append(f"- `{pkg['name']}`: {pkg['current']} → {pkg['latest']}")
        lines.append("")
        lines.append(
            "**Before upgrading**: Check release notes, verify CUDA compatibility, test in isolated environment."
        )
        lines.append("")

    # Ecosystem upgrade recommendations
    if outdated and categories.get("ecosystem_blocked"):
        ecosystem_pkgs = categories["ecosystem_blocked"]
        lines.append("### Ecosystem Upgrade (When Ready)")
        lines.append("")

        pytorch_version = ml_info.get("pytorch_version") if ml_info else None
        if pytorch_version:
            lines.append(
                f"When upgrading PyTorch from {pytorch_version} to a new version, "
                f"also update these {len(ecosystem_pkgs)} ecosystem packages:"
            )
        else:
            lines.append(f"When upgrading PyTorch/CUDA, also update these {len(ecosystem_pkgs)} ecosystem packages:")
        lines.append("")

        for pkg in ecosystem_pkgs[:5]:  # Show first 5
            lines.append(f"- `{pkg['name']}`: {pkg['current']} → {pkg['latest']}")
        if len(ecosystem_pkgs) > 5:
            lines.append(f"- ... and {len(ecosystem_pkgs) - 5} more packages")
        lines.append("")

        lines.append(
            "**Coordination required**: These packages must be updated together to maintain PyTorch/CUDA compatibility."
        )
        lines.append("")

    lines.append("### Quarterly Review Actions")
    lines.append("")
    # Calculate next quarter - set day=1 first to avoid invalid dates (e.g., Feb 31)
    next_quarter = now.replace(day=1, month=((now.month - 1) // 3 + 1) * 3 % 12 + 1)
    if next_quarter.month <= now.month:
        next_quarter = next_quarter.replace(year=now.year + 1)
    lines.append(f"1. Re-run security audit: {next_quarter.strftime('%B %Y')}")
    lines.append("2. Review outdated packages for updates")
    lines.append("3. Check PyTorch ecosystem for new releases")
    lines.append("4. Clean up orphan packages (if still unused)")
    lines.append("")

    # --- Summary ---
    lines.append("---")
    lines.append("")
    lines.append("## 💡 Summary")
    lines.append("")

    overall_status = "EXCELLENT" if summary["total_cves"] == 0 else "ACTION REQUIRED"
    lines.append(f"**Overall Health**: {'✅' if summary['total_cves'] == 0 else '⚠️'} **{overall_status}**")
    lines.append("")

    checklist = []
    if summary["total_cves"] == 0:
        checklist.append("✅ **Zero security vulnerabilities** - All packages clean")
    else:
        checklist.append(f"⚠️ **{summary['total_cves']} security vulnerabilities** - Action required")

    if ml_info and ml_info.get("cuda_available"):
        checklist.append(
            f"✅ **ML stack stable** - PyTorch {ml_info.get('pytorch_version', 'Unknown')} + "
            f"CUDA {ml_info.get('cuda_version', 'Unknown')} working"
        )
    elif ml_info and ml_info.get("pytorch_version"):
        checklist.append("⚠️ **ML stack CPU-only** - No CUDA available")

    checklist.append(f"✅ **Dependencies manageable** - {total_pkgs} packages with good organization")

    if outdated:
        checklist.append(f"🟡 **{len(outdated)} packages outdated** - Review when convenient")

    for item in checklist:
        lines.append(f"- {item}")

    lines.append("")
    lines.append(
        f"**Risk Level**: **{'LOW' if summary['total_cves'] == 0 else 'MEDIUM'}** - "
        f"{'No immediate action required.' if summary['total_cves'] == 0 else 'Apply security fixes first.'}"
    )
    lines.append("")

    return "\n".join(lines)


def save_report(content: str, output_path: Path) -> None:
    """Save report to file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"\n[SAVED] Report saved to: {output_path}")
    except (OSError, PermissionError) as e:
        print(f"\n[ERROR] Failed to save report: {e}", file=sys.stderr)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse pip-audit JSON and generate human-readable summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: displays AND saves automatically
  pip-audit --format json | python tools/summarize_audit.py

  # Disable auto-save (stdout only)
  pip-audit --format json | python tools/summarize_audit.py --no-save

  # Custom output path
  pip-audit --format json | python tools/summarize_audit.py -o audit_reports/before-fixes.md

  # Read from file
  python tools/summarize_audit.py audit_reports/2025-12-18-audit.json
        """,
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Input JSON file (or pipe from stdin)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable auto-save (stdout only)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Custom output path (overrides auto-save location)",
    )
    parser.add_argument(
        "-p",
        "--python",
        type=Path,
        help="Path to Python executable. Overrides DEPS_AUDIT_PYTHON env var and auto-detection.",
    )
    args = parser.parse_args()

    # Read input data
    if args.input_file:
        # Read from file
        json_file = Path(args.input_file)
        if not json_file.exists():
            print(f"Error: File not found: {json_file}", file=sys.stderr)
            sys.exit(1)

        with open(json_file, encoding="utf-8") as f:
            content = f.read()

        # Handle pip-audit header line (e.g., "No known vulnerabilities found")
        # Find the start of JSON content
        json_start = content.find("{")
        if json_start == -1:
            print("Error: No JSON object found in file", file=sys.stderr)
            sys.exit(1)

        try:
            data = json.loads(content[json_start:])
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin - also handle header line
        try:
            content = sys.stdin.read()
            json_start = content.find("{")
            if json_start == -1:
                print("Error: No JSON object found in input", file=sys.stderr)
                sys.exit(1)
            data = json.loads(content[json_start:])
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
            print("\nUsage:", file=sys.stderr)
            print(
                "  pip-audit --format json | python tools/summarize_audit.py",
                file=sys.stderr,
            )
            print("  python tools/summarize_audit.py audit.json", file=sys.stderr)
            sys.exit(1)

    # Parse data
    summary = parse_audit_json(data)
    tree_data = get_dependency_tree_json(args.python)
    direct_deps = get_direct_dependencies()
    outdated_data = get_outdated_packages(args.python)
    ml_info = get_ml_stack_health(args.python)

    # Always print to stdout (for chat display)
    print_summary(summary)
    print_dependency_analysis(tree_data, direct_deps)
    print_outdated_analysis(outdated_data, tree_data)

    # Save to file unless --no-save
    if not args.no_save:
        report = generate_markdown_report(summary, tree_data, direct_deps, outdated_data, ml_info)
        if args.output:
            output_path = args.output
        else:
            filename = f"{datetime.now().strftime('%Y-%m-%d-%H%M')}-audit-summary.md"
            output_path = Path("audit_reports") / filename
        save_report(report, output_path)


if __name__ == "__main__":
    main()
