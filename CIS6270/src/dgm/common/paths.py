"""Where everything lives on disk.

The code no longer sits next to the data it reads. Under a src layout a
module's own location says nothing about where results belong, so every path
in the project is resolved here instead of from `__file__`.

Resolution order for the course root (the directory holding pyproject.toml and
src/dgm):

  1. $DGM_ROOT, if set. This is the escape hatch: point it at a checkout to run
     against that checkout's data, or at a scratch directory to keep a run's
     artifacts out of the repository.
  2. The nearest ancestor of this file that contains both pyproject.toml and
     src/dgm. Found for any editable install, which is how `uv sync` installs
     this package.

A non-editable install into site-packages has no such ancestor, so $DGM_ROOT is
required there and the error says so.
"""
import os
from pathlib import Path


def _looks_like_course_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "dgm").is_dir()


def _find_course_root() -> Path:
    override = os.environ.get("DGM_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if not _looks_like_course_root(path):
            raise RuntimeError(
                f"$DGM_ROOT={path} does not look like the course root "
                f"(expected pyproject.toml and src/dgm inside it)")
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        if _looks_like_course_root(parent):
            return parent
    raise RuntimeError(
        "Could not locate the course root from "
        f"{here}. This happens when dgm is installed non-editably; set "
        "DGM_ROOT to the directory holding pyproject.toml and src/dgm.")


COURSE_ROOT = _find_course_root()          # .../CIS6270
REPO_ROOT   = COURSE_ROOT.parent           # .../Discrete_Generative_Models

# Shared inputs that are not any one project's.
SHARED_DATA = COURSE_ROOT / "data"         # MNIST lives here
LECTURE_DIR = COURSE_ROOT / "lecture"

# Vendored third-party checkouts, cloned separately and gitignored.
METL_ROOT = Path(os.environ.get("METL_ROOT") or REPO_ROOT / "metl").expanduser()

# A project's artifact directory keeps the name it already has on disk, so
# 83 GB of results and cached weights did not have to move when the code did.
_PROJECT_DIRS = {"project1": "project1_eval"}
DEFAULT_PROJECT = "project1"


def project_dir(project: str = DEFAULT_PROJECT) -> Path:
    """The artifact root for one project: its data, outputs, plots and cache."""
    try:
        name = _PROJECT_DIRS[project]
    except KeyError:
        raise KeyError(f"unknown project {project!r}; "
                       f"known: {', '.join(sorted(_PROJECT_DIRS))}") from None
    return COURSE_ROOT / name


def _sub(name: str, project: str, create: bool) -> Path:
    path = project_dir(project) / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(project: str = DEFAULT_PROJECT, create: bool = False) -> Path:
    """Prepared CSVs, splits, wild-type sequences and fitted oracles."""
    return _sub("data", project, create)


def outputs_dir(project: str = DEFAULT_PROJECT, create: bool = False) -> Path:
    """One subdirectory per run: results.pt and the FASTA files."""
    return _sub("outputs", project, create)


def plots_dir(project: str = DEFAULT_PROJECT, create: bool = False) -> Path:
    """Figures and the metric CSVs behind them."""
    return _sub("plots", project, create)


def cache_dir(project: str = DEFAULT_PROJECT, create: bool = False) -> Path:
    """HuggingFace weight cache, shared by every model and every run."""
    return _sub("cache", project, create)


def logs_dir(project: str = DEFAULT_PROJECT, create: bool = False) -> Path:
    """Sweep driver logs."""
    return _sub("logs", project, create)


def lecture_dir(number: int) -> Path:
    """One lecture's directory, e.g. lecture_dir(3) for the ESM-2 examples."""
    return LECTURE_DIR / f"lecture_{number}"
