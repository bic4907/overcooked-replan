from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(dotenv_path: str | Path | None = None) -> bool:
    """Load project-local environment variables without overriding the shell."""
    path = Path(dotenv_path) if dotenv_path is not None else PROJECT_ROOT / ".env"
    return load_dotenv(dotenv_path=path, override=False)
