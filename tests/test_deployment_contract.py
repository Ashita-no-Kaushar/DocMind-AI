import re
import tomllib
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
PIPFILE = ROOT / "Pipfile"
LOCKFILE = ROOT / "Pipfile.lock"
COMPOSE_FILES = (ROOT / "docker-compose.yml", ROOT / "docker-compose.yml-rocm")
QUALITY_WORKFLOW = ROOT / ".github" / "workflows" / "quality.yml"
DOCKER_WORKFLOW = ROOT / ".github" / "workflows" / "main.yaml"
DEPENDABOT_CONFIG = ROOT / ".github" / "dependabot.yml"
BASE_IMAGE = "python:3.13.15-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26"
PARSER_PACKAGES = {
    "python-docx",
    "docx2txt",
    "python-pptx",
    "openpyxl",
    "xlrd",
    "nbconvert",
    "nbformat",
    "odfpy",
    "extract-msg",
    "olefile",
    "striprtf",
    "defusedxml",
    "ebooklib",
    "html2text",
}


def _read(path):
    return path.read_text(encoding="utf-8")


def _validate_narrow_yaml(text):
    if "\t" in text:
        raise AssertionError("YAML contains a tab indentation character")
    top_level = set()
    block_indent = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indentation = len(line) - len(line.lstrip(" "))
        if indentation % 2:
            raise AssertionError(f"Unsupported YAML indentation on line {line_number}")
        if block_indent is not None:
            if indentation > block_indent:
                continue
            block_indent = None
        content = line.strip()
        if indentation == 0:
            if ":" not in content:
                raise AssertionError(f"Invalid top-level YAML on line {line_number}")
            top_level.add(content.split(":", 1)[0].strip().strip("\"'"))
        elif ":" not in content and not content.startswith("-"):
            raise AssertionError(f"Invalid YAML mapping on line {line_number}")
        if content.endswith(("|", ">", "|-", ">-", "|+", ">+")):
            block_indent = indentation
    if (
        "services" not in top_level
        and "version" not in top_level
        and "name" not in top_level
    ):
        raise AssertionError("YAML does not contain a recognized top-level structure")
    return top_level


def _validate_yaml(path):
    text = _read(path)
    if yaml is not None:
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise AssertionError(f"{path.name} is not a YAML mapping")
    else:
        _validate_narrow_yaml(text)
    return text


class LockContractTests(unittest.TestCase):
    def test_pipfile_has_pinned_direct_dependencies_and_python_requirement(self):
        with PIPFILE.open("rb") as handle:
            manifest = tomllib.load(handle)
        packages = manifest["packages"]
        dev_packages = manifest["dev-packages"]
        self.assertEqual(manifest["requires"]["python_version"], "3.13")
        self.assertTrue(packages)
        self.assertTrue(dev_packages)
        self.assertTrue(all("*" not in value for value in packages.values()))
        self.assertTrue(all("*" not in value for value in dev_packages.values()))
        self.assertTrue(PARSER_PACKAGES.issubset(packages))

    def test_lock_contains_hashed_default_and_development_entries(self):
        self.assertTrue(LOCKFILE.is_file())
        lock_text = _read(LOCKFILE)
        self.assertRegex(lock_text, r'"sha256": "[0-9a-f]{64}"')
        import json

        lock = json.loads(lock_text)
        self.assertRegex(lock["_meta"]["hash"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(lock["_meta"]["requires"]["python_version"], "3.13")
        for section in ("default", "develop"):
            self.assertIsInstance(lock[section], dict)
            self.assertTrue(lock[section])
            for name, record in lock[section].items():
                self.assertRegex(record["version"], r"^==")
                self.assertTrue(record.get("hashes"), name)
                self.assertTrue(
                    all(
                        re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                        for value in record["hashes"]
                    ),
                    name,
                )
        all_packages = {**lock["default"], **lock["develop"]}
        self.assertFalse(any(name.lower().startswith("torch") for name in all_packages))
        self.assertFalse(
            any(name.lower().startswith("transformers") for name in all_packages)
        )
        self.assertTrue(PARSER_PACKAGES.issubset(all_packages))

    def test_lock_includes_development_tools(self):
        import json

        lock = json.loads(_read(LOCKFILE))
        self.assertTrue({"ruff", "black", "pytest"}.issubset(lock["develop"]))


class DockerContractTests(unittest.TestCase):
    def test_base_image_is_exact_and_multiarchitecture_digest_pinned(self):
        dockerfile = _read(DOCKERFILE)
        self.assertIn(f"FROM {BASE_IMAGE} AS base", dockerfile)
        self.assertNotRegex(dockerfile, r"(?m)^FROM\s+python:3\.13(?:\.\d+)?-slim\s*$")

    def test_lock_install_is_default_only_and_tooling_is_pinned(self):
        dockerfile = _read(DOCKERFILE)
        self.assertIn("Pipfile.lock", dockerfile)
        self.assertIn("pipenv sync", dockerfile)
        self.assertIn("--categories default", dockerfile)
        self.assertIn("--no-deprecated", dockerfile)
        self.assertIn("PIPENV_VERSION=2026.8.0", dockerfile)
        self.assertNotIn("--dev", dockerfile)
        self.assertIn("RUN --mount=type=cache", dockerfile)
        self.assertIn("pip check", dockerfile)

    def test_runtime_is_non_root_unbuffered_and_health_checked(self):
        dockerfile = _read(DOCKERFILE)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn('useradd --uid "${APP_UID}"', dockerfile)
        self.assertIn('groupadd --gid "${APP_GID}"', dockerfile)
        self.assertIn("ENV PYTHONUNBUFFERED=1", dockerfile)
        self.assertIn("chown root:root /home/appuser", dockerfile)
        self.assertIn("HEALTHCHECK", dockerfile)
        self.assertIn("/_stcore/health", dockerfile)
        self.assertIn("127.0.0.1:8501", dockerfile)
        self.assertNotIn("USER root", dockerfile)

    def test_runtime_paths_and_secrets_are_bounded(self):
        dockerfile = _read(DOCKERFILE)
        for path in (
            "/home/appuser/data",
            "/home/appuser/.index_cache",
            "/home/appuser/.cache",
            "/home/appuser/logs",
        ):
            self.assertIn(path, dockerfile)
        self.assertNotRegex(dockerfile, r"(?i)(api[_-]?key|password|secret|token)\s*=")
        self.assertIn(".env*", _read(DOCKERIGNORE))
        self.assertIn(".streamlit/secrets.toml", _read(DOCKERIGNORE))
        self.assertNotIn("torch", dockerfile.lower())
        self.assertNotIn("transformers", dockerfile.lower())


class ComposeContractTests(unittest.TestCase):
    def test_compose_files_are_yaml_and_share_the_safe_runtime_contract(self):
        for path in COMPOSE_FILES:
            with self.subTest(path=path.name):
                text = _validate_yaml(path)
                self.assertIn('"127.0.0.1:8501:8501/tcp"', text)
                self.assertNotRegex(text, r'(?m)^\s*-\s*["\']?0\.0\.0\.0:')
                self.assertIn('user: "10001:10001"', text)
                self.assertIn("read_only: true", text)
                self.assertIn("no-new-privileges:true", text)
                self.assertIn("cap_drop:", text)
                self.assertIn("healthcheck:", text)
                self.assertIn("/_stcore/health", text)
                self.assertIn('DOCMIND_ALLOW_REMOTE_BIND: "true"', text)
                self.assertIn('DOCMIND_BIND_ADDRESS: "0.0.0.0"', text)
                self.assertIn("DOCMIND_INGESTION_LOCK_PATH:", text)
                self.assertIn("DOCMIND_LOG_FILE:", text)
                self.assertIn("DOCMIND_R2R_STATE_DIR:", text)
                self.assertIn("/home/appuser/data", text)
                self.assertIn("/home/appuser/.index_cache", text)
                self.assertIn("/home/appuser/logs", text)
                self.assertIn("mem_limit:", text)
                self.assertIn("cpus:", text)
                self.assertIn("pids_limit:", text)
                self.assertNotRegex(text, r"(?m)^\s*(gpus|device_ids|runtime):")
                self.assertNotIn("api_key", text.lower())
                self.assertNotIn("password", text.lower())
                self.assertNotIn("secret", text.lower())

    def test_rocm_file_is_application_only_and_has_no_gpu_runtime(self):
        text = _validate_yaml(ROOT / "docker-compose.yml-rocm")
        self.assertIn("com.docmind.container.role: application-only", text)
        self.assertIn("com.docmind.ollama.runtime: external", text)
        self.assertNotRegex(text, r"(?m)^\s*ollama:\s*$")
        self.assertNotIn("runtime: rocm", text.lower())
        self.assertNotIn("gpus:", text)
        self.assertNotIn("deploy:", text)


class WorkflowContractTests(unittest.TestCase):
    def test_quality_workflow_uses_locked_deploy_mode_and_full_checks(self):
        text = _validate_yaml(QUALITY_WORKFLOW)
        self.assertIn("pipenv verify", text)
        self.assertIn("pipenv install --deploy --dev", text)
        self.assertIn('--extra-pip-args="--require-hashes"', text)
        self.assertNotIn(
            "pipenv install --dev", text.replace("pipenv install --deploy --dev", "")
        )
        self.assertNotIn("pip install --upgrade", text)
        self.assertIn("python -m unittest discover -s tests", text)
        self.assertIn("python -m compileall", text)
        self.assertIn("python -m pip check", text)
        self.assertIn("pipenv run ruff check .", text)
        self.assertIn("pipenv run black --check .", text)
        self.assertNotIn("ruff check . --select E9,F63,F7,F82", text)
        self.assertIn("eval_harness.py --mock", text)
        self.assertIn("contents: read", text)
        self.assertIn("cancel-in-progress: true", text)
        self.assertIn("timeout-minutes:", text)
        self.assertIn("actions/checkout@v7.0.1", text)
        self.assertIn("actions/setup-python@v7.0.0", text)
        self.assertIn('python-version: "3.13.15"', text)
        self.assertIn("pipenv==${PIPENV_VERSION}", text)
        self.assertIn('PIPENV_VERSION: "2026.8.0"', text)

    def test_docker_workflow_is_build_only_and_has_no_release_surface(self):
        text = _validate_yaml(DOCKER_WORKFLOW)
        self.assertIn("docker/build-push-action@v7.4.0", text)
        self.assertIn("docker/setup-buildx-action@v4.4.1", text)
        self.assertIn("actions/checkout@v7.0.1", text)
        self.assertIn("platforms: linux/amd64", text)
        self.assertIn("push: false", text)
        self.assertIn("contents: read", text)
        self.assertIn("cancel-in-progress: true", text)
        self.assertIn("timeout-minutes:", text)
        for forbidden in (
            "secrets.",
            "docker/login-action",
            "act10ns/slack",
            "SLACK_WEBHOOK_URL",
            "DOCKER_ACCESS_TOKEN",
            "DOCKER_USERNAME",
            "git tag",
            "git push",
            "tags:",
            "push: true",
            "dockerhub",
        ):
            self.assertNotIn(forbidden, text.lower())

    def test_dependabot_configuration_remains_valid(self):
        text = _validate_yaml(DEPENDABOT_CONFIG)
        self.assertIn("version: 2", text)
        self.assertIn('package-ecosystem: "pip"', text)
        self.assertIn('package-ecosystem: "github-actions"', text)
        self.assertIn('interval: "weekly"', text)


if __name__ == "__main__":
    unittest.main()
