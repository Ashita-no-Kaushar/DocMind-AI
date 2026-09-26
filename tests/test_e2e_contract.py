import json
import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPFILE = ROOT / "Pipfile"
LOCKFILE = ROOT / "Pipfile.lock"
WORKFLOW = ROOT / ".github" / "workflows" / "e2e.yml"


class E2EContractTests(unittest.TestCase):
    def test_playwright_is_a_pinned_direct_development_dependency(self):
        with PIPFILE.open("rb") as handle:
            manifest = tomllib.load(handle)
        self.assertEqual(manifest["dev-packages"]["playwright"], "==1.63.0")
        lock = json.loads(LOCKFILE.read_text(encoding="utf-8"))
        self.assertIn("playwright", lock["develop"])
        self.assertEqual(lock["develop"]["playwright"]["version"], "==1.63.0")
        for name in manifest["packages"]:
            self.assertIn(name, lock["default"], name)
        for name in manifest["dev-packages"]:
            self.assertIn(name, lock["develop"], name)
        for section in ("default", "develop"):
            for name, record in lock[section].items():
                self.assertRegex(record["version"], r"^==", name)
                self.assertTrue(record.get("hashes"), name)

    def test_e2e_workflow_installs_the_lock_and_runs_only_the_browser_module(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        required = (
            "permissions:\n  contents: read",
            "cancel-in-progress: true",
            "timeout-minutes:",
            "actions/checkout@v7.0.1",
            "actions/setup-python@v7.0.0",
            'python-version: "3.13.15"',
            'PIPENV_VERSION: "2026.8.0"',
            "pipenv==${PIPENV_VERSION}",
            "pipenv verify",
            "pipenv install --deploy --dev",
            '--extra-pip-args="--require-hashes"',
            "python -m playwright install --with-deps chromium",
            "pipenv run python -m unittest tests.test_e2e_integration",
        )
        for value in required:
            self.assertIn(value, text)
        self.assertNotIn("unittest discover", text)
        self.assertNotIn("pytest", text)
        self.assertNotIn("eval_harness", text)
        self.assertEqual(
            len(re.findall(r"pipenv run python -m unittest", text)),
            1,
        )

    def test_e2e_workflow_has_no_write_or_publish_surface(self):
        text = WORKFLOW.read_text(encoding="utf-8").casefold()
        for forbidden in (
            "secrets.",
            "id-token: write",
            "contents: write",
            "docker/login-action",
            "git push",
            "git tag",
            "push: true",
            "npm publish",
            "cargo publish",
            "release",
            "publish",
        ):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
