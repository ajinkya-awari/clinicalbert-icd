from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]

_HAS_POWERSHELL = shutil.which("powershell.exe") is not None


def run_codex_guard(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / ".codex" / "hooks" / "pre_tool_use.ps1"),
            command,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_codex_stdin(payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / ".codex" / "hooks" / "pre_tool_use.ps1"),
        ],
        cwd=ROOT,
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )


def run_codex_stop() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / ".codex" / "hooks" / "stop.ps1")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def run_claude_guard(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", ".claude/hooks/pre-tool-use.sh", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_claude_stdin(payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", ".claude/hooks/pre-tool-use.sh"],
        cwd=ROOT,
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.drive:
        return str(resolved)
    drive = resolved.drive[0].lower()
    relative = resolved.as_posix()[3:]
    return f"/mnt/{drive}/{relative}"


def run_pre_commit(temp_root: Path) -> subprocess.CompletedProcess[str]:
    command = " ".join(
        (
            f"GIT_DIR={shlex.quote(to_wsl_path(temp_root / '.git'))}",
            f"GIT_WORK_TREE={shlex.quote(to_wsl_path(temp_root))}",
            "bash .claude/hooks/pre-commit.sh",
        )
    )
    return subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class CommandGuardTests(unittest.TestCase):
    SAFE_COMMANDS = (
        "rg subject_id tests",
        "Get-Content AGENTS.md",
        "nvidia-smi --query-gpu=name --format=csv,noheader",
        "rg from_pretrained src",
        "rg NOTEEVENTS tests",
        "poetry show",
        "uv --version",
        "conda --version",
        "pipx list",
    )
    BLOCKED_COMMANDS = (
        "python -m unittest tests.test_split_contract -v",
        "python -m pytest tests",
        "python --version",
        "python3 -m unittest tests.test_training_gate -v",
        "py -m unittest tests.test_split_contract",
        "pytest tests",
        "python -m pip install transformers",
        "python demo.py from_pretrained",
        "python train.py --config local",
        "kaggle kernels push -p synthetic",
        "curl https://example.invalid/export",
        "Invoke-WebRequest https://example.invalid/export",
        "wandb login",
        "huggingface-cli upload org/model artifact.bin",
        "git push origin main",
        "send-mail report.txt",
        "python publish.py",
        "python deploy.py",
        "python -m training",
        "python -m clinicalbert_icd.training",
        "python3 train.py",
        "pip3 install synthetic-package",
        "/usr/bin/curl https://example.invalid/export",
        "git.exe push origin main",
        "kaggle datasets download synthetic-dataset",
        "python -u train.py --config local",
        "git --no-optional-locks push origin main",
        "cat NOTEEVENTS.csv",
        "python -m pip download synthetic-package",
        "python -X dev train.py --config local",
        "head NOTEEVENTS.csv",
        "ssh example.invalid",
        "pip --no-input download synthetic-package",
        "python -m pip --no-cache-dir download synthetic-package",
        "grep synthetic NOTEEVENTS.csv",
        "findstr synthetic ADMISSIONS.csv",
        "Write-Output synthetic; pip --no-input download synthetic-package",
        "printf synthetic | grep synthetic NOTEEVENTS.csv",
        "pip --timeout 5 download synthetic-package",
        "python -m pip --proxy https://proxy.invalid download synthetic-package",
        "Get-Content 'review/patient notes.csv'",
        "poetry install",
        "uv sync",
        "iwr https://example.invalid/export",
        "irm https://example.invalid/export",
        "poetry lock",
        "conda create --name synthetic-environment",
        "mamba update synthetic-package",
        "micromamba env create -f synthetic-environment.yml",
        "pipx install synthetic-package",
        "pipx run synthetic-package",
        "uv run synthetic-command",
        "uv tool install synthetic-package",
        "uv pip download synthetic-package",
        "poetry --no-interaction install",
        "uv --offline sync",
        "conda --json create --name synthetic-environment",
    )

    @unittest.skipUnless(_HAS_POWERSHELL, "requires Windows/powershell.exe")
    def test_codex_guard_allows_local_read_only_and_synthetic_commands(self) -> None:
        for command in self.SAFE_COMMANDS:
            with self.subTest(command=command):
                result = run_codex_guard(command)
                self.assertEqual(0, result.returncode, result.stderr)

    @unittest.skipUnless(_HAS_POWERSHELL, "requires Windows/powershell.exe")
    def test_codex_guard_blocks_restricted_commands(self) -> None:
        for command in self.BLOCKED_COMMANDS:
            with self.subTest(command=command):
                result = run_codex_guard(command)
                self.assertEqual(2, result.returncode)
                self.assertIn("requires approval", result.stderr.lower())

    @unittest.skipUnless(_HAS_POWERSHELL, "requires Windows/powershell.exe")
    def test_codex_stop_emits_machine_readable_json(self) -> None:
        result = run_codex_stop()
        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual("ok", payload["status"])
        self.assertEqual("08-f3-clinicalbert-icd", payload["project"])
        self.assertEqual("approval_required", payload["restricted_data"])

    def test_claude_guard_allows_local_read_only_and_synthetic_commands(self) -> None:
        for command in self.SAFE_COMMANDS:
            with self.subTest(command=command):
                result = run_claude_guard(command)
                self.assertEqual(0, result.returncode, result.stderr)

    def test_claude_guard_blocks_restricted_commands(self) -> None:
        for command in self.BLOCKED_COMMANDS:
            with self.subTest(command=command):
                result = run_claude_guard(command)
                self.assertEqual(2, result.returncode)
                self.assertIn("requires approval", result.stderr.lower())

    def test_claude_settings_wire_command_guard_before_bash(self) -> None:
        settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
        hooks = settings["hooks"]["PreToolUse"]
        bash_hooks = [entry for entry in hooks if entry["matcher"] == "Bash"]
        self.assertEqual(1, len(bash_hooks))
        self.assertEqual(
            "bash .claude/hooks/pre-tool-use.sh",
            bash_hooks[0]["hooks"][0]["command"],
        )

    @unittest.skipUnless(_HAS_POWERSHELL, "requires Windows/powershell.exe")
    def test_guards_block_actual_style_json_stdin(self) -> None:
        for command in (
            "curl https://example.invalid/export",
            "git push origin main",
            "echo $(curl https://example.invalid/export)",
            '"curl" https://example.invalid/export',
            "echo `curl https://example.invalid/export`",
        ):
            payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
            with self.subTest(guard="codex", command=command):
                self.assertEqual(2, run_codex_stdin(payload).returncode)
            with self.subTest(guard="claude", command=command):
                self.assertEqual(2, run_claude_stdin(payload).returncode)

    @unittest.skipUnless(_HAS_POWERSHELL, "requires Windows/powershell.exe")
    def test_guards_join_multiple_arguments_before_inspection(self) -> None:
        codex = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / ".codex" / "hooks" / "pre_tool_use.ps1"),
                "git.exe",
                "push",
                "origin",
                "main",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        claude = subprocess.run(
            ["bash", ".claude/hooks/pre-tool-use.sh", "git.exe", "push", "origin", "main"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(2, codex.returncode)
        self.assertEqual(2, claude.returncode)

    def test_pre_commit_fails_closed_without_git_metadata(self) -> None:
        result = subprocess.run(
            ["bash", ".claude/hooks/pre-commit.sh"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("not a git worktree", result.stderr.lower())

    def test_pre_commit_allows_staged_data_source_package(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hook-test-", dir=ROOT) as temp_name:
            temp_root = Path(temp_name)
            source = temp_root / "src" / "clinicalbert_icd" / "data" / "example.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 'synthetic'\n", encoding="utf-8")
            subprocess.run(["git", "init", "--quiet"], cwd=temp_root, check=True)
            subprocess.run(["git", "add", "--", source.relative_to(temp_root)], cwd=temp_root, check=True)
            result = run_pre_commit(temp_root)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_pre_commit_blocks_staged_sensitive_paths_at_any_depth(self) -> None:
        for relative_path in (
            Path("ADMISSIONS.csv"),
            Path("patient_notes.csv"),
            Path("review") / "ADMISSIONS.csv",
            Path("review") / "patient_notes.csv",
        ):
            with self.subTest(relative_path=relative_path):
                with tempfile.TemporaryDirectory(prefix="hook-test-", dir=ROOT) as temp_name:
                    temp_root = Path(temp_name)
                    path = temp_root / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("synthetic-only\n", encoding="utf-8")
                    subprocess.run(["git", "init", "--quiet"], cwd=temp_root, check=True)
                    subprocess.run(["git", "add", "--", relative_path], cwd=temp_root, check=True)
                    result = run_pre_commit(temp_root)
                self.assertEqual(2, result.returncode)

    def test_pre_commit_blocks_staged_secret_value(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hook-test-", dir=ROOT) as temp_name:
            temp_root = Path(temp_name)
            path = temp_root / "config" / "example.txt"
            path.parent.mkdir(parents=True)
            canary = "".join(
                ("HF_", "TOKEN", "='synthetic_", "secret_value_", "123456'\n")
            )
            path.write_text(canary, encoding="utf-8")
            subprocess.run(["git", "init", "--quiet"], cwd=temp_root, check=True)
            subprocess.run(["git", "add", "--", path.relative_to(temp_root)], cwd=temp_root, check=True)
            result = run_pre_commit(temp_root)
        self.assertEqual(2, result.returncode)


if __name__ == "__main__":
    unittest.main()
