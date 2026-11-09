"""OS launcher software checks; no native inference or privacy efficacy claim."""
from pathlib import Path

import pytest

from scripts import run_pose_offline as launcher


def test_unknown_platform_fails_without_launch(monkeypatch):
    monkeypatch.setattr(launcher.sys, 'platform', 'linux')
    with pytest.raises(ValueError, match='verified OS network isolation'):
        launcher.command(['--video', 'unused'])


def test_mac_command_preserves_arguments_and_network_profile(monkeypatch):
    monkeypatch.setattr(launcher.sys, 'platform', 'darwin')
    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    args = ['--video', 'file name;$(unsafe).mp4', '--rights', 'local research']
    result = launcher.command(args)
    assert result[:3] == ['/usr/bin/sandbox-exec', '-p', '(version 1)(allow default)(deny network*)']
    assert result[-len(args):] == args
    assert result[3] == str(launcher.ROOT / '.tools/pose-runtime/bin/python')
    assert result[4] == str(launcher.ROOT / 'scripts/extract_pose.py')


def test_missing_isolation_fails(monkeypatch):
    monkeypatch.setattr(launcher.sys, 'platform', 'darwin')
    monkeypatch.setattr(Path, 'is_file', lambda self: False)
    with pytest.raises(ValueError, match='sandbox-exec'):
        launcher.command([])
