"""Tests for the CLI entry point."""

from typer.testing import CliRunner

from court_vision.cli import app

runner = CliRunner()


def test_version_command():
    """Version command prints the current version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help_shows_commands():
    """Top-level help lists available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "process" in result.output
    assert "version" in result.output
