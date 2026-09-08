"""Tests for configuration loading, overrides and the stage tables."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from court_vision.config import (
    STAGE_DEPS,
    STAGE_PARAMS,
    PipelineConfig,
    PipelineSettings,
    apply_overrides,
    load_config,
)


def test_load_config_from_yaml(sample_config_path: Path):
    """Loads config from a YAML file."""
    config = load_config(sample_config_path)
    assert isinstance(config, PipelineConfig)
    assert config.pipeline.target_resolution == [1280, 720]
    assert config.pipeline.ball_confidence_threshold == 0.7
    assert config.pipeline.ball_max_gap_s == 0.5
    assert config.pipeline.contact_method == "proximity"
    assert config.device == "auto"


def test_load_config_defaults():
    """Returns default config when no file is provided."""
    config = load_config(None)
    assert isinstance(config, PipelineConfig)
    assert config.pipeline.target_resolution == [1280, 720]
    assert config.pipeline.ball_confidence_threshold == 0.3
    assert config.pipeline.ball_detection_method == "wasb"
    assert config.pipeline.contact_method == "trajectory"
    assert config.pipeline.outcome_method == "auto"
    assert config.device == "auto"


def test_load_config_missing_file_returns_defaults(tmp_path: Path):
    """A path that does not exist falls back to defaults instead of raising."""
    config = load_config(tmp_path / "nope.yaml")
    assert config == PipelineConfig()


def test_config_output_directory_default():
    """Default output directory is 'output/'."""
    config = load_config(None)
    assert config.output.directory == "output/"


def test_config_device_override():
    """Device field accepts explicit device strings."""
    config = load_config(None)
    config.device = "cpu"
    assert config.device == "cpu"


class TestBallDetectionMethod:
    def test_rejects_hsv(self):
        """The colour-based HSV detector was removed; 'hsv' is no longer valid."""
        with pytest.raises(ValidationError):
            PipelineSettings(ball_detection_method="hsv")

    @pytest.mark.parametrize("method", ["wasb", "tracknet"])
    def test_accepts_neural_methods(self, method: str):
        assert PipelineSettings(ball_detection_method=method).ball_detection_method == method


class TestStrideKnobs:
    def test_strides_default_to_auto(self):
        s = PipelineSettings()
        assert s.ball_frame_step == "auto"
        assert s.player_detect_stride == "auto"

    def test_explicit_integer_strides_accepted(self):
        s = PipelineSettings(ball_frame_step=2, player_detect_stride=3)
        assert s.ball_frame_step == 2
        assert s.player_detect_stride == 3

    def test_other_strings_rejected(self):
        with pytest.raises(ValidationError):
            PipelineSettings(ball_frame_step="fast")


class TestApplyOverrides:
    def test_pipeline_prefixed_key(self):
        config = apply_overrides(PipelineConfig(), {"pipeline.ball_confidence_threshold": 0.5})
        assert config.pipeline.ball_confidence_threshold == 0.5

    def test_bare_key_means_pipeline(self):
        config = apply_overrides(PipelineConfig(), {"contact_method": "proximity"})
        assert config.pipeline.contact_method == "proximity"

    def test_device_and_output_keys(self):
        config = apply_overrides(PipelineConfig(), {"device": "cpu", "output.directory": "elsewhere/"})
        assert config.device == "cpu"
        assert config.output.directory == "elsewhere/"

    def test_returns_copy_and_leaves_original_untouched(self):
        original = PipelineConfig()
        updated = apply_overrides(original, {"ball_confidence_threshold": 0.9, "device": "cpu"})
        assert updated.pipeline.ball_confidence_threshold == 0.9
        assert original.pipeline.ball_confidence_threshold == 0.3
        assert original.device == "auto"
        assert updated is not original

    def test_multiple_overrides_applied_together(self):
        config = apply_overrides(
            PipelineConfig(),
            {"ball_frame_step": 2, "pipeline.outcome_method": "last_hitter", "near_player_hand": "left"},
        )
        assert config.pipeline.ball_frame_step == 2
        assert config.pipeline.outcome_method == "last_hitter"
        assert config.pipeline.near_player_hand == "left"

    def test_invalid_value_is_validated(self):
        with pytest.raises(ValidationError):
            apply_overrides(PipelineConfig(), {"ball_detection_method": "hsv"})

    def test_empty_overrides_is_identity(self):
        config = PipelineConfig()
        assert apply_overrides(config, {}) == config


class TestStageTables:
    def test_every_stage_param_is_a_settings_field(self):
        fields = set(PipelineSettings.model_fields)
        for stage, knobs in STAGE_PARAMS.items():
            missing = [k for k in knobs if k not in fields]
            assert not missing, f"stage {stage!r} references unknown knobs {missing}"

    def test_stage_deps_reference_known_stages(self):
        for stage, deps in STAGE_DEPS.items():
            assert stage in STAGE_PARAMS
            for d in deps:
                assert d in STAGE_PARAMS, f"{stage} depends on unknown stage {d!r}"
            assert stage not in deps

    def test_same_stage_set_in_both_tables(self):
        assert set(STAGE_DEPS) == set(STAGE_PARAMS)

    def test_shots_depends_on_ball_and_players(self):
        assert "ball_post" in STAGE_DEPS["shots"]
        assert "players" in STAGE_DEPS["shots"]
        assert STAGE_DEPS["ingest"] == ()

    def test_neural_knobs_live_in_their_own_stage(self):
        """Changing a Stage-5 knob must not be attributed to the neural stages."""
        for knob in ("contact_min_gap_s", "outcome_method", "slice_drop_ratio"):
            assert knob in STAGE_PARAMS["shots"]
            assert knob not in STAGE_PARAMS["ball"]
            assert knob not in STAGE_PARAMS["players"]
        assert "ball_detection_method" in STAGE_PARAMS["ball"]
        assert "player_model" in STAGE_PARAMS["players"]
