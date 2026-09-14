"""Dataclasses for the system."""

from dataclasses import dataclass


@dataclass
class AudioFrame:
    """A single audio frame.

    channels: list[list[float]]
    start_sample: int
    valid_sample_count: int
    """

    channels: list[list[float]]
    start_sample: int
    valid_sample_count: int


@dataclass
class FramedAudio:
    """A single audio clip, storing multiple frames.

    sample_rate: int
    duration_sec: float
    frames: list[AudioFrame]
    """

    sample_rate: int
    duration_sec: float
    frames: list[AudioFrame]


@dataclass
class FramePrediction:
    """Prediction for a single frame.

    start_sample: int
    valid_sample_count: int
    rms: float
    is_speech: bool
    """

    start_sample: int
    valid_sample_count: int
    rms: float
    is_speech: bool


@dataclass
class SpeechInterval:
    """Speech interval containing the start and end sample indices.

    start_sample: int
    end_sample: int
    """

    start_sample: int
    end_sample: int


@dataclass
class LoadedAudio:
    """Loaded audio class carries extracted information from a WAV clip.

    channels: list[list[float]]
    sample_rate: int
    channel_count: int
    sample_width_bytes: int
    frame_count: int
    duration_sec: float
    """

    channels: list[list[float]]
    sample_rate: int
    channel_count: int
    sample_width_bytes: int
    frame_count: int
    duration_sec: float


@dataclass
class VadResult:
    """Output result of our detection algorithm.

    sample_rate: int
    frame_predictions: list[FramePrediction]
    speech_intervals: list[SpeechInterval]
    threshold_used: float
    """

    sample_rate: int
    frame_predictions: list[FramePrediction]
    speech_intervals: list[SpeechInterval]
    threshold_used: float = 0.0


@dataclass
class EvaluationMetrics:
    """Classification evaluation Metrics.

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
