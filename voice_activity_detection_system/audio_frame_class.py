"""Dataclasses for the system."""

from dataclasses import dataclass


@dataclass
class AudioFrame:
    """A single audio frame.

    samples: list[float]
    start_sample: int
    valid_sample_count: int
    """

    samples: list[float]
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
