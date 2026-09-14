"""Input processor."""

import wave
from pathlib import Path

from voice_activity_detection_system.audio_frame_class import (
    AudioFrame,
    FramedAudio,
    FramePrediction,
    LoadedAudio,
    SpeechInterval,
    VadResult,
)


def frame_audio(
    channels: list[list[float]],
    sample_rate: int,
    frame_duration_ms: float = 20.0,
) -> FramedAudio:
    """Take a list of samples and return the FramedAudio object
    with samples broken into chunks.
    """
    # edge cases
    if sample_rate <= 0:
        raise ValueError(f"sample_rate is {sample_rate}, should be positive")

    if frame_duration_ms <= 0:
        raise ValueError(
            f"frame_duration_ms is {frame_duration_ms}, should be positive"
        )

    # calculate the samples_per_frame
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)

    if samples_per_frame < 1:
        raise ValueError(
            f"samples_per_frame is {samples_per_frame}, check input values"
        )

    if not channels:
        raise ValueError(f"channels cannot be empty, got {channels}")

    if any(len(channel) != len(channels[0]) for channel in channels):
        raise ValueError("Unequal lengths of channels.")

    if not channels[0]:
        return FramedAudio(
            sample_rate=sample_rate,
            duration_sec=0.0,
            frames=[],
        )

    frames_list = []

    len_samples = len(channels[0])

    # now iterate and create AudioFrames
    for i in range(
        len_samples // samples_per_frame
        + (len_samples % samples_per_frame != 0)
    ):
        # get the samples for the current frame
        start_idx = i * samples_per_frame
        end_idx = (i + 1) * samples_per_frame

        curr_channels = []

        for channel in channels:
            curr_samples = channel[start_idx:end_idx]

            valid_sample_count = len(curr_samples)

            if len(curr_samples) < samples_per_frame:
                # we need to pad with silent samples
                curr_samples = curr_samples + [0] * (
                    samples_per_frame - valid_sample_count
                )

            curr_channels.append(curr_samples)

        curr_audio_frame = AudioFrame(
            channels=curr_channels,
            start_sample=start_idx,
            valid_sample_count=valid_sample_count,
        )

        frames_list.append(curr_audio_frame)

    framed_audio = FramedAudio(
        sample_rate=sample_rate,
        duration_sec=len_samples / sample_rate,
        frames=frames_list,
    )

    return framed_audio


def compute_rms(audio_frame: AudioFrame) -> float:
    """Calculate RMS score for an audio frame."""
    if not audio_frame.channels:
        return 0

    if not audio_frame.channels[0]:
        return 0

    if not audio_frame.valid_sample_count:
        return 0

    # use sum of squared values
    return (
        (1 / audio_frame.valid_sample_count)
        * max(
            [
                sum([x**2 for x in channel[: audio_frame.valid_sample_count]])
                for channel in audio_frame.channels
            ]
        )
    ) ** 0.5


def get_speech_label(rms: float, threshold: float) -> bool:
    """Check whether given rms value corresponds to a speech detection."""
    return rms >= threshold


def predict_frame(
    audio_frame: AudioFrame,
    threshold: float,
) -> FramePrediction:
    """Return predictions for a given audio frame."""
    # compute rms value
    rms = compute_rms(audio_frame)

    # calculate is speech label
    is_speech = get_speech_label(rms, threshold)

    # return the prediction object
    return FramePrediction(
        start_sample=audio_frame.start_sample,
        valid_sample_count=audio_frame.valid_sample_count,
        rms=rms,
        is_speech=is_speech,
    )


def predict_frames(
    framed_audio: FramedAudio, threshold: float
) -> list[FramePrediction]:
    """Return predictions for a complete audio clip."""
    if threshold <= 0:
        raise ValueError(f"threshold is {threshold}, must be positive")

    return [predict_frame(x, threshold) for x in framed_audio.frames]


def predictions_to_intervals(
    predictions: list[FramePrediction],
) -> list[SpeechInterval]:
    """Convert predicted frames into merged speech intervals."""
    intervals = []

    start = -1
    end = -1

    for prediction in predictions:
        if not prediction.is_speech:
            # if start and end are valid, add that interval
            if start != -1:
                intervals.append(SpeechInterval(start, end))
                start, end = -1, -1
            else:
                # nothing to do
                continue
        else:
            # check if fresh or in middle
            if start == -1:
                start = prediction.start_sample
                end = prediction.start_sample + prediction.valid_sample_count
            else:
                end = prediction.start_sample + prediction.valid_sample_count

    # once out of the loop, check start again and add a new interval
    # if needed
    if start != -1:
        intervals.append(SpeechInterval(start, end))

    return intervals


def postprocess_intervals(
    intervals: list[SpeechInterval],
    sample_rate: int,
    min_silence_ms: float = 100.0,
    min_speech_ms: float = 100.0,
) -> list[SpeechInterval]:
    """Postprocess the intervals in two stages:

    1. Merge nearby intervals when gap between them is shorter than
    min_silence_ms
    2. Retain the merged intervals whose duration is at least min_speech_ms
    """
    if not intervals:
        return intervals

    # Note the intervals are only speech intervals
    # For any two nearby intervals, we check their start and end samples
    # and if the gap is less than min_silence_ms, merge them
    min_silence_samples = int(sample_rate * min_silence_ms / 1000)

    merged_intervals: list[SpeechInterval] = []

    for interval in intervals:
        # first interval is always added to merged intervals
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            # check the gap between last merged interval and this
            if (
                interval.start_sample - merged_intervals[-1].end_sample
                < min_silence_samples
            ):
                merged_intervals[-1] = SpeechInterval(
                    start_sample=merged_intervals[-1].start_sample,
                    end_sample=interval.end_sample,
                )

            # append as is
            else:
                merged_intervals.append(interval)

    # Now, only retain those intervals that are at least as long as
    # min_speech_ms
    min_speech_samples = int(sample_rate * min_speech_ms / 1000)
    merged_intervals = [
        x
        for x in merged_intervals
        if x.end_sample - x.start_sample >= min_speech_samples
    ]

    return merged_intervals


def deinterleave(values: list[float], channel_count: int) -> list[list[float]]:
    """For stereo channels, the data is interleaved in samples:

    [L0, R0, L1, R1, ...]

    This function returns individual channel values as lists.
    """
    # edge cases
    if channel_count <= 0:
        raise ValueError(
            f"channel count is {channel_count}, must be positive."
        )

    if len(values) % channel_count != 0:
        raise ValueError("length of values is not divisible by channel_count.")

    return [values[idx::channel_count] for idx in range(channel_count)]


def decode_pcm(
    raw_bytes: bytes,
    sample_width_bytes: int,
) -> list[float]:
    """Converts raw bytes into a list of values, each value is scaled between
    -1 and 1.
    """
    allowed_sample_width_bytes = [1, 2, 3, 4]
    if sample_width_bytes not in allowed_sample_width_bytes:
        raise ValueError(
            f"sample_width_bytes must be one of {allowed_sample_width_bytes}, got {sample_width_bytes}"
        )

    len_raw_bytes = len(raw_bytes)
    if len_raw_bytes % sample_width_bytes != 0:
        raise ValueError("raw_bytes must be a multiple of sample_width_bytes.")

    if not raw_bytes:
        return []

    # first convert to a list of values
    values = [
        int.from_bytes(
            raw_bytes[
                sample_width_bytes * idx : sample_width_bytes * (idx + 1)
            ],
            byteorder="little",
            signed=sample_width_bytes != 1,
        )
        for idx in range(len_raw_bytes // sample_width_bytes)
    ]

    # now normalize
    center = 128 if sample_width_bytes == 1 else 0
    scale = (
        128 if sample_width_bytes == 1 else 2 ** (8 * sample_width_bytes - 1)
    )

    return [(v - center) / scale for v in values]


def load_wav(path: str | Path) -> LoadedAudio:
    """Helper to load wav and populate LoadedAudio."""
    if not path:
        raise ValueError("path is not provided.")

    with wave.open(str(path), "rb") as wav_file:
        # rejection case
        if wav_file.getcomptype() != "NONE":
            raise ValueError("Compressed wav format is not supported.")

        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        total_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(total_frames)

        return LoadedAudio(
            channels=deinterleave(
                decode_pcm(audio_bytes, sample_width), channel_count
            ),
            sample_rate=frame_rate,
            channel_count=channel_count,
            sample_width_bytes=sample_width,
            frame_count=total_frames,
            duration_sec=total_frames / frame_rate,
        )


def detect_speech(
    path: str | Path,
    frame_duration_ms: float = 20.0,
    threshold: float = 0.1,
    min_silence_ms: float = 100.0,
    min_speech_ms: float = 100.0,
) -> VadResult:
    """Orchestrator to ingest a wav file and return predictions."""
    # read the file
    loaded_audio = load_wav(path)

    # get the framed audio object containing samples
    framed_audio = frame_audio(
        channels=loaded_audio.channels,
        sample_rate=loaded_audio.sample_rate,
        frame_duration_ms=frame_duration_ms,
    )

    # get predictions
    predictions = predict_frames(
        framed_audio=framed_audio, threshold=threshold
    )

    # process intervals based on predictions
    intervals = predictions_to_intervals(predictions=predictions)

    intervals = postprocess_intervals(
        intervals=intervals,
        sample_rate=loaded_audio.sample_rate,
        min_silence_ms=min_silence_ms,
        min_speech_ms=min_speech_ms,
    )

    return VadResult(
        sample_rate=loaded_audio.sample_rate,
        frame_predictions=predictions,
        speech_intervals=intervals,
    )
