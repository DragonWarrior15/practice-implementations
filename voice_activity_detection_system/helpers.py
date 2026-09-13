"""Input processor."""

from voice_activity_detection_system.audio_frame_class import (
    AudioFrame,
    FramedAudio,
    FramePrediction,
    SpeechInterval,
)


def frame_audio(
    samples: list[float],
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

    if not samples:
        return FramedAudio(
            sample_rate=sample_rate,
            duration_sec=0.0,
            frames=[],
        )

    frames_list = []

    # now iterate and create AudioFrames
    for i in range(
        len(samples) // samples_per_frame
        + (len(samples) % samples_per_frame != 0)
    ):
        # get the samples for the current frame
        start_idx = i * samples_per_frame
        end_idx = (i + 1) * samples_per_frame

        curr_samples = samples[start_idx:end_idx]

        valid_sample_count = len(curr_samples)

        if len(curr_samples) < samples_per_frame:
            # we need to pad with silent samples
            curr_samples = curr_samples + [0] * (
                samples_per_frame - valid_sample_count
            )

        curr_audio_frame = AudioFrame(
            samples=curr_samples,
            start_sample=start_idx,
            valid_sample_count=valid_sample_count,
        )

        frames_list.append(curr_audio_frame)

    framed_audio = FramedAudio(
        sample_rate=sample_rate,
        duration_sec=len(samples) / sample_rate,
        frames=frames_list,
    )

    return framed_audio


def compute_rms(audio_frame: AudioFrame) -> float:
    """Calculate RMS score for an audio frame."""
    if not audio_frame.samples:
        return 0

    if not audio_frame.valid_sample_count:
        return 0

    # use sum of squared values
    return (
        (1 / audio_frame.valid_sample_count)
        * sum(
            [
                x**2
                for x in audio_frame.samples[: audio_frame.valid_sample_count]
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
    pass