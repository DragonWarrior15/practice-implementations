"""Command-line evaluation for the energy-based VAD baseline."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from voice_activity_detection_system.audio_frame_class import (
    FramePrediction,
    SpeechInterval,
)
from voice_activity_detection_system.helpers import (
    check_overlap_with_threshold,
    detect_speech,
    evaluate_predictions,
    result_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VAD predictions against labeled intervals."
    )
    parser.add_argument("wav_path", type=Path, help="Input PCM WAV file")
    parser.add_argument(
        "labels_path",
        type=Path,
        help="JSON file containing speech intervals in seconds",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("fixed", "adaptive"),
        default="fixed",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.1)
    parser.add_argument("--adaptive-multiplier", type=float, default=3.0)
    parser.add_argument("--adaptive-min-threshold", type=float, default=0.005)
    parser.add_argument("--frame-duration-ms", type=float, default=20.0)
    parser.add_argument("--min-silence-ms", type=float, default=100.0)
    parser.add_argument("--min-speech-ms", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.labels_path.open("r", encoding="utf-8") as labels_file:
        labels_data = json.load(labels_file)

    start_time = time.perf_counter()
    result = detect_speech(
        args.wav_path,
        frame_duration_ms=args.frame_duration_ms,
        threshold=args.fixed_threshold,
        min_silence_ms=args.min_silence_ms,
        min_speech_ms=args.min_speech_ms,
        threshold_mode=args.threshold_mode,
        adaptive_multiplier=args.adaptive_multiplier,
        adaptive_min_threshold=args.adaptive_min_threshold,
    )
    runtime_sec = time.perf_counter() - start_time

    labelled_intervals = [
        SpeechInterval(
            start_sample=round(item["start_sec"] * result.sample_rate),
            end_sample=round(item["end_sec"] * result.sample_rate),
        )
        for item in labels_data["speech_intervals"]
    ]

    raw_metrics = evaluate_predictions(
        result.frame_predictions,
        labelled_intervals,
    )
    postprocessed_predictions = [
        FramePrediction(
            start_sample=prediction.start_sample,
            valid_sample_count=prediction.valid_sample_count,
            rms=prediction.rms,
            is_speech=check_overlap_with_threshold(
                prediction,
                result.speech_intervals,
                overlap_threshold=0.5,
            ),
        )
        for prediction in result.frame_predictions
    ]
    postprocessed_metrics = evaluate_predictions(
        postprocessed_predictions,
        labelled_intervals,
    )

    if result.frame_predictions:
        last = result.frame_predictions[-1]
        audio_duration_sec = (
            last.start_sample + last.valid_sample_count
        ) / result.sample_rate
    else:
        audio_duration_sec = 0.0
    real_time_factor = (
        runtime_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
    )

    serialized = result_to_dict(result)
    output = {
        "threshold_mode": args.threshold_mode,
        "threshold_used": result.threshold_used,
        "raw_frame_metrics": asdict(raw_metrics),
        "postprocessed_frame_metrics": asdict(postprocessed_metrics),
        "runtime_sec": runtime_sec,
        "real_time_factor": real_time_factor,
        "speech_intervals": serialized["speech_intervals"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
