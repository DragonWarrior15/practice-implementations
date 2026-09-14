"""Generate the deterministic WAV and labels used in the README example."""

import json
import wave
from pathlib import Path


def main() -> None:
    output_dir = Path(__file__).parent
    sample_rate = 1000
    samples = (
        [0] * 100 + [16_384] * 200 + [0] * 50 + [16_384] * 200 + [0] * 100
    )

    with wave.open(str(output_dir / "audio.wav"), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(
            b"".join(
                sample.to_bytes(2, "little", signed=True) for sample in samples
            )
        )

    labels = {
        "speech_intervals": [
            {"start_sec": 0.10, "end_sec": 0.30},
            {"start_sec": 0.35, "end_sec": 0.55},
        ]
    }
    (output_dir / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
