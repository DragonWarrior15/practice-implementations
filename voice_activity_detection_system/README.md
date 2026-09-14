# Voice Activity Detection baseline

A dependency-free Python baseline that detects speech-like regions in PCM WAV
files using short-frame RMS energy. It returns raw frame predictions and merged,
post-processed speech intervals.

## Design

The offline pipeline loads and normalizes PCM, preserves channels separately,
splits audio into 20 ms frames, uses the maximum per-channel RMS, applies a fixed
or adaptive threshold, and post-processes speech intervals. Short silence gaps
are bridged before speech regions shorter than the minimum duration are removed.

The final partial frame is zero-padded, but RMS and timestamps use only its real
samples. Exact sample indices are converted to seconds only for JSON output.

## Supported audio

- Uncompressed integer PCM WAV with 8-, 16-, 24-, or 32-bit samples
- Mono or multichannel audio
- Any positive sample rate; no resampling is performed

Compressed and floating-point WAV files are outside this baseline.

## Run tests

From the directory containing `voice_activity_detection_system`:

```bash
PYTHONPATH=. python3 -m unittest discover \
  -s voice_activity_detection_system/tests -v
```

## Evaluate

The repository includes `fixtures/audio.wav` and matching
`fixtures/labels.json`. Regenerate them with:

```bash
python3 voice_activity_detection_system/fixtures/generate_fixture.py
```

Ground-truth JSON uses actual speech regions in seconds:

```json
{
  "speech_intervals": [
    {"start_sec": 0.10, "end_sec": 0.30},
    {"start_sec": 0.35, "end_sec": 0.55}
  ]
}
```

```bash
PYTHONPATH=. python3 -m voice_activity_detection_system.evaluate \
  voice_activity_detection_system/fixtures/audio.wav \
  voice_activity_detection_system/fixtures/labels.json \
  --threshold-mode fixed --fixed-threshold 0.1
```

Use `--threshold-mode adaptive` to estimate the threshold from the 20th-percentile
frame RMS noise floor. The output reports the selected threshold, raw and
post-processed frame metrics, intervals, end-to-end runtime, and real-time factor.

Undefined precision, recall, or F1 values are reported as `0.0`. A frame is
ground-truth speech when at least half of its valid samples overlap annotations.

## Baseline example

On the deterministic 650 ms fixture, two 200 ms signal regions separated by 50
ms silence become one interval because the gap is shorter than 100 ms:

```json
{"start_sec": 0.1, "end_sec": 0.56}
```

The 10 ms boundary extension is expected 20 ms frame quantization.

See [`example_output.json`](example_output.json) for a complete compact
evaluation result containing the selected threshold, raw and post-processed
metrics, and detected speech intervals. Run `evaluate.py` to generate fresh
output for another WAV file; `result_to_dict(...)` additionally exposes every
frame's RMS value and speech classification.
