"""Tests for audio framing."""

import unittest
from pathlib import Path
import struct
import tempfile
import wave

from voice_activity_detection_system.audio_frame_class import (
    AudioFrame,
    FramePrediction,
    FramedAudio,
    SpeechInterval,
)
from voice_activity_detection_system.helpers import (
    compute_rms,
    decode_pcm,
    deinterleave,
    frame_audio,
    load_wav,
    postprocess_intervals,
    predict_frame,
    predict_frames,
    predictions_to_intervals,
)


class FrameAudioTests(unittest.TestCase):
    def test_empty_audio_returns_no_frames(self) -> None:
        result = frame_audio([[]], sample_rate=16_000)

        self.assertEqual(result.frames, [])
        self.assertEqual(result.duration_sec, 0.0)

    def test_short_audio_is_padded(self) -> None:
        result = frame_audio([[0.1] * 100], sample_rate=16_000)

        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].start_sample, 0)
        self.assertEqual(result.frames[0].valid_sample_count, 100)
        self.assertEqual(len(result.frames[0].channels), 1)
        self.assertEqual(len(result.frames[0].channels[0]), 320)
        self.assertEqual(result.frames[0].channels[0][100:], [0] * 220)

    def test_exact_frame_is_not_followed_by_padding_frame(self) -> None:
        result = frame_audio([[0.1] * 320], sample_rate=16_000)

        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].valid_sample_count, 320)

    def test_partial_final_frame_tracks_valid_samples(self) -> None:
        result = frame_audio([[0.1] * 650], sample_rate=16_000)

        self.assertEqual(len(result.frames), 3)
        self.assertEqual(
            [(frame.start_sample, frame.valid_sample_count) for frame in result.frames],
            [(0, 320), (320, 320), (640, 10)],
        )
        self.assertTrue(
            all(len(frame.channels[0]) == 320 for frame in result.frames)
        )

    def test_stereo_channels_are_framed_and_padded_together(self) -> None:
        result = frame_audio(
            [[0.1] * 330, [0.2] * 330],
            sample_rate=16_000,
        )

        self.assertEqual(len(result.frames), 2)
        self.assertEqual(len(result.frames[1].channels), 2)
        self.assertEqual(result.frames[1].valid_sample_count, 10)
        self.assertTrue(
            all(len(channel) == 320 for channel in result.frames[1].channels)
        )

    def test_no_channels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            frame_audio([], sample_rate=16_000)

    def test_unequal_channel_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            frame_audio([[0.1] * 320, [0.2] * 319], sample_rate=16_000)

    def test_invalid_parameters_are_rejected_for_empty_audio(self) -> None:
        with self.assertRaises(ValueError):
            frame_audio([[]], sample_rate=0)
        with self.assertRaises(ValueError):
            frame_audio([[]], sample_rate=16_000, frame_duration_ms=0)
        with self.assertRaises(ValueError):
            frame_audio([[]], sample_rate=16_000, frame_duration_ms=0.01)


class ComputeRmsTests(unittest.TestCase):
    def test_compute_rms_of_silence_is_zero(self) -> None:
        frame = AudioFrame(
            channels=[[0.0, 0.0, 0.0, 0.0]],
            start_sample=0,
            valid_sample_count=4,
        )

        self.assertEqual(compute_rms(frame), 0.0)

    def test_compute_rms_ignores_padding(self) -> None:
        frame = AudioFrame(
            channels=[[0.5, -0.5, 0.0, 0.0]],
            start_sample=0,
            valid_sample_count=2,
        )

        self.assertAlmostEqual(compute_rms(frame), 0.5)

    def test_compute_rms_uses_loudest_channel(self) -> None:
        frame = AudioFrame(
            channels=[
                [0.1, -0.1, 0.1, -0.1],
                [0.5, -0.5, 0.5, -0.5],
            ],
            start_sample=0,
            valid_sample_count=4,
        )

        self.assertAlmostEqual(compute_rms(frame), 0.5)


class PredictionTests(unittest.TestCase):
    def test_predict_frame_preserves_metadata_and_labels_threshold(self) -> None:
        frame = AudioFrame(
            channels=[[0.1, -0.1]],
            start_sample=320,
            valid_sample_count=2,
        )

        prediction = predict_frame(frame, threshold=0.1)

        self.assertEqual(prediction.start_sample, 320)
        self.assertEqual(prediction.valid_sample_count, 2)
        self.assertAlmostEqual(prediction.rms, 0.1)
        self.assertIs(prediction.is_speech, True)

    def test_predict_frames_handles_empty_audio(self) -> None:
        framed_audio = FramedAudio(
            sample_rate=16_000,
            duration_sec=0.0,
            frames=[],
        )

        self.assertEqual(predict_frames(framed_audio, threshold=0.1), [])

    def test_predict_frames_rejects_nonpositive_threshold(self) -> None:
        framed_audio = FramedAudio(16_000, 0.0, [])

        with self.assertRaises(ValueError):
            predict_frames(framed_audio, threshold=0.0)


class IntervalTests(unittest.TestCase):
    @staticmethod
    def prediction(
        start_sample: int,
        is_speech: bool,
        valid_sample_count: int = 320,
    ) -> FramePrediction:
        return FramePrediction(
            start_sample=start_sample,
            valid_sample_count=valid_sample_count,
            rms=0.2 if is_speech else 0.0,
            is_speech=is_speech,
        )

    def test_predictions_to_intervals_merges_adjacent_speech(self) -> None:
        predictions = [
            self.prediction(0, False),
            self.prediction(320, True),
            self.prediction(640, True),
            self.prediction(960, False),
        ]

        self.assertEqual(
            predictions_to_intervals(predictions),
            [SpeechInterval(320, 960)],
        )

    def test_predictions_to_intervals_keeps_separate_regions(self) -> None:
        predictions = [
            self.prediction(0, True),
            self.prediction(320, False),
            self.prediction(640, True),
            self.prediction(960, False),
        ]

        self.assertEqual(
            predictions_to_intervals(predictions),
            [SpeechInterval(0, 320), SpeechInterval(640, 960)],
        )

    def test_predictions_to_intervals_uses_partial_final_length(self) -> None:
        predictions = [
            self.prediction(0, False),
            self.prediction(320, True),
            self.prediction(640, True, valid_sample_count=10),
        ]

        self.assertEqual(
            predictions_to_intervals(predictions),
            [SpeechInterval(320, 650)],
        )

    def test_postprocessing_fills_before_filtering(self) -> None:
        intervals = [SpeechInterval(0, 1280), SpeechInterval(1600, 2880)]

        self.assertEqual(
            postprocess_intervals(intervals, sample_rate=16_000),
            [SpeechInterval(0, 2880)],
        )

    def test_postprocessing_respects_exact_boundaries(self) -> None:
        intervals = [
            SpeechInterval(0, 1600),
            SpeechInterval(3200, 4800),
        ]

        self.assertEqual(
            postprocess_intervals(intervals, sample_rate=16_000),
            intervals,
        )
        self.assertEqual(
            postprocess_intervals(
                [SpeechInterval(0, 1584)],
                sample_rate=16_000,
            ),
            [],
        )


class PcmTests(unittest.TestCase):
    def test_deinterleave_stereo(self) -> None:
        self.assertEqual(
            deinterleave([1.0, 10.0, 2.0, 20.0], channel_count=2),
            [[1.0, 2.0], [10.0, 20.0]],
        )

    def test_deinterleave_rejects_incomplete_sample_frame(self) -> None:
        with self.assertRaises(ValueError):
            deinterleave([1.0, 2.0, 3.0], channel_count=2)

    def test_decode_pcm_8_bit_boundaries(self) -> None:
        self.assertEqual(
            decode_pcm(bytes([0, 128, 255]), sample_width_bytes=1),
            [-1.0, 0.0, 127 / 128],
        )

    def test_decode_pcm_signed_width_boundaries(self) -> None:
        for width in (2, 3, 4):
            with self.subTest(width=width):
                bits = 8 * width
                values = [-(2 ** (bits - 1)), 0, 2 ** (bits - 1) - 1]
                raw = b"".join(
                    value.to_bytes(width, "little", signed=True)
                    for value in values
                )

                decoded = decode_pcm(raw, sample_width_bytes=width)

                self.assertEqual(decoded[0], -1.0)
                self.assertEqual(decoded[1], 0.0)
                self.assertAlmostEqual(
                    decoded[2],
                    (2 ** (bits - 1) - 1) / 2 ** (bits - 1),
                )

    def test_decode_pcm_rejects_misaligned_payload(self) -> None:
        with self.assertRaises(ValueError):
            decode_pcm(b"\x00", sample_width_bytes=2)


class LoadWavTests(unittest.TestCase):
    def test_load_wav_reads_stereo_metadata_and_deinterleaves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            interleaved = [1000, 2000, -1000, -2000]
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                wav_file.writeframes(
                    struct.pack("<hhhh", *interleaved)
                )

            loaded = load_wav(path)

        self.assertEqual(loaded.sample_rate, 8000)
        self.assertEqual(loaded.channel_count, 2)
        self.assertEqual(loaded.sample_width_bytes, 2)
        self.assertEqual(loaded.frame_count, 2)
        self.assertEqual(loaded.duration_sec, 2 / 8000)
        self.assertEqual(
            loaded.channels,
            [
                [1000 / 32768, -1000 / 32768],
                [2000 / 32768, -2000 / 32768],
            ],
        )

    def test_load_wav_normalizes_unsigned_8_bit_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mono.wav"
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(1)
                wav_file.setframerate(16_000)
                wav_file.writeframes(bytes([0, 128, 255]))

            loaded = load_wav(path)

        self.assertEqual(loaded.channels, [[-1.0, 0.0, 127 / 128]])
        self.assertEqual(loaded.frame_count, 3)

if __name__ == "__main__":
    unittest.main()
