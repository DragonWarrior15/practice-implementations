"""Tests for audio framing."""

import unittest

from voice_activity_detection_system.audio_frame_class import AudioFrame
from voice_activity_detection_system.helpers import compute_rms, frame_audio


class FrameAudioTests(unittest.TestCase):
    def test_empty_audio_returns_no_frames(self) -> None:
        result = frame_audio([], sample_rate=16_000)

        self.assertEqual(result.frames, [])
        self.assertEqual(result.duration_sec, 0.0)

    def test_short_audio_is_padded(self) -> None:
        result = frame_audio([0.1] * 100, sample_rate=16_000)

        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].start_sample, 0)
        self.assertEqual(result.frames[0].valid_sample_count, 100)
        self.assertEqual(len(result.frames[0].samples), 320)
        self.assertEqual(result.frames[0].samples[100:], [0] * 220)

    def test_exact_frame_is_not_followed_by_padding_frame(self) -> None:
        result = frame_audio([0.1] * 320, sample_rate=16_000)

        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].valid_sample_count, 320)

    def test_partial_final_frame_tracks_valid_samples(self) -> None:
        result = frame_audio([0.1] * 650, sample_rate=16_000)

        self.assertEqual(len(result.frames), 3)
        self.assertEqual(
            [(frame.start_sample, frame.valid_sample_count) for frame in result.frames],
            [(0, 320), (320, 320), (640, 10)],
        )
        self.assertTrue(all(len(frame.samples) == 320 for frame in result.frames))

    def test_invalid_parameters_are_rejected_for_empty_audio(self) -> None:
        with self.assertRaises(ValueError):
            frame_audio([], sample_rate=0)
        with self.assertRaises(ValueError):
            frame_audio([], sample_rate=16_000, frame_duration_ms=0)
        with self.assertRaises(ValueError):
            frame_audio([], sample_rate=16_000, frame_duration_ms=0.01)


class ComputeRmsTests(unittest.TestCase):
    def test_compute_rms_of_silence_is_zero(self) -> None:
        frame = AudioFrame(
            samples=[0.0, 0.0, 0.0, 0.0],
            start_sample=0,
            valid_sample_count=4,
        )

        self.assertEqual(compute_rms(frame), 0.0)

    def test_compute_rms_ignores_padding(self) -> None:
        frame = AudioFrame(
            samples=[0.5, -0.5, 0.0, 0.0],
            start_sample=0,
            valid_sample_count=2,
        )

        self.assertAlmostEqual(compute_rms(frame), 0.5)


if __name__ == "__main__":
    unittest.main()
