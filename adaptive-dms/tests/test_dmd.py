import json

from adaptive_dms.data.dmd import load_drowsiness_session


def test_load_drowsiness_session_clips_intervals(tmp_path):
    root = tmp_path / "dmd"
    session_dir = root / "gA" / "1" / "s5"
    session_dir.mkdir(parents=True)
    video = session_dir / "sample_rgb_face.mp4"
    video.touch()
    annotation = session_dir / "sample_ann_drowsiness.json"
    annotation.write_text(
        json.dumps(
            {
                "openlabel": {
                    "objects": {"0": {"object_data": {"text": []}}},
                    "contexts": {"0": {"context_data": {"text": []}}},
                    "streams": {
                        "face_camera": {
                            "uri": "gA/1/s5/sample_rgb_face.mp4",
                            "stream_properties": {"total_frames": 5},
                        }
                    },
                    "actions": {
                        "0": {
                            "type": "eyes_state/open",
                            "frame_intervals": [{"frame_start": 0, "frame_end": 2}],
                        },
                        "1": {
                            "type": "eyes_state/close",
                            "frame_intervals": [{"frame_start": 3, "frame_end": 9}],
                        },
                        "2": {
                            "type": "blinks/blinking",
                            "frame_intervals": [{"frame_start": 2, "frame_end": 3}],
                        },
                        "3": {
                            "type": "yawning/Yawning without hand",
                            "frame_intervals": [{"frame_start": 1, "frame_end": 2}],
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    session = load_drowsiness_session(annotation)

    assert session.group == "gA"
    assert session.subject == "1"
    assert session.face_video_path == video.resolve()
    assert session.eye_state == ("open", "open", "open", "close", "close")
    assert session.blinking == (False, False, True, True, False)
    assert session.yawning == (
        "none",
        "yawning_without_hand",
        "yawning_without_hand",
        "none",
        "none",
    )
