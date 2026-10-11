"""Replay entrypoint checks without cameras, GPU, or recorded data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from config.config_loader import ConfigLoader
from config_live.config_loader import ConfigLoader as LiveConfigLoader

PROJECT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_all_scene_overlays_disable_rotation(self):
        for loader in (ConfigLoader, LiveConfigLoader):
            for overlay in loader.get_available_configs():
                with self.subTest(overlay=overlay):
                    config = loader(overlay)
                    self.assertIs(config.get("alignment.replay_apply_rotation"), False)
                    self.assertIn("/processed_data/", config.get("paths.record_dir"))

    def test_single_entrypoint_launches_only_requested_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / "processed_data" / "saved_data_l"
            for kind, ext in (("event", "dat"), ("rgb", "mp4")):
                (data / kind).mkdir(parents=True)
                (data / kind / f"5.{ext}").touch()
            interpreter = folder / "capture-python"
            interpreter.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "print('CAPTURE=' + json.dumps(dict(argv=sys.argv[1:], "
                "record_dir=os.environ.get('RECORD_DIR'), config=os.environ.get('ACTIVE_CONFIG_FILE'))))\n"
            )
            interpreter.chmod(0o700)
            env = os.environ.copy()
            env.update(DATA_ROOT=str(folder / "processed_data"), DATA_DIR="",
                       TEST_MODE="Event", TASK="normal", RECORD_INDEX="0",
                       MPLCONFIGDIR=str(folder / "mpl"), SAVE_CSV="0")
            env.pop("ACTIVE_CONFIG_FILE", None)
            for explicit in (None, "config_live/overlays/config_l.yaml",
                             str(PROJECT / "config_live/overlays/config_l.yaml")):
                with self.subTest(explicit=explicit):
                    if explicit is not None:
                        env["ACTIVE_CONFIG_FILE"] = explicit
                    result = subprocess.run(
                        ["bash", str(PROJECT / "run_record.sh"), "--record_index", "5",
                         "--python", str(interpreter), "--no-gui"],
                        cwd=folder, env=env, text=True, capture_output=True, check=True,
                    )
                    captures = [line for line in result.stdout.splitlines() if line.startswith("CAPTURE=")]
                    self.assertEqual(len(captures), 1)
                    call = json.loads(captures[0].split("=", 1)[1])
                    self.assertEqual(call["argv"], ["record.py", "--record_index", "5",
                                                  "--test_mode", "Event", "--task", "normal"])
                    self.assertEqual(call["record_dir"], str(data))
                    expected = (PROJECT / explicit if explicit is not None
                                else PROJECT / "config/overlays/config_l_event.yaml")
                    self.assertEqual(call["config"], str(expected))
                    self.assertIn("already rotated; RGB alignment only", result.stdout)

    def test_missing_explicit_config_is_not_replaced_by_shell(self):
        env = dict(os.environ, ACTIVE_CONFIG_FILE="config/config_l.yaml",
                   TEST_MODE="Event", TASK="normal", RECORD_INDEX="0", PYTHON_BIN=sys.executable)
        result = subprocess.run(
            ["bash", str(PROJECT / "run_record.sh"), "--dry-run"],
            env=env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("config not found", result.stderr)
        self.assertIn("/config/config_l.yaml", result.stderr)


    def test_batch_entrypoint_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = Path(tmp) / "capture-python"
            interpreter.write_text(
                "#!" + sys.executable + "\n"
                "import json, sys\n"
                "print(json.dumps(sys.argv[1:]))\n"
            )
            interpreter.chmod(0o700)
            env = dict(os.environ, PYTHON_BIN=str(interpreter))
            result = subprocess.run(
                ["bash", str(PROJECT / "run_record.sh"), "--batch", "--limit", "1", "--dry-run"],
                env=env, text=True, capture_output=True, check=True,
            )
            self.assertEqual(json.loads(result.stdout),
                             ["-u", str(PROJECT / "tools/evaluation_batch.py"),
                              "--gui", "--limit", "1", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
