"""Strict path selection and failure-before-startup regression tests."""
import argparse
import ast
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import config.config_loader as cfg
import config_live.config_loader as live

REPO = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(cfg, "PROJECT_ROOT", self.root))
        self.stack.enter_context(patch.object(cfg, "_config", None))
        self.stack.enter_context(patch.object(live, "_config", None))
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        for name in ("config", "config_live"):
            base = self.root / name
            (base / "overlays").mkdir(parents=True)
            (base / "camera_config.yaml").write_text(
                f"source: {name}\nnested: {{camera: 1, winner: camera}}\n"
            )
            (base / "tracking_config.yaml").write_text(
                "nested: {tracking: 2, winner: tracking}\n"
            )
            (base / "overlays/config_l.yaml").write_text(
                f"overlay: {name}\nnested: {{winner: overlay}}\n"
            )

    def test_explicit_relative_and_absolute_live_path_select_live_bases(self):
        for path in ("config_live/overlays/config_l.yaml",
                     str(self.root / "config_live/overlays/config_l.yaml")):
            with self.subTest(path=path):
                cfg._config = None
                os.environ["ACTIVE_CONFIG_FILE"] = path
                old = Path.cwd()
                try:
                    os.chdir(self.root / "config")
                    loaded = cfg.get_config()
                finally:
                    os.chdir(old)
                self.assertEqual(loaded.get("source"), "config_live")
                self.assertEqual(loaded.get("overlay"), "config_live")
                self.assertEqual(loaded.camera_config_path.parent, self.root / "config_live")
                self.assertEqual(loaded.tracking_config_path.parent, self.root / "config_live")

    def test_missing_explicit_path_never_uses_same_basename(self):
        for path in ("missing/config_l.yaml", "config/config_l.yaml"):
            with self.subTest(path=path):
                os.environ["ACTIVE_CONFIG_FILE"] = path
                with self.assertRaises(FileNotFoundError) as error:
                    cfg.get_config()
                self.assertIn(str(self.root / path), str(error.exception))
                self.assertIsNone(cfg._config)

    def test_empty_explicit_path_is_an_error(self):
        for path in ("", "   "):
            os.environ["ACTIVE_CONFIG_FILE"] = path
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                cfg.get_config()

    def test_every_layer_is_required(self):
        for name in ("camera_config.yaml", "tracking_config.yaml", "overlays/config_l.yaml"):
            path = self.root / "config" / name
            text = path.read_text()
            path.unlink()
            try:
                with self.subTest(name=name), self.assertRaises(FileNotFoundError) as error:
                    cfg.ConfigLoader()
                self.assertIn(str(path), str(error.exception))
            finally:
                path.write_text(text)

    def test_invalid_yaml_and_top_level_types_report_the_file(self):
        for name in ("camera_config.yaml", "tracking_config.yaml", "overlays/config_l.yaml"):
            path = self.root / "config" / name
            original = path.read_text()
            for value in ("", "null", "[]", "42", "key: [unterminated"):
                path.write_text(value)
                with self.subTest(name=name, value=value), self.assertRaises(ValueError) as error:
                    cfg.ConfigLoader()
                self.assertIn(str(path), str(error.exception))
            path.write_text(original)

    def test_merge_order_and_intentionally_empty_overlay(self):
        loaded = cfg.ConfigLoader()
        self.assertEqual(loaded.get("nested"), {"camera": 1, "tracking": 2, "winner": "overlay"})
        (self.root / "config/overlays/config_l.yaml").write_text("{}")
        loaded.reload()
        self.assertEqual(loaded.get("nested.winner"), "tracking")

    def test_switch_changes_whole_stack_and_failures_preserve_state(self):
        loaded = cfg.ConfigLoader()
        loaded.switch_config("config_live/overlays/config_l.yaml")
        before = (loaded.config.copy(), loaded.config_path, loaded.camera_config_path,
                  loaded.tracking_config_path, loaded.current_config_file)
        with self.assertRaises(FileNotFoundError):
            loaded.switch_config("missing/config_l.yaml")
        self.assertEqual(before, (loaded.config, loaded.config_path, loaded.camera_config_path,
                                  loaded.tracking_config_path, loaded.current_config_file))
        loaded.config_path.write_text("[]")
        with self.assertRaises(ValueError):
            loaded.reload()
        self.assertEqual(loaded.config, before[0])

    def test_external_overlay_uses_loader_default_stack(self):
        path = self.root / "external.yaml"
        path.write_text("external: true")
        self.assertEqual(cfg.ConfigLoader(path).get("source"), "config")
        self.assertEqual(live.ConfigLoader(path).get("source"), "config_live")
        self.assertEqual(live.ConfigLoader("config/overlays/config_l.yaml").get("source"), "config")

    def test_cli_selection_and_initialize_export_exact_path(self):
        path = self.root / "config/overlays/config_d_event.yaml"
        path.write_text("scene: dark")
        with patch.object(sys, "argv", ["record.py", "--task=dark", "--test_mode=Event"]):
            self.assertEqual(cfg.get_config().config_path, path)
        loaded = cfg.initialize_config("dark", "Event")
        self.assertEqual(os.environ["ACTIVE_CONFIG_FILE"], str(path))
        self.assertEqual(os.environ["ACTIVE_TASK"], "dark")
        cfg._config = None
        self.assertEqual(cfg.get_config().config, loaded.config)

    def test_initialize_respects_explicit_path(self):
        os.environ["ACTIVE_CONFIG_FILE"] = "config_live/overlays/config_l.yaml"
        loaded = cfg.initialize_config("dark", "Event")
        self.assertEqual(loaded.get("source"), "config_live")
        self.assertEqual(os.environ["ACTIVE_CONFIG_FILE"], str(loaded.config_path))

    def test_main_entrypoints_fail_before_resources_or_subprocesses(self):
        # Execute the real startup blocks without importing hardware dependencies.
        for filename, mode in (("record.py", "Event"), ("live_real.py", "Event"),
                               ("tracking.py", "Event"), ("record_compare.py", "All")):
            tree = ast.parse((REPO / filename).read_text())
            main = next(node for node in tree.body if isinstance(node, ast.If)
                        and isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__")
            definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name in ("resolve_config_file", "set_active_config")]
            module = ast.Module(body=definitions + main.body, type_ignores=[])
            blocked = Mock(side_effect=AssertionError("Resources must not be created"))
            ns = dict(argparse=argparse, os=os, sys=sys, __file__=str(REPO/filename),
                      torch=Mock(device=blocked), subprocess=Mock(run=blocked))
            os.environ["ACTIVE_CONFIG_FILE"] = "missing/config_l.yaml"
            error_output = io.StringIO()
            with self.subTest(entry=filename), patch.object(sys, "argv",
                    [filename, "--task", "normal", "--test_mode", mode]), \
                 redirect_stderr(error_output), redirect_stdout(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                exec(compile(module, filename, "exec"), ns)
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Configuration error", error_output.getvalue())
            self.assertIn("missing/config_l.yaml", error_output.getvalue())
            blocked.assert_not_called()

    def test_batch_validates_before_workers_and_output_creation(self):
        import evaluation_batch as batch
        os.environ["ACTIVE_CONFIG_FILE"] = "missing/config_l.yaml"
        output = self.root / "run"
        with patch.object(sys, "argv", ["batch", "--data-root", str(self.root),
                                       "--output-root", str(output)]), \
             patch.object(batch, "run_sample") as run, redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as error:
            batch.main()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
