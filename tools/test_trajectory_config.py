"""Verify YAML-to-fitter wiring and numerical behavior without camera hardware."""
import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import config.config_loader as cfg
from tracking.fitting import ransac_fit as ransac
from tracking.fitting.parabola_fitter import MonoRansacDrag


PROJECT = Path(__file__).resolve().parents[1]


def trajectory(p, v, t, gravity, k):
    """Independent synthetic measurements in meters and seconds."""
    t = np.asarray(t)[:, None]
    g = np.array([0., 0., gravity])
    if k == 0:
        return p + v * t + 0.5 * g * t ** 2
    return p + (v - g / k) * (-np.expm1(-k * t)) / k + g * t / k


class TrajectoryConfigTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([3., 0.1, 2.])
        self.v = np.array([-5., 0.2, 1.])
        self.t = np.linspace(0., 0.5, 25)

    def make_fitter(self, **kwargs):
        return MonoRansacDrag({}, np.eye(3), np.eye(3), **kwargs)

    def fit(self, fitter, points):
        fitter.set_data(np.column_stack([points, self.t]))
        module = sys.modules[type(fitter).__module__]
        # Isolate projection; run actual RANSAC, refinement and prediction.
        with patch.object(module, 'pixel_to_world', return_value=points), \
                patch.object(np.random, 'choice', return_value=np.array([0, 1])):
            return fitter.get_params()

    def fitter_from_worker(self):
        """Execute actual worker initialization through fitter construction."""
        path = PROJECT / 'tracking/fitting/fitting_process.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'landing_prediction')
        index = next(i for i, node in enumerate(function.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'fitter'
                             for t in node.targets))
        function.body = function.body[:index + 1] + [ast.Return(value=ast.Name(id='fitter', ctx=ast.Load()))]
        code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = dict(Config=cfg.Config, __file__=str(path))
        exec(compile(code, str(path), 'exec'), namespace)
        return namespace['landing_prediction'](**{arg.arg: None for arg in function.args.args})

    def test_yaml_stacks_and_overlay_reach_actual_worker_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(cfg, 'PROJECT_ROOT', Path(tmp)), \
                patch.dict(os.environ, {}, clear=True):
            for stack in ('config', 'config_live'):
                base = Path(tmp) / stack
                (base / 'overlays').mkdir(parents=True)
                (base / 'camera_config.yaml').write_text(
                    (PROJECT / stack / 'camera_config.yaml').read_text())
                (base / 'tracking_config.yaml').write_text(
                    'trajectory:\n  gravity: -3.5\n  inlier_threshold: 0.35\n')
                overlay = base / 'overlays/custom.yaml'
                for override, threshold in (('{}', 0.35),
                                            ('trajectory: {inlier_threshold: 0.05}', 0.05)):
                    with self.subTest(stack=stack, threshold=threshold), \
                            patch.object(cfg, '_config', None):
                        overlay.write_text(override)
                        os.environ['ACTIVE_CONFIG_FILE'] = str(overlay)
                        fitter = self.fitter_from_worker()
                        self.assertEqual(fitter.gravity, -3.5)
                        self.assertEqual(fitter.inlier_threshold, threshold)
                        points = trajectory(self.p, self.v, self.t, -3.5, 0.05)
                        points[-1, 1] += 0.2
                        result = self.fit(fitter, points)
                        self.assertEqual(bool(result['inliers'][-1]), threshold == 0.35)
                        if threshold == 0.05:
                            impact_t = -np.log1p(0.05 * self.p[0] / self.v[0]) / 0.05
                            expected = trajectory(self.p, self.v, [impact_t], -3.5, 0.05)[0]
                            np.testing.assert_allclose(fitter.predict(), expected, atol=1e-6)

    def test_custom_gravity_in_ransac_drag_and_zero_drag_fallback(self):
        for k in (0., 0.05):
            with self.subTest(k=k):
                points = trajectory(self.p, self.v, self.t, -2., k)
                with patch.object(np.random, 'choice', return_value=np.array([0, 1])):
                    p, v, inliers, _ = ransac.adaptive_ransac_parabola_3d(
                        points, self.t, gravity=-2., use_drag=True, drag_k=k,
                        inlier_threshold=0.001)
                np.testing.assert_allclose(p, self.p, atol=1e-8)
                np.testing.assert_allclose(v, self.v, atol=1e-8)
                self.assertTrue(inliers.all())

    def test_custom_gravity_in_plain_parabola_ransac(self):
        points = trajectory(self.p, self.v, self.t, -2., 0.)
        with patch.object(np.random, 'choice', return_value=np.array([0, 1])):
            p, v, inliers, _ = ransac.adaptive_ransac_parabola_3d(
                points, self.t, gravity=-2., inlier_threshold=0.001)
        np.testing.assert_allclose(p, self.p, atol=1e-8)
        np.testing.assert_allclose(v, self.v, atol=1e-8)
        self.assertTrue(inliers.all())

    def test_fitter_instances_use_independent_gravity_for_prediction(self):
        for k in (0., 0.05):
            for gravity in (-9.81, -2., 0.):
                with self.subTest(k=k, gravity=gravity):
                    fitter = self.make_fitter(gravity=gravity)
                    fitter.predeicted_params = {'params': dict(
                        x0=3., y0=0.1, z0=2., vx=-5., vy=0.2, vz=1., k=k)}
                    impact_t = 0.6 if k == 0 else -np.log1p(k * 3. / -5.) / k
                    expected = trajectory(self.p, self.v, [impact_t], gravity, k)[0]
                    np.testing.assert_allclose(fitter.predict(), expected, atol=1e-8)
        np.testing.assert_array_equal(ransac.GRAVITY, [0., 0., -9.81])

    def test_default_parameters_preserve_trajectory_and_fit(self):
        default = self.make_fitter()
        explicit = self.make_fitter(gravity=-9.81, inlier_threshold=0.35)
        points = trajectory(self.p, self.v, self.t, -9.81, 0.05)
        first = self.fit(default, points)
        second = self.fit(explicit, points)
        np.testing.assert_allclose(list(first['params'].values()), list(second['params'].values()))
        np.testing.assert_array_equal(first['inliers'], second['inliers'])
        np.testing.assert_allclose(default.predict(), explicit.predict())

    def test_invalid_parameters_fail_at_construction(self):
        for kwargs in ({'gravity': float('nan')}, {'gravity': float('inf')},
                       {'inlier_threshold': 0}, {'inlier_threshold': -0.1},
                       {'inlier_threshold': float('nan')}, {'inlier_threshold': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'trajectory'):
                self.make_fitter(**kwargs)


if __name__ == '__main__':
    unittest.main()
