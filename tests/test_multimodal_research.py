import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research import multimodal

ROOT = Path(__file__).resolve().parents[1]


class PrerequisiteTests(unittest.TestCase):
    def test_no_rasterizer_is_reported_when_none_is_installed(self):
        with patch.object(multimodal, "available_rasterizer", return_value=None):
            self.assertIsNone(multimodal.available_rasterizer())
            status = multimodal.probe(None, 150)
            self.assertIn(multimodal.SKIP_NO_RASTERIZER, status["missing"])

    def test_no_vision_model_is_reported_when_none_is_configured(self):
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("DOCMIND_VISION")
        }
        with patch.dict(os.environ, environment, clear=True):
            found = multimodal.available_vision_model()
        if found is not None:
            self.assertNotIn("pixel", found.lower())
            self.assertNotIn("hash", found.lower())
            self.assertNotIn("statistic", found.lower())
        else:
            self.assertIsNone(found)

    def test_probe_never_claims_a_measurement(self):
        status = multimodal.probe(None, 150)
        self.assertEqual(status["status"], "skipped")
        self.assertIn("No image-resolution measurement", status["note"])
        self.assertEqual(status["planned_dpi_ladder"], [72, 150, 300])

    def test_a_configured_vision_model_is_detected(self):
        with patch.dict(os.environ, {"DOCMIND_VISION_MODEL": "qwen2-vl"}, clear=False):
            self.assertEqual(multimodal.available_vision_model(), "ollama:qwen2-vl")

    def test_requested_dpi_is_clamped(self):
        self.assertEqual(multimodal.probe(None, 5)["requested_dpi"], 36)
        self.assertEqual(multimodal.probe(None, 5000)["requested_dpi"], 600)


class NoFabricationTests(unittest.TestCase):
    def test_rasterizing_without_a_rasterizer_raises(self):
        with (
            patch.object(multimodal, "available_rasterizer", return_value=None),
            self.assertRaises(multimodal.PrerequisiteMissing),
        ):
            multimodal.rasterize("missing.pdf", 150)

    def test_embedding_pages_refuses_instead_of_returning_a_placeholder(self):
        with self.assertRaises(multimodal.PrerequisiteMissing):
            multimodal.embed_pages([object()])

    def test_the_harness_never_defines_a_synthetic_image_encoder(self):
        source = (ROOT / "research" / "multimodal.py").read_text(encoding="utf-8")
        for banned in ("def _fake_embed", "def _pixel_hash", "random.uniform"):
            self.assertNotIn(banned, source)


class StatusFileTests(unittest.TestCase):
    def test_status_is_written_with_the_honest_outcome(self):
        target = Path(tempfile.mkdtemp(prefix="docmind_multimodal_"))
        self.addCleanup(shutil.rmtree, target, True)
        code = multimodal.main(["--out-dir", str(target), "--quiet"])
        self.assertEqual(code, 0)
        payload = json.loads(
            (target / "results" / "multimodal.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["status"], "skipped")
        self.assertIn("generated_at", payload)
        self.assertTrue(payload["missing"])


if __name__ == "__main__":
    unittest.main()
