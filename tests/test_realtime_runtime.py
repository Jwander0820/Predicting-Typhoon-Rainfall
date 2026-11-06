import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from typhoon_rainfall_realtime import runtime


class RuntimeTests(unittest.TestCase):
    def test_absent_local_runtime_preserves_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "not-installed"
            with patch.object(runtime, "CUDA_DIRECTORY", missing), patch.dict(os.environ, {"PATH": "original"}):
                result = runtime.configure_runtime()
                self.assertFalse(result["local_cuda_registered"])
                self.assertEqual(os.environ["PATH"], "original")

    def test_registration_is_idempotent_and_preserves_existing_memory_preference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            handle = Mock()
            with patch.object(runtime, "CUDA_DIRECTORY", directory), patch.object(runtime, "_DLL_HANDLES", {}), \
                    patch.object(runtime.os, "name", "nt"), \
                    patch.object(runtime.os, "add_dll_directory", return_value=handle) as add_directory, \
                    patch.dict(os.environ, {"PATH": "original", "TF_FORCE_GPU_ALLOW_GROWTH": "false"}):
                first = runtime.configure_runtime()
                second = runtime.configure_runtime()
                self.assertEqual(first, second)
                self.assertTrue(first["local_cuda_registered"])
                add_directory.assert_called_once_with(str(directory.resolve()))
                self.assertEqual(os.environ["PATH"].split(os.pathsep), [str(directory.resolve()), "original"])
                self.assertEqual(os.environ["TF_FORCE_GPU_ALLOW_GROWTH"], "false")
                self.assertIs(runtime._DLL_HANDLES[str(directory.resolve())], handle)


if __name__ == "__main__":
    unittest.main()
