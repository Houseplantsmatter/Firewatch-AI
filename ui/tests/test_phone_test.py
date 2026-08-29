import importlib.util
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch


UI_DIR = Path(__file__).resolve().parents[1]
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))
SPEC = importlib.util.spec_from_file_location(
    "firewatch_phone_test", UI_DIR / "phone_test.py"
)
PHONE_TEST = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PHONE_TEST
SPEC.loader.exec_module(PHONE_TEST)


class PhoneTestLauncherTests(unittest.TestCase):
    def test_select_private_ipv4_skips_unusable_and_public_addresses(self):
        address = PHONE_TEST.select_private_ipv4(
            [
                "not-an-address",
                "127.0.0.1",
                "169.254.2.8",
                "8.8.8.8",
                "192.168.1.42",
            ]
        )

        self.assertEqual(address, "192.168.1.42")

    def test_select_private_ipv4_returns_none_without_a_usable_address(self):
        self.assertIsNone(
            PHONE_TEST.select_private_ipv4(
                ["127.0.0.1", "169.254.2.8", "8.8.8.8"]
            )
        )

    def test_main_prints_phone_url_and_uses_safe_lan_server_settings(self):
        flask_app = Mock()
        output = StringIO()

        with patch.object(
            PHONE_TEST, "discover_private_ipv4", return_value="192.168.1.42"
        ), patch.object(
            PHONE_TEST, "create_app", return_value=flask_app
        ), redirect_stdout(output):
            PHONE_TEST.main()

        self.assertIn("http://192.168.1.42:5000", output.getvalue())
        self.assertIn("trusted Wi-Fi", output.getvalue())
        flask_app.run.assert_called_once_with(
            host="0.0.0.0", port=5000, debug=False, use_reloader=False
        )

    def test_main_prints_manual_fallback_when_discovery_fails(self):
        flask_app = Mock()
        output = StringIO()

        with patch.object(
            PHONE_TEST, "discover_private_ipv4", return_value=None
        ), patch.object(
            PHONE_TEST, "create_app", return_value=flask_app
        ), redirect_stdout(output):
            PHONE_TEST.main()

        self.assertIn("ipconfig", output.getvalue())
        self.assertIn("http://<IPv4-address>:5000", output.getvalue())
        flask_app.run.assert_called_once_with(
            host="0.0.0.0", port=5000, debug=False, use_reloader=False
        )


if __name__ == "__main__":
    unittest.main()
