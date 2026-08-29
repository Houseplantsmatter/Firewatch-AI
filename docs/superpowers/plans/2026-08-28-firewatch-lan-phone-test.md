# FireWatch LAN Phone Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit launcher that serves FireWatch to a phone on the same Wi-Fi and prints the URL to open.

**Architecture:** A focused `phone_test.py` launcher will discover a usable private IPv4 address, print same-network instructions, and run the existing Flask application on all LAN interfaces. The normal `app.py` entry point remains loopback-only, and no inference or browser behavior changes.

**Tech Stack:** Python 3.11 standard library (`ipaddress`, `socket`), Flask application factory, `unittest`, PowerShell run instructions

## Global Constraints

- The phone and Windows development computer must be connected to the same trusted Wi-Fi network.
- Keep `python app.py` bound to `127.0.0.1`; LAN exposure must require the explicit launcher.
- Run the LAN server on `0.0.0.0:5000` with debug mode and the reloader disabled.
- Do not change Windows Firewall rules automatically.
- Keep the computer and launcher terminal running throughout the phone test.
- Do not change upload validation, inference, result history, responsive styling, or safety messaging.
- This uses Flask's development server for a short local test only and must not be presented as an internet deployment.

---

### Task 1: Tested LAN phone launcher and instructions

**Files:**
- Create: `ui/phone_test.py`
- Create: `ui/tests/test_phone_test.py`
- Modify: `ui/README.md:17-29`

**Interfaces:**
- Consumes: `app.create_app() -> flask.Flask`
- Produces: `select_private_ipv4(addresses: Iterable[str]) -> str | None`
- Produces: `discover_private_ipv4() -> str | None`
- Produces: `main() -> None`, which prints the phone URL or manual fallback and calls `Flask.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)`

- [ ] **Step 1: Write failing unit tests for address selection and launcher behavior**

Create `ui/tests/test_phone_test.py`:

```python
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
            ["not-an-address", "127.0.0.1", "169.254.2.8", "8.8.8.8", "192.168.1.42"]
        )

        self.assertEqual(address, "192.168.1.42")

    def test_select_private_ipv4_returns_none_without_a_usable_address(self):
        self.assertIsNone(
            PHONE_TEST.select_private_ipv4(["127.0.0.1", "169.254.2.8", "8.8.8.8"])
        )

    def test_main_prints_phone_url_and_uses_safe_lan_server_settings(self):
        flask_app = Mock()
        output = StringIO()

        with (
            patch.object(PHONE_TEST, "discover_private_ipv4", return_value="192.168.1.42"),
            patch.object(PHONE_TEST, "create_app", return_value=flask_app),
            redirect_stdout(output),
        ):
            PHONE_TEST.main()

        self.assertIn("http://192.168.1.42:5000", output.getvalue())
        self.assertIn("trusted Wi-Fi", output.getvalue())
        flask_app.run.assert_called_once_with(
            host="0.0.0.0", port=5000, debug=False, use_reloader=False
        )

    def test_main_prints_manual_fallback_when_discovery_fails(self):
        flask_app = Mock()
        output = StringIO()

        with (
            patch.object(PHONE_TEST, "discover_private_ipv4", return_value=None),
            patch.object(PHONE_TEST, "create_app", return_value=flask_app),
            redirect_stdout(output),
        ):
            PHONE_TEST.main()

        self.assertIn("ipconfig", output.getvalue())
        self.assertIn("http://<IPv4-address>:5000", output.getvalue())
        flask_app.run.assert_called_once_with(
            host="0.0.0.0", port=5000, debug=False, use_reloader=False
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run from `ui`:

```powershell
..\yolo311_env\Scripts\python.exe -m unittest tests.test_phone_test -v
```

Expected: `ERROR` because `phone_test.py` does not exist.

- [ ] **Step 3: Implement the minimal LAN launcher**

Create `ui/phone_test.py`:

```python
import socket
from collections.abc import Iterable
from ipaddress import IPv4Address, ip_address

from app import create_app


HOST = "0.0.0.0"
PORT = 5000


def select_private_ipv4(addresses: Iterable[str]) -> str | None:
    for candidate in addresses:
        try:
            address = ip_address(candidate)
        except ValueError:
            continue
        if (
            isinstance(address, IPv4Address)
            and address.is_private
            and not address.is_loopback
            and not address.is_link_local
            and not address.is_unspecified
        ):
            return str(address)
    return None


def discover_private_ipv4() -> str | None:
    candidates: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
            route_socket.connect(("192.0.2.1", 80))
            candidates.append(route_socket.getsockname()[0])
    except OSError:
        pass

    try:
        candidates.extend(
            entry[4][0]
            for entry in socket.getaddrinfo(
                socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM
            )
        )
    except OSError:
        pass
    return select_private_ipv4(candidates)


def main() -> None:
    address = discover_private_ipv4()
    print("\nFireWatch phone test")
    print("Keep this terminal and computer running during the test.")
    print("Use this only on a trusted Wi-Fi network.")
    if address:
        print(f"\nOn your phone, open: http://{address}:{PORT}")
    else:
        print("\nI could not detect the Wi-Fi IPv4 address automatically.")
        print("Run ipconfig, find the Wi-Fi IPv4 Address, then open:")
        print(f"http://<IPv4-address>:{PORT}")
    print("If Windows asks, allow Python access on private networks only.\n")

    create_app().run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run from `ui`:

```powershell
..\yolo311_env\Scripts\python.exe -m unittest tests.test_phone_test -v
```

Expected: four tests run and `OK`.

- [ ] **Step 5: Document the phone-test command and troubleshooting**

Insert after the local run instructions in `ui/README.md`:

````markdown
## Quick phone test on the same Wi-Fi

Keep the computer and terminal running, connect the phone to the same trusted
Wi-Fi network, and run this command from `ui`:

```powershell
python phone_test.py
```

Open the printed `http://<IPv4-address>:5000` URL on the phone. If Windows
Firewall prompts you, allow Python on **private networks only**. If the page
does not load, confirm both devices are on the same non-guest Wi-Fi; guest
networks often prevent devices from reaching each other.

Press `Ctrl+C` in the terminal when the test is finished. This launcher uses
Flask's development server and is not intended for public internet access.
````

- [ ] **Step 6: Run all automated verification**

Run from `ui`:

```powershell
..\yolo311_env\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
node --test tests/core.test.js
node --check static/js/core.js
node --check static/js/app.js
..\yolo311_env\Scripts\python.exe -m compileall -q .
```

Expected: all Python and JavaScript tests pass, both JavaScript syntax checks exit 0, and compilation exits 0 without output.

- [ ] **Step 7: Smoke-test startup without loading the model**

Run from `ui`:

```powershell
..\yolo311_env\Scripts\python.exe phone_test.py
```

Expected: output includes a URL shaped like `http://192.168.x.x:5000` or `http://10.x.x.x:5000`, Flask reports `Running on all addresses (0.0.0.0)`, and the server stays running until `Ctrl+C`. Opening the printed URL should render FireWatch without loading the YOLO checkpoint until an image is analyzed.

- [ ] **Step 8: Commit the tested launcher**

```powershell
git add ui/phone_test.py ui/tests/test_phone_test.py ui/README.md
git commit -m "feat: add LAN phone test launcher"
```
