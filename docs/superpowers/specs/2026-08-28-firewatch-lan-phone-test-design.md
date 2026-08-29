# FireWatch LAN Phone Test Design

## Goal

Make the existing FireWatch web interface available for a quick test on a
phone connected to the same Wi-Fi network as the Windows development computer.
The computer will continue to run the Flask server and perform YOLO inference.
This change does not create a permanent public deployment or support access
from cellular data or another network.

## Approach

Keep the existing Flask application, responsive interface, inference service,
and API contract. Add a development launcher that binds Flask to all local
network interfaces on port 5000, disables the interactive debugger, discovers
the computer's usable private IPv4 address, and prints the exact URL to open on
the phone.

The existing direct `python app.py` behavior will remain loopback-only so an
ordinary local launch does not unexpectedly expose the service to the LAN.
LAN access will require the explicit phone-test launcher.

## Components and Data Flow

The launcher will create the existing Flask app through `create_app()`, choose
a private LAN IPv4 address when one is available, print
`http://<lan-address>:5000`, and run the server on `0.0.0.0:5000` with debug and
the reloader disabled.

The phone will load the existing HTML, CSS, and JavaScript from that address.
Image uploads will follow the unchanged `POST /api/analyze` path to the
computer, where the existing validation and YOLO inference run. Results return
to the phone using the current JSON contract and remain in that phone
browser's local history.

## Safety and Failure Handling

The launcher will explain that the URL is intended only for a trusted local
network and that the terminal must remain open. If no usable private IPv4
address can be discovered, it will print a clear fallback explaining how to
find the computer's IPv4 address manually while still starting the server.

If Windows Firewall blocks the first connection, the user may need to allow
Python on private networks. The launcher will not change firewall rules
automatically. The server will use Flask's development server and is strictly
for a short local test, not an internet-facing deployment.

## Testing

Automated tests will verify private-address selection, the no-address fallback,
the printed phone URL, and that the launcher requests LAN binding with debug
and reloader disabled. Existing Flask, inference, and JavaScript tests must
continue to pass.

Manual acceptance consists of starting the launcher, opening its printed URL
on a phone connected to the same Wi-Fi, loading the interface, selecting an
image, receiving an analysis result, and confirming that the mobile layout is
usable. The computer must remain on and the launcher terminal must remain open
during the test.

## Acceptance Criteria

- One documented command starts the phone-test server.
- The command prints a usable same-Wi-Fi URL when a private IPv4 address exists.
- The app is reachable from a phone on the same Wi-Fi without weakening the
  default loopback-only launch.
- Debug mode and the reloader are disabled for LAN serving.
- Existing upload validation, inference results, history, responsive styling,
  and safety messaging remain unchanged.
- Tests cover launcher behavior and all existing automated suites pass.
