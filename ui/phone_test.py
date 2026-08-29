import socket
from ipaddress import IPv4Address, ip_address
from typing import Iterable, List, Optional

from app import create_app


HOST = "0.0.0.0"
PORT = 5000


def select_private_ipv4(addresses: Iterable[str]) -> Optional[str]:
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


def discover_private_ipv4() -> Optional[str]:
    candidates: List[str] = []
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
        print("\nOn your phone, open: http://{}:{}".format(address, PORT))
    else:
        print("\nI could not detect the Wi-Fi IPv4 address automatically.")
        print("Run ipconfig, find the Wi-Fi IPv4 Address, then open:")
        print("http://<IPv4-address>:{}".format(PORT))
    print("If Windows asks, allow Python access on private networks only.\n")

    create_app().run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
