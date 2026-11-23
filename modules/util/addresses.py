"""Public and local web addresses: what may be fetched, and what names this machine or its network.

A public address is global unicast, carries no local IPv4 address inside an IPv6 one, and lies
outside this machine's own networks.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from urllib.parse import urlsplit

__all__ = [
    "LOCAL_SUFFIXES",
    "WEB_SCHEMES",
    "is_public",
    "local_name",
    "public_addresses",
    "refuse_local",
]

#: Schemes a fetched web address may use.
WEB_SCHEMES = ("http", "https")

#: Host name endings that name this machine or a network it sits on.
LOCAL_SUFFIXES = (
    ".localhost",
    ".local",
    ".lan",
    ".home",
    ".home.arpa",
    ".internal",
    ".intranet",
    ".corp",
    ".private",
)

#: NAT64 prefixes that carry an IPv4 address in their last 32 bits.
NAT64_PREFIXES = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)


#: Single addresses a cloud host answers on for its own machines.
HOST_SERVICES = (ipaddress.ip_network("168.63.129.16/32"),)

#: Seconds the list of this machine's own networks is kept.
OWN_NETWORKS_SECONDS = 30.0

_own = {"at": -1e9, "networks": ()}


def _own_networks() -> tuple:
    """Every network this machine's interfaces hold an address on."""
    now = time.monotonic()
    if now - _own["at"] < OWN_NETWORKS_SECONDS:
        return _own["networks"]
    networks = []
    try:
        import psutil

        for entries in psutil.net_if_addrs().values():
            for entry in entries:
                if entry.family not in (socket.AF_INET, socket.AF_INET6) or not entry.address:
                    continue
                address = str(entry.address).split("%", 1)[0]
                mask = str(entry.netmask or "").split("%", 1)[0]
                try:
                    networks.append(ipaddress.ip_interface(f"{address}/{mask}" if mask else address).network)
                except ValueError:
                    continue
    except Exception:
        networks = []
    _own["at"], _own["networks"] = now, tuple(networks)
    return _own["networks"]


def _ip(value: str):
    """An IP address from a literal, including legacy IPv4 spellings, or ``None``."""
    text = str(value or "").strip().strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        pass
    if text and ":" not in text and all(part for part in text.split(".")):
        try:
            return ipaddress.IPv4Address(socket.inet_aton(text))
        except (OSError, ValueError):
            return None
    return None


def is_public(address: str) -> bool:
    """Whether an IP address is global unicast.

    Args:
        address: An IPv4 or IPv6 literal, in any spelling a browser accepts.

    Returns:
        False for loopback, private, link-local, multicast, reserved, unspecified, shared,
        documentation and benchmark ranges, for an IPv6 address carrying one of those, for a
        cloud host's own service address, for an address on a network this machine sits on,
        and for anything that is not an IP address.
    """
    ip = _ip(address)
    if ip is None:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        inner = [ip.ipv4_mapped, ip.sixtofour, (ip.teredo or (None, None))[1]]
        for prefix in NAT64_PREFIXES:
            if ip in prefix:
                inner.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        for carried in inner:
            if carried is not None and not is_public(str(carried)):
                return False
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast             or ip.is_reserved or ip.is_unspecified or not ip.is_global:
        return False
    return not any(ip in network for network in (*HOST_SERVICES, *_own_networks())
                   if network.version == ip.version)


def local_name(host: str) -> bool:
    """Whether a host name, before any lookup, names this machine or its local network.

    Args:
        host: The host part of an address, without brackets or port.

    Returns:
        True for ``localhost``, a name with no dot, a name ending in one of
        :data:`LOCAL_SUFFIXES`, and an IP literal that is not public.
    """
    name = str(host or "").strip().strip("[]").rstrip(".").lower()
    if not name:
        return True
    if _ip(name) is not None:
        return not is_public(name)
    if name == "localhost" or "." not in name:
        return True
    return any(name.endswith(suffix) for suffix in LOCAL_SUFFIXES)


def public_addresses(host: str, port: int, who: str) -> list[str]:
    """Every address a host resolves to, once each one is public.

    Args:
        host: The host part of an address.
        port: The port it is reached on.
        who: The node's display name, for the message.

    Returns:
        The resolved addresses, each once, in resolver order.

    Raises:
        ValueError: The host is a local name, or any address it resolves to is not public.
        RuntimeError: The host does not resolve.
    """
    if local_name(host):
        raise ValueError(
            f"{who} refuses {host}: it names this machine or a local network. Only public web "
            f"addresses are fetched."
        )
    literal = _ip(host)
    if literal is not None:
        resolved = [str(literal)]
    else:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(f"{who} could not find the host {host} ({error}).") from None
        resolved = []
        for info in infos:
            address = str(info[4][0]).split("%", 1)[0]
            if address not in resolved:
                resolved.append(address)
    if not resolved:
        raise RuntimeError(f"{who} could not find the host {host}.")
    for address in resolved:
        if not is_public(address):
            raise ValueError(
                f"{who} refuses {host}: it resolves to {address}, an address on this machine "
                f"or a private network. Only public web addresses are fetched."
            )
    return resolved


def refuse_local(url: str, who: str) -> None:
    """Refuse a web address that reaches this machine or a private network.

    Args:
        url: An ``http`` or ``https`` address.
        who: The node's display name, for the message.

    Raises:
        ValueError: The address has no host, is not ``http`` or ``https``, or its host is local
            or resolves to an address that is not public.
        RuntimeError: The host does not resolve.
    """
    try:
        parts = urlsplit(str(url or "").strip())
        port = parts.port
    except ValueError as error:
        raise ValueError(f"{who} cannot read that address ({error}).") from None
    if parts.scheme.lower() not in WEB_SCHEMES or not parts.hostname:
        raise ValueError(f"{who} fetches only whole http or https addresses, such as https://example.com/a.png.")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"{who} refuses an address carrying a user name or password.")
    public_addresses(parts.hostname, port or (443 if parts.scheme.lower() == "https" else 80), who)
