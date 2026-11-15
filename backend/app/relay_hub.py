"""Relay between the platform host and agents that cannot be reached directly (home networks, other sites).

An agent that connects with ?relay=1 gets its own block of worker ports. The platform host listens on those ports and carries
every connection through the agent's WebSocket to the same port on the agent's machine. The agent in turn listens on the job
DHT port and on every other machine's worker ports and carries those connections back here, so every machine reaches every
other one through the platform host. Workers keep announcing 127.0.0.1:<port>, which is valid everywhere.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from app.relay import Relay

log = logging.getLogger("bloombee_platform.relay_hub")

Send = Callable[[int, bytes], Awaitable[None]]
Notify = Callable[[int, dict], Awaitable[None]]


class RelayHub:
    def __init__(self, send: Send, notify: Notify, dht_port: int, local_ports: Optional[tuple[int, int]] = None,
                 base_port: int = 31100, block: int = 10, max_agents: int = 100):
        self._send, self._notify = send, notify
        self.dht_port = dht_port
        self.local = list(range(local_ports[0], local_ports[1] + 1)) if local_ports else []  # worker ports of agents on this host
        self.base_port, self.block, self.max_agents = base_port, block, max_agents
        self.agents: dict[int, tuple[Relay, list[int]]] = {}

    @property
    def enabled(self) -> bool:
        return self.dht_port > 0

    async def attach(self, token_id: int) -> Optional[list[int]]:
        """A relay agent connected: give it a port block and start carrying traffic. Returns its ports, or None if full."""
        if not self.enabled:
            return None
        await self.detach(token_id)
        taken = {ports[0] for _, ports in self.agents.values()}
        for slot in range(self.max_agents):
            start = self.base_port + slot * self.block
            if start not in taken:
                break
        else:
            log.warning("relay: no free port block for token %s", token_id)
            return None
        ports = list(range(start, start + self.block))
        relay = Relay(lambda data, t=token_id: self._send(t, data), parity=0)
        await relay.listen(ports)
        self.agents[token_id] = (relay, ports)
        await self._notify(token_id, {"type": "relay_config", "port_range": [ports[0], ports[-1]]})
        await self._publish_expose()
        log.info("relay: token %s relays ports %d-%d", token_id, ports[0], ports[-1])
        return ports

    async def detach(self, token_id: int) -> None:
        entry = self.agents.pop(token_id, None)
        if entry is not None:
            await entry[0].close()
            await self._publish_expose()

    async def frame(self, token_id: int, data: bytes) -> None:
        entry = self.agents.get(token_id)
        if entry is not None:
            await entry[0].handle(data)

    def expose_for(self, token_id: int) -> list[int]:
        """What that agent must listen on: the job DHT, the worker ports on this host and every other relay agent's ports."""
        ports = {self.dht_port, *self.local}
        for other, (_, other_ports) in self.agents.items():
            if other != token_id:
                ports.update(other_ports)
        return sorted(ports)

    async def _publish_expose(self) -> None:
        for token_id in list(self.agents):
            try:
                await self._notify(token_id, {"type": "relay_expose", "ports": self.expose_for(token_id)})
            except Exception:  # noqa: BLE001 - a dropped agent is cleaned up by its own connection handler
                log.debug("relay: could not update token %s", token_id)
