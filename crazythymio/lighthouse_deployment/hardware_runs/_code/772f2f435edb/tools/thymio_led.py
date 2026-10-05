#!/usr/bin/env python3
"""Runs ON A PI (with ~/Desktop/crazy_thymio/.venv/bin/python). usage: thymio_led.py <r> <g> <b>   (0..32 each)
Sets the Thymio's top LED by loading a one-line Aseba program (a plain variable write can be overridden by the Thymio's
built-in behaviours). The colour stays after this script exits, until the next program is loaded (run_hebbian.py loads
an 'LED off' program when it connects)."""
import asyncio, sys
from tdmclient import ClientAsync
r, g, b = (int(v) for v in sys.argv[1:4])
async def main():
    with ClientAsync() as client:
        for _ in range(50):
            client.process_waiting_messages()
            if list(client.nodes): break
            await asyncio.sleep(0.1)
        nodes = list(client.nodes)
        if not nodes:
            print("no Thymio found"); return 1
        node = nodes[0]
        await node.lock()
        err = await node.compile(f"call leds.top({r},{g},{b})")
        if err is None: err = await node.run()
        await node.unlock()
        print("ok" if err is None else f"error {err}")
asyncio.run(main())
