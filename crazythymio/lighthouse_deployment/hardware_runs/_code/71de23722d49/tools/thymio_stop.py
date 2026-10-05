#!/usr/bin/env python3
"""Runs ON A PI: sets the Thymio's motor targets to 0 directly through the Thymio Device Manager (last line of defence
after a controller process was killed or hung -- a killed process does not reset the motors itself)."""
import asyncio
from tdmclient import ClientAsync
async def main():
    with ClientAsync() as client:
        for _ in range(50):
            client.process_waiting_messages()
            if list(client.nodes): break
            await asyncio.sleep(0.1)
        node = list(client.nodes)[0]
        await node.lock()
        await node.set_variables({"motor.left.target": [0], "motor.right.target": [0]})
        await node.unlock()
        print("motors 0")
asyncio.run(main())
