#!/usr/bin/env python3
"""Runs ON A PI: makes the Thymio wiggle in place (left-right) a few times so a person can spot it, then stops the motors."""
import asyncio, sys
from tdmclient import ClientAsync
times = int(sys.argv[1]) if len(sys.argv) > 1 else 3
async def main():
    with ClientAsync() as client:
        for _ in range(50):
            client.process_waiting_messages()
            if list(client.nodes): break
            await asyncio.sleep(0.1)
        node = list(client.nodes)[0]
        await node.lock()
        try:
            for _ in range(times):
                for l, r in ((-150, 150), (150, -150)):
                    await node.set_variables({"motor.left.target": [l], "motor.right.target": [r]})
                    await asyncio.sleep(0.35)
        finally:
            await node.set_variables({"motor.left.target": [0], "motor.right.target": [0]})
            await node.unlock()
        print("wiggled")
asyncio.run(main())
