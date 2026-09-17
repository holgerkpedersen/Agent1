"""Temporary test script for hue set_light_color fix."""
import asyncio
import httpx
from agent_core.hue.bridge import HueBridge


async def main():
    b = HueBridge.from_env()
    
    # First, set light to blue via RGB to test the method
    print("Setting light 3 to blue via RGB...", flush=True)
    await b.set_light_color("3", r=0, g=0, b=255)
    
    # Verify via V1 API
    async with httpx.AsyncClient() as client:
        url = f"http://{b._base_ip}/api/{b._api_key}/lights/3"
        resp = await client.get(url, timeout=10.0)
        state = resp.json().get("state", {})
        print(f"Current xy: {state.get('xy')}, bri: {state.get('bri')}", flush=True)
    
    # Now set to red via direct xy
    print("Setting light 3 to red via xy...", flush=True)
    await b.set_light_color("3", x=0.675, y=0.322)
    
    async with httpx.AsyncClient() as client:
        url = f"http://{b._base_ip}/api/{b._api_key}/lights/3"
        resp = await client.get(url, timeout=10.0)
        state = resp.json().get("state", {})
        print(f"Current xy: {state.get('xy')}, bri: {state.get('bri')}", flush=True)
    
    print("DONE", flush=True)


asyncio.run(main())
