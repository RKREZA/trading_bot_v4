import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/events"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket!")
            for _ in range(3):
                message = await websocket.recv()
                data = json.loads(message)
                print(f"Received: {data['type']} at {data['timestamp']}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
