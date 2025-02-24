import uvicorn
from fastapi import FastAPI
import os #add this import

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from Render!"}

if __name__ == "__main__":
    print("Starting server...")
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    print(f"Uvicorn host: {config.host}, port: {config.port}")
    print(f"Current Working Directory: {os.getcwd()}") # added line.
    print(f"Files in Current Directory: {os.listdir()}") # added line.
    server.run()
