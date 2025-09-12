from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "API is working!"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)