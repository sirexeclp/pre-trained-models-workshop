import uvicorn

from .app import app

uvicorn.run(app, host="0.0.0.0")
