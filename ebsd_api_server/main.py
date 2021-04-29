import uvicorn
from fastapi import Depends, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .routers import ebsd, process_intensities, vbse

app = FastAPI()
app.include_router(ebsd.router)
app.include_router(vbse.router)
app.include_router(process_intensities.router)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await ebsd.load_patterns(
        ebsd.LoadPatterns(
            uid="string",
            file_path="/home/ole/Masteroppgave/09_DataSet/sdss/nordif/Pattern.dat",
            lazy=False,
        )
    )
    return


@app.on_event("shutdown")
def shutdown():
    return None


def start():
    uvicorn.run("ebsd_api_server.main:app", host="0.0.0.0", port=8008, reload=True)
