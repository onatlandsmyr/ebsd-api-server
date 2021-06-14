import io
import sys
import uvicorn
import janus
import asyncio

from fastapi import Depends, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .routers import ebsd, process_intensities, vbse, preindexed_maps, indexing

app = FastAPI()
app.include_router(ebsd.router)
app.include_router(vbse.router)
app.include_router(process_intensities.router)
app.include_router(preindexed_maps.router)
app.include_router(indexing.router)

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
    return
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


class LinesStdOut(io.TextIOBase):
    def __init__(self, orig_stdout=None, lines=None):
        self.orig_stdout = orig_stdout
        self.lines = lines

    def write(self, s):
        # Process output in whatever way you like
        self.lines.append(s)
        # sync_q.put(s)
        # Write output to original stream, if desired
        if self.orig_stdout:
            self.orig_stdout.write(s)


@app.websocket("/log/ws")
async def websocket_log(ws: WebSocket):
    await ws.accept()
    # lines = [""]
    # sys.stdout = LinesStdOut(sys.stdout, lines)
    while True:
        log_key = await ws.receive_text()
        await ws.send_text(lines[-1])
        if log_key in ebsd.logs:
            f = ebsd.logs[log_key]
            log = "\n".join([line.split("\r")[-1] for line in f.getvalue().split("\n")])
            await ws.send_text(log)


def start():
    uvicorn.run("ebsd_api_server.main:app", host="0.0.0.0", port=8008, reload=True)
