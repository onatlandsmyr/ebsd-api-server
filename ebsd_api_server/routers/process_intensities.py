import io
from contextlib import redirect_stdout
from typing import List, Optional, Dict, Literal, Union, Tuple
from enum import Enum
from uuid import UUID
from fastapi import Depends, APIRouter, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import numpy as np
import kikuchipy as kp

from scipy.ndimage import correlate
from .ebsd import kp_signals, EBSDInfo, logs
from ..utils.image import get_image_from_array

router = APIRouter(
    prefix="/patterns/process",
    tags=["process"],
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)


class RemoveStaticBackground(BaseModel):
    operation: Literal["subtract", "divide"]
    relative: Optional[bool] = True
    scale_bg: Optional[bool] = False
    # static_bg: Optional[Any] = None


class RemoveDynamicBackground(BaseModel):
    operation: Literal["subtract", "divide"]
    filter_domain: Literal["frequency", "spatial"]
    std: Optional[Union[int, float]] = None
    truncate: Optional[Union[int, float]] = 4.0
    # kwargs


class AverageNeighbors(BaseModel):
    window: Literal["gaussian", "circular"] = "gaussian"
    shape: Tuple[int, int] = (3, 3)
    std: Optional[float] = None


class Rescale(BaseModel):
    pass


class ProcessIntensities(BaseModel):
    ebsd_info: Optional[EBSDInfo]
    static_bg: Optional[RemoveStaticBackground] = None
    dynamic_bg: Optional[RemoveDynamicBackground] = None
    average: Optional[AverageNeighbors] = None
    rescale: Optional[Rescale] = None


@router.post("/preview")
async def preview_process_intensities(req: ProcessIntensities):
    if req.ebsd_info.uid not in kp_signals:
        raise HTTPException(status_code=404, detail="Patterns not found")
    x = req.ebsd_info.nav_x_index
    y = req.ebsd_info.nav_y_index
    s = kp_signals[req.ebsd_info.uid]

    # TODO: Sjekk om det er likebra å tvinge lazy
    # Dask Array + Compute for et enkelt pattern stedetfor alt det greiene her

    is_averaging = req.average is not None
    if is_averaging:
        # window bounds
        wt = 0
        wb = req.average.shape[0]
        wl = 0
        wr = req.average.shape[1]

        # window extent
        dy = wb // 2
        dx = wr // 2

        # index of pattern in roi
        roi_y = dy
        roi_x = dx

        # nav bounds roi
        t = y - dy
        b = y + dy + 1
        l = x - dx
        r = x + dx + 1

        if t < 0:
            wt = -t
            t = 0
            roi_y = y
        if b >= req.ebsd_info.nav_height:
            wb -= b - req.ebsd_info.nav_height
            b = req.ebsd_info.nav_height - 1
        if l < 0:
            wl = -l
            l = 0
            roi_x = x
        if r >= req.ebsd_info.nav_width:
            wr -= r - req.ebsd_info.nav_width
            r = req.ebsd_info.nav_width - 1

        # l = min(max(x - dx, 0), req.ebsd_info.nav_width)
        # r = min(x + dx + 1, req.ebsd_info.nav_width - 1)
        # t = min(max(y - dy, 0), req.ebsd_info.nav_height)
        # b = min(y + dy + 1, req.ebsd_info.nav_height - 1)
        s = s.inav[l:r, t:b].deepcopy()
    else:
        s = s.inav[x, y].deepcopy()

    # TODO: more manually with sent static_bg or stored here in py, dont rely on metadata and EBSD method.
    if req.static_bg is not None:
        s.remove_static_background(**req.static_bg.dict())
    # TODO: with chunk.remove_dynamic_background
    if req.dynamic_bg is not None:
        s.remove_dynamic_background(
            operation=req.dynamic_bg.operation,
            filter_domain=req.dynamic_bg.filter_domain,
        )
    if req.ebsd_info.lazy:
        s.compute()
    if is_averaging:
        window = kp.filters.Window(**req.average.dict(exclude_none=True))
        window = window[wt:wb, wl:wr, None, None]
        window_sum = np.sum(window)
        img_data = (np.sum(window * s.data, axis=(0, 1)) / window_sum).astype(np.uint8)
    else:
        img_data = s.data

    image = get_image_from_array(
        img_data, encoding="binary", percentiles=None, mode="L"
    )
    headers = {"Cache-Control": "private"}
    return StreamingResponse(image, media_type="image/png", headers=headers)


@router.post("/execute", response_class=PlainTextResponse)
def process_intensities(req: ProcessIntensities):
    if req.ebsd_info.uid not in kp_signals:
        raise HTTPException(status_code=404, detail="Patterns not found")

    s = kp_signals[req.ebsd_info.uid]

    f = io.StringIO()
    logs["process_intensities"] = f
    if req.static_bg is not None:
        with redirect_stdout(f):
            s.remove_static_background(**req.static_bg.dict())
    if req.dynamic_bg is not None:
        with redirect_stdout(f):
            s.remove_dynamic_background(
                operation=req.dynamic_bg.operation,
                filter_domain=req.dynamic_bg.filter_domain,
            )
    if req.average is not None:
        with redirect_stdout(f):
            window = kp.filters.Window(**req.average.dict(exclude_none=True))
            s.average_neighbour_patterns(window=window)

    # TODO: average, rescale, lazy and capture logs/progression
    return "\n".join([line.split("\r")[-1] for line in f.getvalue().split("\n")])


@router.get("/log", response_class=PlainTextResponse)
def log():
    log_id = "process_intensities"
    if log_id not in logs:
        raise HTTPException(status_code=404, detail="Log not found")
    f = logs[log_id]
    log_text = "\n".join([line.split("\r")[-1] for line in f.getvalue().split("\n")])
    return log_text
