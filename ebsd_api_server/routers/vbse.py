from typing import List, Optional, Dict, Literal, Union, Tuple
from enum import Enum
from uuid import UUID
from fastapi import Depends, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import kikuchipy as kp

from .ebsd import kp_signals, EBSDInfo
from ..utils.image import get_image_from_array, rescale_intensities

router = APIRouter(
    prefix="/vbse",
    tags=["process"],
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)


class RectangularROI(BaseModel):
    x_start: int
    x_size: int
    y_start: int
    y_size: int


class vBSEColorImagingIn(BaseModel):
    ebsd: EBSDInfo
    red: Optional[List[RectangularROI]] = None
    green: Optional[List[RectangularROI]] = None
    blue: Optional[List[RectangularROI]] = None
    percentile_contrast_stretch: Optional[Tuple[float, float]] = None


vbse_color_arrays = {}


class vBSEColorImagingOut(BaseModel):
    red: Optional[str]
    green: Optional[str]
    blue: Optional[str]
    rgb: Optional[str]


@router.post("/rgb", response_model=vBSEColorImagingOut)
def vbse_color_imaging(vbse: vBSEColorImagingIn):
    ebsd = vbse.ebsd
    s = kp_signals[ebsd.uid]
    if ebsd.uid not in vbse_color_arrays:
        vbse_color_arrays[ebsd.uid] = np.zeros(
            (ebsd.nav_height, ebsd.nav_width, 3), "uint8"
        )
    colors = ["red", "green", "blue"]
    response = {}
    for i, rects in enumerate([vbse.red, vbse.green, vbse.blue]):
        if rects is not None:
            vbse_data = np.zeros((ebsd.nav_height, ebsd.nav_width))
            for rect in rects:
                l = rect.x_start
                r = l + rect.x_size
                t = rect.y_start
                b = t + rect.y_size
                vbse_data += np.sum(s.data[..., t:b, l:r], axis=(-2, -1))
            vbse_data = rescale_intensities(
                vbse_data, percentiles=vbse.percentile_contrast_stretch
            )
            vbse_color_arrays[ebsd.uid][..., i] = vbse_data
            response[colors[i]] = get_image_from_array(vbse_data)
    response["rgb"] = get_image_from_array(
        vbse_color_arrays[ebsd.uid], mode="RGB", percentiles=None
    )
    return vBSEColorImagingOut.parse_obj(response)

