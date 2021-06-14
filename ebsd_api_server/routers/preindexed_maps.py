import io
import base64
from typing import List, Optional, Dict, Literal, Union, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import kikuchipy as kp

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from .ebsd import kp_signals, ebsd_info, maps
from ..utils.image import get_image_from_array

# IQ ADP

router = APIRouter(
    prefix="/maps/preindexed",
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)


class ImageQuality(BaseModel):
    normalize: bool = True


class AverageDotProduct(BaseModel):
    zero_mean: bool = True
    normalize: bool = True


class PreindexedMapsIn(BaseModel):
    ebsd_uid: str
    adp: Optional[AverageDotProduct] = None
    iq: Optional[ImageQuality] = None


class PreindexedMapsOut(BaseModel):
    adp: Optional[str]
    iq: Optional[str]


@router.post("/", response_model=PreindexedMapsOut)
def preindexed_maps(req: PreindexedMapsIn):
    s = kp_signals[req.ebsd_uid]
    response = {}
    if req.iq is not None:
        if "iq" in maps[req.ebsd_uid]:
            iq = maps[req.ebsd_uid]["iq"]
        else:
            iq = s.get_image_quality(**req.iq.dict())
            maps[req.ebsd_uid]["iq"] = iq
        svg_string = io.StringIO()
        plt.figure()
        plt.imshow(iq, cmap="gray")
        plt.colorbar(label=r"Image quality, $Q$", pad=0.01)
        _ = plt.axis("off")
        plt.savefig(svg_string, format="svg", bbox_inches="tight", transparent=True)
        svg_string.seek(0)
        response["iq"] = svg_string.getvalue()
    if req.adp is not None:
        if "adp" in maps[req.ebsd_uid]:
            adp = maps[req.ebsd_uid]["adp"]
        else:
            adp = s.get_average_neighbour_dot_product_map(**req.adp.dict())
            maps[req.ebsd_uid]["adp"] = adp

        svg_string = io.StringIO()
        plt.figure()
        plt.imshow(adp, cmap="gray")
        plt.colorbar(label="Average dot product", pad=0.01)
        _ = plt.axis("off")
        plt.savefig(svg_string, format="svg", bbox_inches="tight", transparent=True)
        svg_string.seek(0)
        response["adp"] = svg_string.getvalue()

    return PreindexedMapsOut.parse_obj(response)
