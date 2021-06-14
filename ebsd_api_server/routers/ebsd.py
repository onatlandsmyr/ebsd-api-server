import os

from typing import List, Optional, Tuple, Dict, Literal, Union
from enum import Enum
from fastapi import Depends, APIRouter
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from ..utils.image import get_image_from_array
from fastapi.responses import StreamingResponse

import numpy as np
import kikuchipy as kp

router = APIRouter(
    prefix="/patterns",
    tags=["dataset"],
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)


class MetadataEBSD(BaseModel):
    name: Optional[str] = None
    comments: Optional[str] = None
    sample_preperation: Optional[str] = None
    microscope: Optional[str] = None
    detector: Optional[str] = None
    scanning_mode: Optional[Literal["Beam", "Stage", "Combined"]] = None
    working_distance: Optional[float] = None
    accelerating_voltage: Optional[float] = None  # beam_energy
    magnification: Optional[float] = None
    sample_tilt: Optional[float] = None
    step_size: Optional[Tuple[float, str]] = None
    pattern_center: Optional[Tuple[float, float, float, str]] = None
    original_raw_metadata: Optional[str] = None


class EBSDInfo(BaseModel):
    uid: str  # = Field(default_factory=uuid4)
    file_path: Optional[str]
    metadata: Optional[MetadataEBSD] = None
    static_background: Optional[str] = None
    sig_width: int
    sig_height: int
    nav_width: int
    nav_height: int
    nav_x_index: int = 0
    nav_y_index: int = 0
    nav_map: Optional[str] = None
    lazy: bool = False

    # TODO: validate indexes against width and height


class LoadPatterns(BaseModel):
    uid: str
    file_path: str
    lazy: bool = True


class SavePatterns(BaseModel):
    uid: str
    file_path: str
    overwrite: Optional[bool] = None


kp_signals: Dict[str, kp.signals.EBSD] = {}
ebsd_info: Dict[str, EBSDInfo] = {}
logs = {}
maps = {}


@router.post("/load", response_model=EBSDInfo)
async def load_patterns(l: LoadPatterns):
    kp_signals.clear()
    ebsd_info.clear()

    # Try to load patterns into the kikuchipy.signals.EBSD class
    s = kp.load(l.file_path, lazy=l.lazy)

    # Extract metadata to EBSDInfo class defined above
    nav_height, nav_width, sig_height, sig_width = s.data.shape

    #  General.title is set to the filename which usually is Pattern(.dat),
    name = s.metadata.get_item("Sample.Phases.1.material_name")
    microscope = s.metadata.get_item("Acquisition_instrument.SEM.microscope")
    working_distance = s.metadata.get_item(
        "Acquisition_instrument.SEM.working_distance"
    )
    accelerating_voltage = s.metadata.get_item("Acquisition_instrument.SEM.beam_energy")
    magnification = s.metadata.get_item("Acquisition_instrument.SEM.magnification")
    detector = s.metadata.get_item("Acquisition_instrument.SEM.Detector.detector")
    sample_tilt = s.metadata.get_item("Acquisition_instrument.SEM.Detector.sample_tilt")
    original_raw_metadata = "\n".join(
        list(s.original_metadata.as_dictionary()["nordif_header"])
    )
    # TODO step_size <- axes_manager
    nav_scale = s.axes_manager.navigation_axes[0].scale
    nav_units = s.axes_manager.navigation_axes[0].units
    metadata = MetadataEBSD(
        name=name,
        microscope=microscope,
        working_distance=working_distance,
        accelerating_voltage=accelerating_voltage,
        magnification=magnification,
        detector=detector,
        sample_tilt=sample_tilt,
        original_raw_metadata=original_raw_metadata,
        step_size=(nav_scale, nav_units),
    )

    # Store the EBSD signal into the dictionary db with the uid generated.
    filename, file_extension = os.path.splitext(l.file_path)
    ebsd = EBSDInfo.parse_obj(
        {
            "uid": l.uid,
            "file_path": l.file_path,
            "file_extension": file_extension[1:],
            "metadata": metadata,
            "sig_width": sig_width,
            "sig_height": sig_height,
            "nav_width": nav_width,
            "nav_height": nav_height,
            "lazy": l.lazy,
        }
    )
    kp_signals[l.uid] = s
    nav_map = mean_intensity(ebsd)
    maps[l.uid] = {"mean_intensity": nav_map, "img_type": "base64"}
    ebsd.nav_map = nav_map

    ebsd_info[l.uid] = ebsd
    ##ebsd.nav_map = mean_intensity(uuid)
    return ebsd


def mean_intensity(ebsd: EBSDInfo, percentiles=(0.1, 99.9)):
    s = kp_signals[ebsd.uid]
    sig_axes = s.axes_manager.signal_indices_in_array
    data = np.mean(s.data, axis=sig_axes)
    if ebsd.lazy:
        data = data.compute()
    base64str = get_image_from_array(data, percentiles=percentiles)
    return base64str


@router.post("/save")
async def save_patterns(save: SavePatterns):
    kp_signals[save.uid].save(save.file_path, overwrite=save.overwrite)


@router.get("/{uid}")
async def read_pattern(uid: str, x: int, y: int):
    im_data = kp_signals[uid].data[y, x]
    if ebsd_info[uid].lazy:
        im_data = im_data.compute()
    # Update db_EBSD TODO

    image = get_image_from_array(im_data, encoding="binary", percentiles=None, mode="L")
    headers = {
        "Cache-Control": "private",
    }
    return StreamingResponse(image, media_type="image/png", headers=headers)


@router.get("/logs")
def get_log(log_id: str):
    if log_id not in logs:
        raise HTTPException(status_code=404, detail="Log not found")
    f = logs[log_id]
    print("log_id")
    log_text = "\n".join([line.split("\r")[-1] for line in f.getvalue().split("\n")])
    return log_text


@router.get("/info/{uid}")
def get_ebsd_info(uid: str):
    if uid not in ebsd_info:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    return ebsd_info[uid]


@router.get("/")
def loaded_datasets():
    return list(ebsd_info.values())
