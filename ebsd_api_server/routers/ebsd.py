import os

from typing import List, Optional, Tuple, Dict, Literal, Union
from enum import Enum
from fastapi import Depends, APIRouter
from pydantic import BaseModel

import kikuchipy as kp

router = APIRouter(
    prefix="/patterns",
    tags=["dataset"],
    # dependencies=[Depends(get_token_header)],
    # responses={404: {"description": "Not found"}},
)

kp_signals: Dict[str, kp.signals.EBSD] = {}

# ebsd_info: Dict[str, EBSDInfo] = {}


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
    lazy: bool = False

    # TODO: validate indexes against width and height


class LoadPatterns(BaseModel):
    uid: str
    file_path: str
    lazy: bool = False


@router.post("/load", response_model=EBSDInfo)
async def load_patterns(l: LoadPatterns):

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
            "path": l.file_path,
            "file_extension": file_extension[1:],
            "name": name,
            "metadata": metadata,
            "sig_width": sig_width,
            "sig_height": sig_height,
            "nav_width": nav_width,
            "nav_height": nav_height,
            "lazy": l.lazy,
        }
    )
    kp_signals[l.uid] = s

    ##ebsd.nav_map = mean_intensity(uuid)
    return ebsd

