from typing import List, Optional, Dict  # , Literal # Python 3.8
from enum import Enum
from uuid import UUID, uuid4
from fastapi import Depends, APIRouter

from pydantic import BaseModel

import kikuchipy as kp


class PhaseInfo(BaseModel):
    atom_coordinates: Optional[dict] = None
    formula: Optional[str] = None
    info: str = None
    lattice_constants: Optional[List[float]] = None  # six lattice constants
    laue_group: Optional[str] = None
    material_name: Optional[str] = None
    setting: Optional[int] = None
    point_group: Optional[str] = None
    source: Optional[str] = None
    space_group: Optional[str] = None
    symmetry: Optional[int] = None
