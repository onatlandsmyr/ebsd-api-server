from logging import debug
from typing import List, Optional, Tuple, Union, Literal
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from enum import Enum
from uuid import UUID, uuid4
from fastapi import Depends, APIRouter, HTTPException

from pydantic import BaseModel
import numpy as np

import kikuchipy as kp
from orix import sampling, plot
from orix import io as orix_io
from orix.quaternion import Rotation
from starlette.responses import PlainTextResponse
from .ebsd import EBSDInfo, kp_signals, ebsd_info, logs

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

router = APIRouter(
    prefix="/indexing",
    tags=["indexing"],
)


class SampleDetectorGeometry(BaseModel):
    pc: Tuple[float, float, float]
    pc_convention: Literal["tsl", "emsoft5", "bruker", "oxford", "emsoft4"]
    sample_tilt: float
    camera_tilt: float = 0.0
    px_size: float = 1.0  # microns


class MasterPattern(BaseModel):
    filepath: str
    projection: Literal["lambert"] = "lambert"
    energy: Union[int, float]


class DictionaryIndexingIn(BaseModel):
    ebsd_info: EBSDInfo
    masterpatterns: List[MasterPattern]
    detector: SampleDetectorGeometry
    angle_resolution: float
    metric: Literal["ncc", "ndp"]
    keep_n: int = 50
    n_slices: int = 10  # tricky
    result_dir: str  # non-ideal for seperation of UI and  backend?
    osm: bool = True
    preview_angles: Optional[Tuple[float, float, float]] = None


class DictionaryIndexingOut(BaseModel):
    score_map: Optional[str] = None
    os_map: Optional[str] = None
    phase_map: Optional[str] = None
    rgb_orientation_map: Optional[str] = None


class PreviewDictionary(BaseModel):
    patterns: List[str]


@router.post("/preview", response_model=PreviewDictionary)
async def preview_dictionary(req: DictionaryIndexingIn):
    if req.ebsd_info.uid not in kp_signals:
        raise HTTPException(status_code=404, detail="Patterns not found")
    s = kp_signals[req.ebsd_info.uid]

    detector = kp.detectors.EBSDDetector(
        shape=s.axes_manager.signal_shape[::-1],
        pc=req.detector.pc,
        sample_tilt=req.detector.sample_tilt,
        convention=req.detector.pc_convention,
        tilt=req.detector.camera_tilt,
    )
    res = PreviewDictionary(patterns=[])
    if req.preview_angles:
        euler = np.array(req.preview_angles)
    else:
        euler = np.array([0.0, 0.0, 0.0])
    for mp_req in req.masterpatterns:
        mp = kp.load(
            mp_req.filepath,
            projection=mp_req.projection,
            hemisphere="both",
            energy=mp_req.energy,
        )
        sim = mp.get_patterns(
            rotations=Rotation.from_euler(np.deg2rad(euler)),
            detector=detector,
            energy=mp_req.energy,
            compute=True,
        )
        svg_string = io.StringIO()
        plt.figure()
        plt.imshow(sim.data[0], cmap="gray")
        plt.title(
            f"{mp.phase.name.capitalize()}\n($\phi_1, \Phi, \phi_2)$ = {np.array_str(euler, precision=1)}"
        )
        plt.axis("off")
        plt.savefig(svg_string, format="svg", bbox_inches="tight", transparent=True)
        svg_string.seek(0)
        res.patterns.append(svg_string.getvalue())
    return res


@router.post("/execute", response_model=DictionaryIndexingOut)
def dictionary_indexing(req: DictionaryIndexingIn):
    f = io.StringIO()
    logs["pattern_matching"] = f

    if req.ebsd_info.uid not in kp_signals:
        raise HTTPException(status_code=404, detail="Patterns not found")

    s = kp_signals[req.ebsd_info.uid]
    detector = kp.detectors.EBSDDetector(
        shape=s.axes_manager.signal_shape[::-1],
        pc=req.detector.pc,
        sample_tilt=req.detector.sample_tilt,
        convention=req.detector.pc_convention,
        tilt=req.detector.camera_tilt,
    )

    mps = []
    rotations_per_mp = []
    sims_per_mp = []
    for mp_req in req.masterpatterns:
        mp = kp.load(
            mp_req.filepath,
            projection=mp_req.projection,
            hemisphere="both",
            energy=mp_req.energy,
        )
        mps.append(mp)
        rotations = sampling.get_sample_fundamental(
            resolution=req.angle_resolution,
            point_group=mp.phase.point_group.proper_subgroup,
        )
        rotations_per_mp.append(rotations)

        sims = mp.get_patterns(
            rotations=rotations,
            detector=detector,
            energy=mp_req.energy,
        )
        sims_per_mp.append(sims)

    with redirect_stderr(f):
        xmaps = s.match_patterns(
            sims_per_mp,
            metric=req.metric,
            keep_n=req.keep_n,
            n_slices=req.n_slices,
            get_orientation_similarity_map=req.osm,
            return_merged_crystal_map=True,
        )

    res = DictionaryIndexingOut()

    if len(mps) == 1:
        # xmaps is a single xmap if 1 set of simulated patterns is matched against
        phase_name = xmaps.phases_in_data.names[0]
        orix_io.save(os.path.join(req.result_dir, f"xmap_{phase_name}.h5"), xmaps)

        xmaps.top_score = xmaps.scores[:, 0]
        orix_io.save(
            os.path.join(req.result_dir, f"xmap_{phase_name}.ang"),
            xmaps,
            confidence_index_prop="top_score",
        )
        main_xmap = xmaps
    else:
        for xmap in xmaps[:-1]:
            phase_name = xmap.phases_in_data.names[0]
            orix_io.save(os.path.join(req.result_dir, f"xmap_{phase_name}.h5"), xmap)
            xmap.top_score = xmap.scores[:, 0]
            orix_io.save(
                os.path.join(req.result_dir, f"xmap_{phase_name}.ang"),
                xmap,
                confidence_index_prop="top_score",
            )
        merged_xmap = xmaps[-1]
        orix_io.save(os.path.join(req.result_dir, f"xmap_merged.h5"), merged_xmap)
        merged_xmap.top_score = merged_xmap.scores[:, 0]
        orix_io.save(
            os.path.join(req.result_dir, f"xmap_merged.ang"),
            merged_xmap,
            confidence_index_prop="top_score",
        )

        svg_string = io.StringIO()
        fig = plt.figure()
        ax = fig.add_subplot(projection="plot_map")
        ax.plot_map(merged_xmap)
        # ax.add_overlay(merged_xmap, merged_xmap.top_score)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.savefig(
            svg_string,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
        svg_string.seek(0)
        res.phase_map = svg_string.getvalue()

        main_xmap = merged_xmap

    svg_string = io.StringIO()
    fig = plt.figure()
    ax = fig.add_subplot(projection="plot_map")
    im = ax.plot_map(main_xmap, main_xmap.top_score, cmap="gray")
    cbar = ax.add_colorbar(req.metric.upper())
    ax.set_axis_off()
    plt.savefig(
        svg_string, format="svg", bbox_inches="tight", transparent=True, pad_inches=0
    )
    svg_string.seek(0)
    res.score_map = svg_string.getvalue()

    svg_string = io.StringIO()
    fig = plt.figure()
    ax = fig.add_subplot(projection="plot_map")
    im = ax.plot_map(main_xmap, main_xmap.osm, cmap="gray")
    cbar = ax.add_colorbar("Orientation Similarity")
    ax.set_axis_off()
    plt.savefig(
        svg_string, format="svg", bbox_inches="tight", transparent=True, pad_inches=0
    )
    svg_string.seek(0)
    res.os_map = svg_string.getvalue()

    top_euler_rotations = main_xmap.rotations.to_euler()[:, 0, :]
    res.rgb_orientation_map = euler_rgb_orientation_map(
        req.ebsd_info, top_euler_rotations
    )

    return res


def euler_rgb_orientation_map(ebsd_info: EBSDInfo, euler_rotations):
    r = euler_rotations[:, 0]
    g = euler_rotations[:, 1]
    b = euler_rotations[:, 2]
    r /= np.max(r)
    g /= np.max(g)
    b /= np.max(b)
    divide_by = np.max((np.max(r), np.max(g), np.max(b))) / 256
    r /= divide_by
    g /= divide_by
    b /= divide_by

    nav_shape = (ebsd_info.nav_height, ebsd_info.nav_width)
    rgb = np.dstack(
        (r.reshape(nav_shape), g.reshape(nav_shape), b.reshape((nav_shape)))
    ).astype("uint8")
    fig = plt.figure()
    im = plt.imshow(rgb)
    plt.axis("off")
    svg_string = io.StringIO()
    plt.savefig(
        svg_string, format="svg", bbox_inches="tight", transparent=True, pad_inches=0
    )
    svg_string.seek(0)
    svg_string = svg_string.getvalue()
    return svg_string


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


@router.get("/log", response_class=PlainTextResponse)
def log():
    log_id = "pattern_matching"
    if log_id not in logs:
        raise HTTPException(status_code=404, detail="Log not found")
    f = logs[log_id]

    log_text = "\n".join([line.split("\r")[-1] for line in f.getvalue().split("\n")])
    return log_text
