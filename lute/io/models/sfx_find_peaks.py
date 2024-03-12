from typing import Literal, Union

from pydantic import BaseModel, Field

from .base import BaseBinaryParameters, TaskParameters


class FindPeaksPyAlgosParameters(TaskParameters):
    n_events: int = Field(
        0,
        description="Number of events to process (0 to process all events)",
    )
    det_name: str = Field(
        description="Psana name of the detector storing the image data",
    )
    event_receiver: Literal["evr0", "evr1"] = Field(
        description="Event Receiver to be used: evr0 or evr1",
    )
    tag: str = Field(
        "",
        description="Tag to add to the output file names",
    )
    pv_camera_length: Union[str, float] = Field(
        "",
        description="PV associated with camera length "
        "(if a number, camera length directly)",
    )
    event_logic: bool = Field(
        False,
        description="True if only events with a specific event code should be "
        "processed. False if the event code should be ignored",
    )
    event_code: int = Field(
        0,
        description="Required events code for events to be processed if event logic "
        "is True",
    )
    psana_mask: bool = Field(
        False,
        description="If True, apply mask from psana Detector object",
    )
    mask_file: Union[str, None] = Field(
        None,
        description="File with a custom mask to apply. If None, no custom mask is "
        "applied",
    )
    min_peaks: int = Field(2, description="Minimum number of peaks per image")
    max_peaks: int = Field(
        2048,
        description="Maximum number of peaks per image",
    )
    npix_min: int = Field(
        2,
        description="Minimum number of pixels per peak",
    )
    npix_max: int = Field(
        30,
        description="Maximum number of pixels per peak",
    )
    amax_thr: float = Field(
        80.0,
        description="Minimum intensity threshold for starting a peak",
    )
    atot_thr: float = Field(
        120.0,
        description="Minimum summed intensity threshold for pixel collection",
    )
    son_min: float = Field(
        7.0,
        description="Minimum signal-to-noise ratio to be considered a peak",
    )
    peak_rank: int = Field(
        3,
        description="Radius in which central peak pixel is a local maximum",
    )
    r0: float = Field(
        3.0,
        description="Radius of ring for background evaluation in pixels",
    )
    dr: float = Field(
        2.0,
        description="Width of ring for background evaluation in pixels",
    )
    nsigm: float = Field(
        7.0,
        description="Intensity threshold to include pixel in connected group",
    )
