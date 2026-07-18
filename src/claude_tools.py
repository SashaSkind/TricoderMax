"""
Hot-path tools for the Claude layer — bounded agency.

Claude may REQUEST evidence rather than only receiving a fixed package: pull a
different slice range or window, or measure HU statistics over a region. This
enables the representative diagnostic move the spec calls out — hemorrhage
attention at the skull base → get_slices(z, z, "bone") → confirm/exclude partial
voluming from adjacent bone.

Hard limits (enforced by the caller in claude_layer): max 3 tool calls,
wall-clock timeout, and MONOTONIC panel expansion — tools only ADD information,
never suppress a module or gate perception. On any failure the caller falls back
to detector-only priority.
"""

from __future__ import annotations

import base64

import numpy as np

from src.windowing import to_uint8, window_named

# Anthropic tool definitions offered on the hot path (record_assessment is added
# separately by the caller).
TOOL_DEFS = [
    {
        "name": "get_slices",
        "description": (
            "Render one or more axial slices in a chosen CT window and see them. Use to "
            "re-examine a finding in a different window — e.g. the bone window at a slice "
            "where hemorrhage attention sits near the skull, to check for partial voluming."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {"type": "integer", "description": "first slice index"},
                "end": {"type": "integer", "description": "last slice index (inclusive)"},
                "window": {"type": "string", "enum": ["brain", "subdural", "bone"]},
            },
            "required": ["start", "end", "window"],
        },
    },
    {
        "name": "measure_roi",
        "description": (
            "Report Hounsfield-Unit statistics (mean/min/max, % bone-density) over a "
            "rectangular region of one slice. Use to quantify whether bright attention is "
            "blood (~50-90 HU) vs bone/calcification (>300 HU)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slice_idx": {"type": "integer"},
                "bbox": {"type": "array", "items": {"type": "integer"},
                         "description": "[x0, y0, x1, y1] in pixels"},
            },
            "required": ["slice_idx", "bbox"],
        },
    },
    {
        "name": "fetch_prior",
        "description": "Retrieve the previous report for this patient, if one exists (text).",
        "input_schema": {"type": "object", "properties": {
            "patient_id": {"type": "string"}}, "required": []},
    },
]


def _img_block(hu_slice: np.ndarray, window: str) -> dict:
    png = to_uint8(window_named(hu_slice, window))
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(png).save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def handle_tool(name: str, tool_input: dict, volume: np.ndarray, ctx) -> list[dict]:
    """Execute a hot-path tool → content blocks for the tool_result. Never raises."""
    try:
        z_max = volume.shape[0] - 1
        if name == "get_slices":
            window = tool_input.get("window", "brain")
            start = max(0, min(int(tool_input.get("start", 0)), z_max))
            end = max(start, min(int(tool_input.get("end", start)), z_max))
            end = min(end, start + 3)  # cap at 4 images per call
            blocks: list[dict] = [{"type": "text", "text": f"slices {start}..{end} in {window} window:"}]
            for z in range(start, end + 1):
                blocks.append(_img_block(volume[z], window))
            return blocks
        if name == "measure_roi":
            z = max(0, min(int(tool_input.get("slice_idx", 0)), z_max))
            x0, y0, x1, y1 = tool_input.get("bbox", [0, 0, volume.shape[2], volume.shape[1]])
            roi = volume[z, max(0, y0):y1, max(0, x0):x1]
            if roi.size == 0:
                return [{"type": "text", "text": "empty ROI"}]
            bone_frac = float((roi > 300).mean())
            return [{"type": "text", "text": (
                f"ROI slice {z} {[x0, y0, x1, y1]}: mean={roi.mean():.0f} HU, "
                f"min={roi.min():.0f}, max={roi.max():.0f}, %bone(>300HU)={bone_frac*100:.0f}%. "
                f"(blood ~50-90 HU; calcification/bone >300 HU)")}]
        if name == "fetch_prior":
            prior = getattr(ctx, "prior_report", None)
            return [{"type": "text", "text": prior or "No prior report available for this patient."}]
        return [{"type": "text", "text": f"unknown tool {name}"}]
    except Exception as e:  # noqa: BLE001 — tool errors must not crash the loop
        return [{"type": "text", "text": f"tool {name} failed: {type(e).__name__}: {e}"}]
