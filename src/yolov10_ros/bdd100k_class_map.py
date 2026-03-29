from __future__ import annotations

"""
bdd100k_class_map.py
====================
Maps BDD100K class names (as reported by the YOLOv10 model trained on BDD100K)
to the perception message ObjObjectType enum and to the six downstream topic
categories used by the rest of the stack.

BDD100K classes
---------------
  car, truck, bus, person, rider, bicycle, motorcycle,
  traffic light, traffic sign

ObjObjectType encoding (from perception_msgs)
----------------------------------------------
  0  Unknown
  1  4-wheel vehicle  (car, small truck)
  2  Large vehicle    (semi / bus)
  3  Motorcycle / bicycle
  4  Pedestrian
  5  Fixed object / barricade
  6  Animal
  7  Automatic gate / railroad

Topic routing keys
------------------
  "objects"          → /objects_scoring           (Objects)
  "trafficsigns"     → /TrafficSigns_scoring       (TrafficSigns)
  "trafficsignalhead"→ /TrafficSignalsHead_scoring  (TrafficSignalsHead)
  "lanemarkings"     → /LaneMarkings_scoring        (LaneMarkings)
  "limitlines"       → /LimitLines_scoring          (LimitLines)
  "road_state"       → /road_state_scoring          (road_state)

Usage
-----
  from bdd100k_class_map import get_topic_key, get_object_type

  key  = get_topic_key("car")        # → "objects"
  typ  = get_object_type("car")      # → 1
  key  = get_topic_key("traffic light")  # → "trafficsignalhead"
"""

# ---------------------------------------------------------------------------
# Internal tables — edit here to adjust routing without touching node code
# ---------------------------------------------------------------------------

# Maps each BDD100K class name → which downstream topic it belongs to.
# Classes absent from this dict are silently dropped (Unknown / irrelevant).
_CLASS_TO_TOPIC: dict[str, str] = {
    # ── Drivable objects ────────────────────────────────────────────────────
    "car":          "objects",
    "truck":        "objects",
    "bus":          "objects",
    "person":       "objects",
    "rider":        "objects",
    "bicycle":      "objects",
    "motorcycle":   "objects",

    # ── Traffic control ─────────────────────────────────────────────────────
    "traffic light":  "trafficsignalhead",
    "traffic sign":   "trafficsigns",

    # ── Extend here as the model class set grows ────────────────────────────
    # "train":    "objects",
    # "stop sign": "trafficsigns",
}

# Maps each BDD100K class name → ObjObjectType integer.
# Only classes routed to "objects" need an entry here; others are ignored.
_CLASS_TO_OBJ_TYPE: dict[str, int] = {
    "car":        1,   # 4-wheel vehicle
    "truck":      1,   # small truck treated as 4-wheel; change to 2 for large-vehicle behaviour
    "bus":        2,   # large vehicle
    "person":     4,   # pedestrian
    "rider":      3,   # motorcycle / bicycle rider
    "bicycle":    3,   # bicycle
    "motorcycle": 3,   # motorcycle
}

# Confidence category thresholds (mirrors the lambda in the old node).
# Returns 0 (low), 1 (medium-low), 2 (medium-high), 3 (high).
_CONF_THRESHOLDS = [(55, 70, 1), (70, 85, 2), (85, 101, 3)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_topic_key(class_name: str) -> str | None:
    """Return the routing key for *class_name*, or None if not routed."""
    return _CLASS_TO_TOPIC.get(class_name)


def get_object_type(class_name: str) -> int:
    """Return the ObjObjectType integer for *class_name* (default 0 = Unknown)."""
    return _CLASS_TO_OBJ_TYPE.get(class_name, 0)


def categorize_confidence(conf_0_to_1: float) -> int:
    """
    Map a [0, 1] confidence score to a 4-level category integer.

    Returns:
        0  below 55 % — low confidence
        1  55–69 %    — medium-low
        2  70–84 %    — medium-high
        3  85–100 %   — high
    """
    pct = conf_0_to_1 * 100
    for lo, hi, cat in _CONF_THRESHOLDS:
        if lo <= pct < hi:
            return cat
    return 0


# All class names the model is expected to emit.  Used by the fusion node
# to validate detections and warn on unexpected class names at startup.
KNOWN_CLASSES: frozenset[str] = frozenset(_CLASS_TO_TOPIC.keys()).union(
    # Known classes that are intentionally not routed:
    # (none currently — add any you want to silently accept but not publish)
)
