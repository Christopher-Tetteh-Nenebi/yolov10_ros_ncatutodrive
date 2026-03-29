#!/usr/bin/env python3
"""
Scoring CAN Node
================
Subscribes to all six perception scoring topics published by
``perception_fusion_node.py`` and logs the relevant fields to the terminal.

Topics
------
  /objects_scoring              (perception_msgs/Objects)
  /TrafficSigns_scoring         (perception_msgs/TrafficSigns)
  /TrafficSignalsHead_scoring   (perception_msgs/TrafficSignalsHead)
  /LaneMarkings_scoring         (perception_msgs/LaneMarkings)
  /LimitLines_scoring           (perception_msgs/LimitLines)
  /road_state_scoring           (perception_msgs/road_state)
"""

import rospy

from perception_msgs.msg import (
    Objects,
    TrafficSigns,
    TrafficSignalsHead,
    LaneMarkings,
    LimitLines,
    road_state,
)

# ── Object type integer → human-readable label ────────────────────────────────
# Matches the mapping in bdd100k_class_map.get_object_type().
_OBJECT_TYPE_LABELS = {
    0: "unknown",
    1: "car",
    2: "truck",
    3: "bus",
    4: "person",
    5: "bicycle",
    6: "motorcycle",
    7: "animal",
}

# ── Sign type integer → human-readable label ──────────────────────────────────
_SIGN_TYPE_LABELS = {
    0:  "unknown",
    1:  "stop",
    2:  "yield",
    3:  "speed_limit",
    4:  "no_left_turn",
    5:  "no_right_turn",
    6:  "no_u_turn",
    7:  "no_straight",
    8:  "go_straight_only",
    9:  "turn_left_only",
    10: "turn_right_only",
    11: "pedestrian_crossing",
    12: "railroad_crossing",
    13: "roadwork",
    14: "detour",
    15: "do_not_enter",
}

# ── Confidence integer → label (matches categorize_confidence()) ──────────────
_CONFIDENCE_LABELS = {
    1: "low",
    2: "medium",
    3: "high",
}


def _conf_str(conf_int: int) -> str:
    return _CONFIDENCE_LABELS.get(conf_int, str(conf_int))


# ── Callbacks ─────────────────────────────────────────────────────────────────

def objects_callback(msg: Objects):
    rospy.loginfo(
        "[scoring] Objects — count=%d  t=%.1f s",
        msg.Current_Object_Count,
        msg.Observation_Time_of_Hour,
    )
    for t in msg.objects_tracks:
        obj_label = _OBJECT_TYPE_LABELS.get(t.ObjObjectType, "type_%d" % t.ObjObjectType)
        rospy.loginfo(
            "  obj id=%d  type=%s  conf=%s  long=%.2f m  lat=%.2f m  w=%.0f  h=%.0f  src_cam=%d  src_lidar=%d",
            t.ObjObjectIdA,
            obj_label,
            _conf_str(t.Confidence),
            t.LongPos,
            t.LatPos,
            t.Width,
            t.Height,
            t.Object_Source_Camera,
            t.Object_Source_Lidar,
        )


def traffic_signs_callback(msg: TrafficSigns):
    rospy.loginfo(
        "[scoring] TrafficSigns — count=%d  t=%.1f s",
        msg.Current_Sign_Count,
        msg.Observation_Time_of_Hour,
    )
    for t in msg.trafficSign_track:
        sign_label = _SIGN_TYPE_LABELS.get(t.Sign_Type, "sign_%d" % t.Sign_Type)
        rospy.loginfo(
            "  sign id=%d  type=%s  conf=%s  long=%.2f m  lat=%.2f m",
            t.SignObjectID,
            sign_label,
            _conf_str(t.Confidence),
            t.LongPos,
            t.LatPos,
        )


def traffic_signals_head_callback(msg: TrafficSignalsHead):
    rospy.loginfo(
        "[scoring] TrafficSignalsHead — count=%d  t=%.1f s",
        msg.Current_Signal_Head_Count,
        msg.Observation_Time_of_Hour,
    )
    for t in msg.trafficSignalsHead_track:
        # Find which illumination field is active (set to 1).
        illum_fields = [
            "IllumLtRedBall", "IllumLtGreenBall", "IllumLtYellowBall",
            "IllumLtRedLeftArrow", "IllumLtGreenLeftArrow", "IllumLtYellowLeftArrow",
        ]
        active = [f for f in illum_fields if getattr(t, f, 0) == 1]
        state_str = ", ".join(active) if active else "unknown"
        rospy.loginfo(
            "  signal id=%d  state=%s  conf=%s  long=%.2f m  lat=%.2f m",
            t.SignalObjectID,
            state_str,
            _conf_str(t.Confidence),
            t.LongPos,
            t.LatPos,
        )


def lane_markings_callback(msg: LaneMarkings):
    rospy.loginfo("[scoring] LaneMarkings — count=%d", len(msg.laneMarking_track))


def limit_lines_callback(msg: LimitLines):
    rospy.loginfo("[scoring] LimitLines — count=%d", len(msg.limitline_track))


def road_state_callback(msg: road_state):
    rospy.loginfo("[scoring] road_state received")


# ── Node setup ────────────────────────────────────────────────────────────────

def main():
    rospy.init_node("Scoring_CAN_Node", anonymous=False)

    rospy.Subscriber("/objects_scoring",            Objects,           objects_callback)
    rospy.Subscriber("/TrafficSigns_scoring",        TrafficSigns,      traffic_signs_callback)
    rospy.Subscriber("/TrafficSignalsHead_scoring",  TrafficSignalsHead, traffic_signals_head_callback)
    rospy.Subscriber("/LaneMarkings_scoring",        LaneMarkings,      lane_markings_callback)
    rospy.Subscriber("/LimitLines_scoring",          LimitLines,        limit_lines_callback)
    rospy.Subscriber("/road_state_scoring",          road_state,        road_state_callback)

    rospy.loginfo("[scoring] Scoring CAN node started — listening on all scoring topics.")
    rospy.spin()


if __name__ == "__main__":
    main()
