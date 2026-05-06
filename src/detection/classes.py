"""Canonical class taxonomy for the pipeline.

The general COCO-trained YOLO11 detector emits its own integer class IDs.
We translate those into a stable enum the rest of the pipeline uses, so
that swapping detectors (e.g. moving to a custom-trained Indian-traffic
model) only changes the mapping in this file.
"""
from __future__ import annotations

from enum import Enum

# COCO class indices we care about (Ultralytics yolo*.pt default classes).
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7
COCO_TRAFFIC_LIGHT = 9
# Luggage classes — used by AbandonedObjectDetector owner-association
COCO_BACKPACK = 24
COCO_HANDBAG = 26
COCO_SUITCASE = 28


class ObjectClass(str, Enum):
    PERSON = "person"
    TWO_WHEELER = "two_wheeler"          # motorcycle, scooter, bike
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    TRAFFIC_LIGHT = "traffic_light"
    LICENSE_PLATE = "license_plate"
    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    LUGGAGE = "luggage"                  # backpack, handbag, suitcase


COCO_TO_CLASS: dict[int, ObjectClass] = {
    COCO_PERSON: ObjectClass.PERSON,
    COCO_BICYCLE: ObjectClass.TWO_WHEELER,
    COCO_MOTORCYCLE: ObjectClass.TWO_WHEELER,
    COCO_CAR: ObjectClass.CAR,
    COCO_BUS: ObjectClass.BUS,
    COCO_TRUCK: ObjectClass.TRUCK,
    COCO_TRAFFIC_LIGHT: ObjectClass.TRAFFIC_LIGHT,
    COCO_BACKPACK: ObjectClass.LUGGAGE,
    COCO_HANDBAG: ObjectClass.LUGGAGE,
    COCO_SUITCASE: ObjectClass.LUGGAGE,
}

VEHICLE_CLASSES = {
    ObjectClass.TWO_WHEELER,
    ObjectClass.CAR,
    ObjectClass.BUS,
    ObjectClass.TRUCK,
}
