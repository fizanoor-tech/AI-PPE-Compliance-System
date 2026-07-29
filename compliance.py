from dataclasses import dataclass, field
from typing import List

# -------------------------------------------------------
# Data class for each worker
# -------------------------------------------------------

@dataclass
class WorkerCompliance:
    person_box: list
    helmet: bool = False
    vest: bool = False
    gloves: bool = False
    boots: bool = False
    goggles: bool = False
    status: str = "VIOLATION"
    missing: List[str] = field(default_factory=list)


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_inside(point, region):
    px, py = point
    x1, y1, x2, y2 = region
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def body_regions(person_box):
    x1, y1, x2, y2 = person_box

    width = x2 - x1
    height = y2 - y1

    head = [x1, y1, x2, y1 + height * 0.25]
    torso = [x1, y1 + height * 0.20, x2, y1 + height * 0.65]
    feet = [x1, y1 + height * 0.70, x2, y2]
    arms = [
        x1 - width * 0.20,
        y1 + height * 0.20,
        x2 + width * 0.20,
        y1 + height * 0.80,
    ]

    return head, torso, feet, arms


# -------------------------------------------------------
# Compliance Engine
# -------------------------------------------------------

def evaluate_compliance(detections, required_items={"helmet", "vest"}):

    persons = []
    helmets = []
    vests = []
    gloves = []
    boots = []
    goggles = []

    # Separate detections by class
    for det in detections:

        cls = det["class_name"].lower()

        if cls == "person":
            persons.append(det)

        elif cls == "helmet":
            helmets.append(det)

        elif cls == "vest":
            vests.append(det)

        elif cls == "gloves":
            gloves.append(det)

        elif cls == "boots":
            boots.append(det)

        elif cls == "goggles":
            goggles.append(det)

    workers = []

    # Check PPE for every detected person
    for person in persons:

        worker = WorkerCompliance(person_box=person["box"])

        head, torso, feet, arms = body_regions(person["box"])

        worker.helmet = any(
            point_inside(center(item["box"]), head)
            for item in helmets
        )

        worker.vest = any(
            point_inside(center(item["box"]), torso)
            for item in vests
        )

        worker.gloves = any(
            point_inside(center(item["box"]), arms)
            for item in gloves
        )

        worker.boots = any(
            point_inside(center(item["box"]), feet)
            for item in boots
        )

        worker.goggles = any(
            point_inside(center(item["box"]), head)
            for item in goggles
        )

        detected_map = {
            "helmet": worker.helmet,
            "vest": worker.vest,
            "gloves": worker.gloves,
            "boots": worker.boots,
            "goggles": worker.goggles,
        }

        for req in required_items:

            if not detected_map.get(req, False):
                worker.missing.append(req.capitalize())

        worker.status = (
            "SAFE"
            if len(worker.missing) == 0
            else "VIOLATION"
        )

        workers.append(worker)

    return workers