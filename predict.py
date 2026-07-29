from ultralytics import YOLO
from compliance import evaluate_compliance
import cv2

# -----------------------------------
# Load model once
# -----------------------------------
model = YOLO("best.pt")


def analyze_image(image_path):

    # -------------------------------
    # Run YOLO
    # -------------------------------
    result = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        verbose=False
    )[0]

    # -------------------------------
    # Convert detections
    # -------------------------------
    detections = []

    for box in result.boxes:

        cls = int(box.cls[0])

        detections.append(
            {
                "class_name": model.names[cls],
                "box": box.xyxy[0].cpu().numpy().tolist()
            }
        )

    # -------------------------------
    # Compliance Engine
    # -------------------------------
    workers = evaluate_compliance(detections)

    # -------------------------------
    # Sort: VIOLATION workers first, SAFE workers after
    # -------------------------------
    workers = sorted(
        workers,
        key=lambda w: 0 if w.status == "VIOLATION" else 1
    )

    # -------------------------------
    # Read original image
    # -------------------------------
    image = cv2.imread(image_path)

    h, w = image.shape[:2]

    compliant = 0
    violation = 0

    # -------------------------------
    # Sort workers left-to-right just for label placement,
    # so "staggering" alternates correctly across the row.
    # (drawing order / numbering still follows violation-first order above)
    # -------------------------------
    order_by_x = sorted(
        range(len(workers)),
        key=lambda idx: workers[idx].person_box[0]
    )
    stagger_rank = {idx: rank for rank, idx in enumerate(order_by_x)}

    # ===============================
    # Draw Workers
    # ===============================
    for i, worker in enumerate(workers):

        x1, y1, x2, y2 = map(int, worker.person_box)

        if worker.status == "SAFE":
            color = (0, 255, 0)
            compliant += 1
        else:
            color = (0, 0, 255)
            violation += 1

        # Worker box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # ----------------------------
        # Label (centered above box, clamped to image,
        # staggered up/down so neighboring labels never collide)
        # ----------------------------
        label = f"Worker {i+1}: {worker.status}"

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        box_center_x = (x1 + x2) // 2

        label_x1 = box_center_x - (tw + 10) // 2
        label_x2 = label_x1 + tw + 10

        # keep label inside image horizontally
        if label_x1 < 0:
            label_x1 = 0
            label_x2 = tw + 10

        if label_x2 > w:
            label_x2 = w
            label_x1 = w - (tw + 10)

        # stagger every other worker (by left-to-right position) higher up
        row_height = th + 20
        is_odd_row = stagger_rank[i] % 2 == 1
        extra_offset = row_height if is_odd_row else 0

        label_y = max(th + 15, y1 - 10 - extra_offset)

        cv2.rectangle(
            image,
            (label_x1, label_y - th - 8),
            (label_x2, label_y + 4),
            color,
            -1
        )

        text_x = label_x1 + 5

        cv2.putText(
            image,
            label,
            (text_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------
        # Missing PPE
        # ----------------------------
        if worker.status == "VIOLATION":

            yy = y2 + 25

            cv2.putText(
                image,
                "Missing:",
                (x1, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            for item in worker.missing:

                yy += 25

                cv2.putText(
                    image,
                    f"- {item}",
                    (x1 + 10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    # ===============================
    # Top Dashboard
    # ===============================
    banner = (
        f"Workers: {len(workers)} | "
        f"SAFE: {compliant} | "
        f"Violations: {violation}"
    )

    cv2.rectangle(
        image,
        (0, 0),
        (w, 50),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        banner,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # ===============================
    # Save image
    # ===============================
    output_path = "result_annotated.jpg"

    cv2.imwrite(output_path, image)

    return output_path, workers