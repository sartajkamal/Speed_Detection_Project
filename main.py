import cv2
import numpy as np
import json
import os
import pandas as pd
import warnings
import time
import argparse
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from datetime import datetime
from ultralytics import YOLO
import easyocr

warnings.filterwarnings("ignore")


# ============================================================
# ADMIN CONFIGURATION - FR-8
# ============================================================

class AdminConfiguration:
    def __init__(self, config_file="admin_config.json"):
        self.config_file = config_file

        self.config = {
            "camera_id": "CAM-01",
            "location_name": "Main Road Camera",
            "video_source": (
                r"C:\Users\Ainee\OneDrive\Desktop"
                r"\Speed_Detection_Project\videos"
                r"\The green and Red light traffic.mp4"
            ),
            "speed_limit_kmh": 50.0,
            "violation_threshold_kmh": 0.0,

            # OCR settings
            "min_ocr_confidence": 0.20,
            "min_blur_threshold": 50.0,
            "ocr_scale_width": 1000,
            "ocr_sampling_frames": 15,

            # Detection
            "yolo_imgsz": 480,

            # Evidence
            "save_full_frame_evidence": True
        }

        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.config.update(saved)
            except Exception as e:
                print(f"[WARNING] Could not load config: {e}")
                self.save_config()
        else:
            self.save_config()

    def save_config(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    def get(self, key):
        return self.config.get(key)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()


# ============================================================
# VIOLATION DATA / REVIEW / SEARCH - FR-7 + FR-9
# ============================================================

class ViolationManager:
    def __init__(self, output_dir="violations_evidence"):
        self.output_dir = output_dir
        self.json_file = os.path.join(output_dir, "violations.json")
        self.csv_file = os.path.join(output_dir, "violations.csv")

    def load_records(self):
        if not os.path.exists(self.json_file):
            return []

        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[ERROR] Could not load violation records: {e}")
            return []

    def save_records(self, records):
        os.makedirs(self.output_dir, exist_ok=True)

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)

        pd.DataFrame(records).to_csv(self.csv_file, index=False)

    def search_and_filter(
        self,
        plate="",
        date_text="",
        min_speed=None,
        max_speed=None,
        location="",
        camera_id=""
    ):
        records = self.load_records()
        results = []

        plate = plate.strip().upper()
        location = location.strip().lower()
        camera_id = camera_id.strip().upper()
        date_text = date_text.strip()

        for record in records:
            record_plate = str(
                record.get("license_plate", "")
            ).upper()

            record_date = str(
                record.get("date", "")
                or str(record.get("timestamp", ""))[:10]
            )

            record_location = str(
                record.get("location", "")
            ).lower()

            record_camera = str(
                record.get("camera_id", "")
            ).upper()

            try:
                speed = float(
                    record.get("measured_speed_kmh", 0)
                )
            except (ValueError, TypeError):
                speed = 0.0

            if plate and plate not in record_plate:
                continue

            if date_text and date_text not in record_date:
                continue

            if location and location not in record_location:
                continue

            if camera_id and camera_id not in record_camera:
                continue

            if min_speed is not None and speed < min_speed:
                continue

            if max_speed is not None and speed > max_speed:
                continue

            results.append(record)

        return results

    def update_record(self, violation_id, **changes):
        records = self.load_records()

        found = False

        for record in records:
            if str(record.get("violation_id")) == str(violation_id):
                record.update(changes)
                record["last_reviewed_at"] = datetime.now().isoformat()
                found = True
                break

        if found:
            self.save_records(records)

    def review_confirm(self, violation_id):
        self.update_record(
            violation_id,
            review_status="Confirmed"
        )

    def review_dismiss(self, violation_id):
        self.update_record(
            violation_id,
            review_status="Dismissed"
        )

    def correct_plate(self, violation_id, plate):
        self.update_record(
            violation_id,
            license_plate=plate.upper().strip(),
            review_status="Corrected"
        )


# ============================================================
# SPEED + PLATE DETECTOR
# ============================================================

class SpeedAndPlateDetector:

    def __init__(
        self,
        model_path="yolov8n.pt",
        admin_config=None,
        output_dir="violations_evidence"
    ):
        self.admin_config = admin_config or AdminConfiguration()

        self.camera_id = self.admin_config.get("camera_id")
        self.location_name = self.admin_config.get("location_name")

        self.speed_limit = float(
            self.admin_config.get("speed_limit_kmh")
        )

        self.violation_threshold = float(
            self.admin_config.get("violation_threshold_kmh")
        )

        self.min_ocr_confidence = float(
            self.admin_config.get("min_ocr_confidence")
        )

        self.min_blur_threshold = float(
            self.admin_config.get("min_blur_threshold")
        )

        self.ocr_scale_width = int(
            self.admin_config.get("ocr_scale_width")
        )

        self.ocr_sampling_frames = int(
            self.admin_config.get("ocr_sampling_frames")
        )

        self.yolo_imgsz = int(
            self.admin_config.get("yolo_imgsz")
        )

        self.save_full_frame_evidence = bool(
            self.admin_config.get("save_full_frame_evidence")
        )

        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(
            os.path.join(self.output_dir, "crops"),
            exist_ok=True
        )
        os.makedirs(
            os.path.join(self.output_dir, "enhanced"),
            exist_ok=True
        )
        os.makedirs(
            os.path.join(self.output_dir, "frames"),
            exist_ok=True
        )

        print("[INFO] Loading YOLO model...")
        self.model = YOLO(model_path)

        print("[INFO] Loading EasyOCR...")
        self.reader = easyocr.Reader(["en"], gpu=False)

        self.track_history = {}
        self.flagged_vehicles = set()
        self.violation_records = []

        # Tracks that are being sampled for better OCR/evidence.
        self.ocr_sampling = {}

        # Vehicle classes: car, motorcycle, bus, truck
        self.target_classes = [2, 3, 5, 7]

        # NOTE:
        # These perspective points MUST be calibrated for the actual
        # camera/video and actual road distances. The values below are
        # the existing project defaults, not a measured calibration.
        self.src_points = np.float32([
            [100, 300],
            [500, 300],
            [600, 600],
            [0, 600]
        ])

        self.dst_points = np.float32([
            [0, 0],
            [10, 0],
            [10, 40],
            [0, 40]
        ])

        self.M = cv2.getPerspectiveTransform(
            self.src_points,
            self.dst_points
        )

    # --------------------------------------------------------
    # FR-3: Perspective / Speed
    # --------------------------------------------------------

    def _transform_point(self, point):
        pt = np.array(
            [[[point[0], point[1]]]],
            dtype=np.float32
        )

        transformed = cv2.perspectiveTransform(
            pt,
            self.M
        )

        return transformed[0][0]

    def _calculate_speed(self, history):
        if len(history) < 5:
            return 0.0

        p1, t1 = history[-5]
        p2, t2 = history[-1]

        dt = t2 - t1

        if dt <= 0:
            return 0.0

        m1 = self._transform_point(p1)
        m2 = self._transform_point(p2)

        distance_meters = np.linalg.norm(m2 - m1)

        speed_mps = distance_meters / dt

        return round(
            float(speed_mps * 3.6),
            1
        )

    # --------------------------------------------------------
    # Image quality / enhancement
    # --------------------------------------------------------

    def _calculate_blur_score(self, image):
        if image is None or image.size == 0:
            return 0.0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        return float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

    def _enhance_frame_for_ocr(self, image):
        """
        Enhancement requested by the team lead:
        - upscale small vehicle/plate regions
        - denoise
        - CLAHE contrast enhancement
        - unsharp masking
        - mild adaptive threshold variant

        The function returns several OCR-ready variants rather than
        modifying the original evidence image.
        """

        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        # Upscale small crops.
        if w < self.ocr_scale_width:
            scale = self.ocr_scale_width / max(w, 1)
            new_w = self.ocr_scale_width
            new_h = max(1, int(h * scale))

            upscaled = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC
            )
        else:
            upscaled = image.copy()

        gray = cv2.cvtColor(
            upscaled,
            cv2.COLOR_BGR2GRAY
        )

        # Denoising before sharpening.
        denoised = cv2.fastNlMeansDenoising(
            gray,
            None,
            7,
            7,
            21
        )

        # Contrast enhancement.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(denoised)

        # Unsharp mask.
        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            1.5
        )

        sharpened = cv2.addWeighted(
            enhanced,
            1.7,
            blurred,
            -0.7,
            0
        )

        # Otsu threshold.
        _, otsu = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Adaptive threshold.
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7
        )

        return [
            ("original_upscaled", upscaled),
            ("clahe", enhanced),
            ("sharpened", sharpened),
            ("otsu", otsu),
            ("adaptive", adaptive)
        ]

    # --------------------------------------------------------
    # FR-5: Enhanced OCR
    # --------------------------------------------------------

    def _ocr_single_image(self, image):
        if image is None or image.size == 0:
            return "", 0.0, None

        try:
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
        except Exception:
            return "", 0.0, None

        best_text = ""
        best_conf = 0.0
        best_bbox = None

        for bbox, text, conf in results:
            cleaned = "".join(
                c for c in text
                if c.isalnum()
            ).upper()

            try:
                confidence = float(conf)
            except (ValueError, TypeError):
                confidence = 0.0

            # Avoid accepting extremely short noise.
            if len(cleaned) >= 4 and confidence > best_conf:
                best_text = cleaned
                best_conf = confidence
                best_bbox = bbox

        return (
            best_text,
            round(best_conf, 2),
            best_bbox
        )

    def _read_license_plate(self, vehicle_crop):
        """
        OCR is performed on multiple enhanced versions of the
        vehicle region.

        This directly implements the team lead's instruction to
        enhance frames where the plate is visible and compare OCR
        performance.
        """

        if vehicle_crop is None or vehicle_crop.size == 0:
            return "", 0.0, None, "", None

        variants = self._enhance_frame_for_ocr(
            vehicle_crop
        )

        best_text = ""
        best_conf = 0.0
        best_variant = ""
        best_image = None
        best_bbox = None

        for variant_name, image in variants:

            text, conf, bbox = self._ocr_single_image(
                image
            )

            if conf > best_conf:
                best_text = text
                best_conf = conf
                best_variant = variant_name
                best_image = image
                best_bbox = bbox

        return (
            best_text,
            round(best_conf, 2),
            best_bbox,
            best_variant,
            best_image
        )

    # --------------------------------------------------------
    # Evidence helpers
    # --------------------------------------------------------

    def _save_evidence(
        self,
        track_id,
        video_timestamp,
        frame,
        crop,
        enhanced_image=None,
        prefix="violation"
    ):
        safe_time = int(
            round(video_timestamp * 100)
        )

        crop_filename = (
            f"{prefix}_{self.camera_id}_"
            f"{track_id}_{safe_time}.jpg"
        )

        crop_path = os.path.join(
            self.output_dir,
            "crops",
            crop_filename
        )

        if crop is not None and crop.size > 0:
            cv2.imwrite(
                crop_path,
                crop
            )

        frame_path = ""

        if self.save_full_frame_evidence:
            frame_filename = (
                f"{prefix}_{self.camera_id}_"
                f"{track_id}_{safe_time}_full.jpg"
            )

            frame_path = os.path.join(
                self.output_dir,
                "frames",
                frame_filename
            )

            if frame is not None and frame.size > 0:
                cv2.imwrite(
                    frame_path,
                    frame
                )

        enhanced_path = ""

        if enhanced_image is not None and enhanced_image.size > 0:
            enhanced_filename = (
                f"{prefix}_{self.camera_id}_"
                f"{track_id}_{safe_time}_enhanced.jpg"
            )

            enhanced_path = os.path.join(
                self.output_dir,
                "enhanced",
                enhanced_filename
            )

            cv2.imwrite(
                enhanced_path,
                enhanced_image
            )

        return (
            crop_path,
            frame_path,
            enhanced_path
        )

    # --------------------------------------------------------
    # Create final violation record
    # --------------------------------------------------------

    def _create_violation_record(
        self,
        track_id,
        speed,
        video_timestamp,
        frame,
        crop,
        plate_text,
        ocr_conf,
        blur_score,
        ocr_variant,
        enhanced_image
    ):
        if (
            ocr_conf >= self.min_ocr_confidence
            and plate_text
        ):
            final_plate = plate_text
        else:
            final_plate = "UNKNOWN"

        (
            crop_path,
            frame_path,
            enhanced_path
        ) = self._save_evidence(
            track_id,
            video_timestamp,
            frame,
            crop,
            enhanced_image,
            prefix="violation"
        )

        now = datetime.now()

        record = {
            "violation_id": (
                f"V-{self.camera_id}-"
                f"{track_id}-"
                f"{int(video_timestamp * 100)}"
            ),

            # Actual system timestamp.
            "timestamp": now.isoformat(),

            # Actual position inside the source video.
            "video_timestamp_seconds": round(
                video_timestamp,
                2
            ),

            "date": now.strftime(
                "%Y-%m-%d"
            ),

            "camera_id": self.camera_id,
            "location": self.location_name,

            "track_id": int(track_id),

            "measured_speed_kmh": speed,
            "speed_limit_kmh": self.speed_limit,

            "violation_threshold_kmh": (
                self.violation_threshold
            ),

            "over_limit_kmh": round(
                speed - self.speed_limit,
                2
            ),

            "license_plate": final_plate,
            "ocr_confidence": ocr_conf,

            "ocr_status": (
                "Accepted"
                if final_plate != "UNKNOWN"
                else "Needs Review"
            ),

            "ocr_enhancement": ocr_variant,

            "blur_score": round(
                blur_score,
                2
            ),

            "evidence_image_path": crop_path,
            "full_frame_evidence_path": frame_path,
            "enhanced_evidence_path": enhanced_path,

            "review_status": "Pending",
            "reviewer_notes": ""
        }

        self.violation_records.append(
            record
        )

        return record

    # --------------------------------------------------------
    # FR-1 to FR-6: Video processing
    # --------------------------------------------------------

    def _process_ai_video(
        self,
        video_path,
        display=True
    ):
        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():
            print(
                f"[ERROR] Unable to open video: "
                f"{video_path}"
            )
            return

        fps = cap.get(
            cv2.CAP_PROP_FPS
        ) or 30.0

        frame_idx = 0
        processed_frames = 0
        total_yolo_time = 0.0

        print(
            f"[INFO] Started processing footage: "
            f"{video_path} ({fps:.1f} FPS)"
        )

        print(
            f"[CONFIG] Camera={self.camera_id} | "
            f"Location={self.location_name} | "
            f"Speed Limit={self.speed_limit} km/h | "
            f"YOLO image size={self.yolo_imgsz}"
        )

        print(
            f"[OCR CONFIG] Min confidence="
            f"{self.min_ocr_confidence} | "
            f"Sampling frames="
            f"{self.ocr_sampling_frames}"
        )

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            video_timestamp = (
                frame_idx / fps
            )

            start_time = time.perf_counter()

            results = self.model.track(
                frame,
                persist=True,
                classes=self.target_classes,
                verbose=False,
                imgsz=self.yolo_imgsz
            )

            yolo_time = (
                time.perf_counter()
                - start_time
            )

            total_yolo_time += yolo_time
            processed_frames += 1

            if processed_frames % 30 == 0:

                average = (
                    total_yolo_time
                    / processed_frames
                )

                print(
                    f"[PERFORMANCE] "
                    f"Frame: {frame_idx} | "
                    f"YOLO time: "
                    f"{yolo_time:.3f} sec | "
                    f"Average: "
                    f"{average:.3f} sec"
                )

            current_track_ids = set()

            if (
                results[0].boxes is not None
                and results[0].boxes.id is not None
            ):

                boxes = (
                    results[0]
                    .boxes
                    .xyxy
                    .cpu()
                    .numpy()
                )

                track_ids = (
                    results[0]
                    .boxes
                    .id
                    .int()
                    .cpu()
                    .numpy()
                )

                for box, track_id in zip(
                    boxes,
                    track_ids
                ):

                    track_id = int(
                        track_id
                    )

                    current_track_ids.add(
                        track_id
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    x1 = max(
                        0,
                        min(
                            x1,
                            frame.shape[1] - 1
                        )
                    )

                    x2 = max(
                        0,
                        min(
                            x2,
                            frame.shape[1]
                        )
                    )

                    y1 = max(
                        0,
                        min(
                            y1,
                            frame.shape[0] - 1
                        )
                    )

                    y2 = max(
                        0,
                        min(
                            y2,
                            frame.shape[0]
                        )
                    )

                    if x2 <= x1 or y2 <= y1:
                        continue

                    bottom_center = (
                        (x1 + x2) // 2,
                        y2
                    )

                    if (
                        track_id
                        not in self.track_history
                    ):
                        self.track_history[
                            track_id
                        ] = []

                    self.track_history[
                        track_id
                    ].append(
                        (
                            bottom_center,
                            video_timestamp
                        )
                    )

                    if len(
                        self.track_history[
                            track_id
                        ]
                    ) > 10:

                        self.track_history[
                            track_id
                        ].pop(0)

                    speed = self._calculate_speed(
                        self.track_history[
                            track_id
                        ]
                    )

                    box_color = (
                        0,
                        255,
                        0
                    )

                    trigger_speed = (
                        self.speed_limit
                        + self.violation_threshold
                    )

                    # ------------------------------------------------
                    # Speed violation starts
                    # ------------------------------------------------

                    if speed > trigger_speed:

                        box_color = (
                            0,
                            0,
                            255
                        )

                        crop = frame[
                            y1:y2,
                            x1:x2
                        ]

                        # First time this vehicle violates:
                        if (
                            track_id
                            not in self.flagged_vehicles
                        ):

                            self.flagged_vehicles.add(
                                track_id
                            )

                            blur_score = (
                                self._calculate_blur_score(
                                    crop
                                )
                            )

                            (
                                plate_text,
                                ocr_conf,
                                best_bbox,
                                best_variant,
                                best_enhanced
                            ) = self._read_license_plate(
                                crop
                            )

                            self.ocr_sampling[
                                track_id
                            ] = {
                                "remaining": max(
                                    0,
                                    self.ocr_sampling_frames - 1
                                ),
                                "best_plate": plate_text,
                                "best_conf": ocr_conf,
                                "best_bbox": best_bbox,
                                "best_variant": best_variant,
                                "best_enhanced": best_enhanced,
                                "best_crop": crop.copy(),
                                "best_frame": frame.copy(),
                                "best_timestamp": video_timestamp,
                                "best_speed": speed,
                                "best_blur": blur_score
                            }

                            print(
                                f"[ALERT] "
                                f"Speeding detected! "
                                f"ID: {track_id} | "
                                f"Speed: {speed} km/h | "
                                f"Initial Plate: "
                                f"{plate_text or 'NONE'} | "
                                f"OCR Confidence: "
                                f"{ocr_conf}"
                            )

                        # --------------------------------------------
                        # Continue sampling later frames.
                        # This is the main OCR improvement requested
                        # by the team lead.
                        # --------------------------------------------

                        elif (
                            track_id
                            in self.ocr_sampling
                        ):

                            sample = (
                                self.ocr_sampling[
                                    track_id
                                ]
                            )

                            if sample["remaining"] > 0:

                                (
                                    plate_text,
                                    ocr_conf,
                                    best_bbox,
                                    best_variant,
                                    best_enhanced
                                ) = self._read_license_plate(
                                    crop
                                )

                                blur_score = (
                                    self._calculate_blur_score(
                                        crop
                                    )
                                )

                                # Prefer higher OCR confidence.
                                # If confidence is equal, prefer
                                # the sharper crop.
                                better = (
                                    ocr_conf
                                    > sample["best_conf"]
                                )

                                equal_but_sharper = (
                                    abs(
                                        ocr_conf
                                        - sample["best_conf"]
                                    ) < 0.001
                                    and blur_score
                                    > sample["best_blur"]
                                )

                                if (
                                    better
                                    or equal_but_sharper
                                ):

                                    sample[
                                        "best_plate"
                                    ] = plate_text

                                    sample[
                                        "best_conf"
                                    ] = ocr_conf

                                    sample[
                                        "best_bbox"
                                    ] = best_bbox

                                    sample[
                                        "best_variant"
                                    ] = best_variant

                                    sample[
                                        "best_enhanced"
                                    ] = best_enhanced

                                    sample[
                                        "best_crop"
                                    ] = crop.copy()

                                    sample[
                                        "best_frame"
                                    ] = frame.copy()

                                    sample[
                                        "best_timestamp"
                                    ] = video_timestamp

                                    sample[
                                        "best_speed"
                                    ] = speed

                                    sample[
                                        "best_blur"
                                    ] = blur_score

                                sample[
                                    "remaining"
                                ] -= 1

                        # --------------------------------------------
                        # Finalize after sampling window.
                        # --------------------------------------------

                        if (
                            track_id
                            in self.ocr_sampling
                            and self.ocr_sampling[
                                track_id
                            ]["remaining"] <= 0
                        ):

                            sample = (
                                self.ocr_sampling.pop(
                                    track_id
                                )
                            )

                            final_record = (
                                self._create_violation_record(
                                    track_id=track_id,
                                    speed=sample[
                                        "best_speed"
                                    ],
                                    video_timestamp=sample[
                                        "best_timestamp"
                                    ],
                                    frame=sample[
                                        "best_frame"
                                    ],
                                    crop=sample[
                                        "best_crop"
                                    ],
                                    plate_text=sample[
                                        "best_plate"
                                    ],
                                    ocr_conf=sample[
                                        "best_conf"
                                    ],
                                    blur_score=sample[
                                        "best_blur"
                                    ],
                                    ocr_variant=sample[
                                        "best_variant"
                                    ],
                                    enhanced_image=sample[
                                        "best_enhanced"
                                    ]
                                )
                            )

                            print(
                                f"[OCR RESULT] "
                                f"ID: {track_id} | "
                                f"Plate: "
                                f"{final_record['license_plate']} | "
                                f"Confidence: "
                                f"{final_record['ocr_confidence']} | "
                                f"Enhancement: "
                                f"{final_record['ocr_enhancement']} | "
                                f"Blur score: "
                                f"{final_record['blur_score']}"
                            )

                    # ------------------------------------------------
                    # Drawing
                    # ------------------------------------------------

                    label = (
                        f"ID: {track_id} | "
                        f"{speed} km/h"
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    (w, h), _ = (
                        cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            2
                        )
                    )

                    if y1 - 25 > 20:

                        box_y1 = y1 - 22
                        box_y2 = y1
                        text_y = y1 - 7

                    else:

                        box_y1 = y1
                        box_y2 = (
                            y1 + h + 12
                        )
                        text_y = (
                            y1 + h + 8
                        )

                    cv2.rectangle(
                        frame,
                        (
                            x1,
                            box_y1
                        ),
                        (
                            x1 + w + 10,
                            box_y2
                        ),
                        box_color,
                        -1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1 + 5,
                            text_y
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

            # ----------------------------------------------------
            # If a sampled vehicle disappears, finalize its best
            # available result rather than losing the record.
            # ----------------------------------------------------

            disappeared = [
                tid
                for tid in list(
                    self.ocr_sampling.keys()
                )
                if tid not in current_track_ids
            ]

            for track_id in disappeared:

                sample = (
                    self.ocr_sampling.pop(
                        track_id
                    )
                )

                final_record = (
                    self._create_violation_record(
                        track_id=track_id,
                        speed=sample[
                            "best_speed"
                        ],
                        video_timestamp=sample[
                            "best_timestamp"
                        ],
                        frame=sample[
                            "best_frame"
                        ],
                        crop=sample[
                            "best_crop"
                        ],
                        plate_text=sample[
                            "best_plate"
                        ],
                        ocr_conf=sample[
                            "best_conf"
                        ],
                        blur_score=sample[
                            "best_blur"
                        ],
                        ocr_variant=sample[
                            "best_variant"
                        ],
                        enhanced_image=sample[
                            "best_enhanced"
                        ]
                    )
                )

                print(
                    f"[OCR RESULT] "
                    f"ID: {track_id} | "
                    f"Plate: "
                    f"{final_record['license_plate']} | "
                    f"Confidence: "
                    f"{final_record['ocr_confidence']} | "
                    f"Enhancement: "
                    f"{final_record['ocr_enhancement']}"
                )

            if display:

                cv2.imshow(
                    "Speed Detection & Plate Extraction",
                    frame
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    break

        cap.release()
        if display:
           cv2.destroyAllWindows()

        # Finalize any remaining samples when video ends.
        for track_id in list(
            self.ocr_sampling.keys()
        ):

            sample = self.ocr_sampling.pop(
                track_id
            )

            final_record = (
                self._create_violation_record(
                    track_id=track_id,
                    speed=sample[
                        "best_speed"
                    ],
                    video_timestamp=sample[
                        "best_timestamp"
                    ],
                    frame=sample[
                        "best_frame"
                    ],
                    crop=sample[
                        "best_crop"
                    ],
                    plate_text=sample[
                        "best_plate"
                    ],
                    ocr_conf=sample[
                        "best_conf"
                    ],
                    blur_score=sample[
                        "best_blur"
                    ],
                    ocr_variant=sample[
                        "best_variant"
                    ],
                    enhanced_image=sample[
                        "best_enhanced"
                    ]
                )
            )

            print(
                f"[OCR RESULT] "
                f"ID: {track_id} | "
                f"Plate: "
                f"{final_record['license_plate']} | "
                f"Confidence: "
                f"{final_record['ocr_confidence']} | "
                f"Enhancement: "
                f"{final_record['ocr_enhancement']}"
            )

        if processed_frames:

            average = (
                total_yolo_time
                / processed_frames
            )

            approx_fps = (
                1.0 / average
                if average > 0
                else 0
            )

            print(
                "\n[PERFORMANCE SUMMARY]"
            )

            print(
                f"Total video frames: "
                f"{frame_idx}"
            )

            print(
                f"YOLO processed frames: "
                f"{processed_frames}"
            )

            print(
                f"Average YOLO processing time: "
                f"{average:.3f} sec/frame"
            )

            print(
                f"Approximate processing FPS: "
                f"{approx_fps:.2f}"
            )

        self._export_records()

    def process_video(self, video_path, display=True):
        """Run AI in a separate process while the main process only plays video.

        The playback process NEVER waits for YOLO/EasyOCR and does not draw
        inference results onto the video. This keeps the source video moving
        at its original FPS even when CPU inference is slow.
        """
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return

        ctx = mp.get_context("spawn")
        ai_process = ctx.Process(
            target=_background_ai_process,
            args=(
                video_path,
                self.admin_config.config_file,
                self.output_dir,
                "yolov8n.pt"
            ),
            daemon=False
        )

        print("[INFO] Starting background YOLO + OCR process...")
        ai_process.start()

        # IMPORTANT: this capture belongs ONLY to playback.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Unable to open video for playback: {video_path}")
            if ai_process.is_alive():
                ai_process.terminate()
                ai_process.join(timeout=2)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / max(fps, 1.0)
        frame_count = 0
        playback_start = time.perf_counter()
        next_frame_time = playback_start

        print(f"[INFO] Started ORIGINAL playback: {video_path} ({fps:.1f} FPS)")
        print("[PLAYBACK] Playback is completely independent from YOLO/OCR.")
        print("[PLAYBACK] Press Q to stop playback.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Display the ORIGINAL frame. No AI drawing and no AI wait.
                if display:
                    cv2.imshow("Original Video Playback", frame)

                    now = time.perf_counter()
                    sleep_time = next_frame_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[INFO] Playback stopped by user.")
                        break

                    next_frame_time += frame_delay
                    # If the local machine briefly falls behind, recover
                    # immediately instead of accumulating a large delay.
                    if time.perf_counter() > next_frame_time + frame_delay * 3:
                        next_frame_time = time.perf_counter()

        finally:
            cap.release()
            if display:
              cv2.destroyAllWindows()

        playback_time = time.perf_counter() - playback_start
        actual_fps = frame_count / playback_time if playback_time > 0 else 0.0

        print("\n[PLAYBACK SUMMARY]")
        print(f"Playback frames displayed: {frame_count}")
        print(f"Playback time: {playback_time:.2f} sec")
        print(f"Average playback FPS: {actual_fps:.2f}")
        print("[INFO] YOLO + OCR continues independently in the background.")

        # Do NOT wait here. The AI process is deliberately independent.
        if ai_process.is_alive():
            print("[INFO] Background AI process is still running.")
        else:
            print("[INFO] Background AI process finished.")

    # --------------------------------------------------------
    # FR-6: Export records
    # --------------------------------------------------------

    def _export_records(self):

        json_file = os.path.join(
            self.output_dir,
            "violations.json"
        )

        csv_file = os.path.join(
            self.output_dir,
            "violations.csv"
        )

        with open(
            json_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.violation_records,
                f,
                indent=4
            )

        pd.DataFrame(
            self.violation_records
        ).to_csv(
            csv_file,
            index=False
        )

        print(
            f"[SUCCESS] Exported "
            f"{len(self.violation_records)} "
            f"violation record(s) to "
            f"'{self.output_dir}'."
        )


# ============================================================
# REVIEW + SEARCH GUI
# ============================================================

class ViolationDashboard:

    def __init__(
        self,
        root,
        admin
    ):
        self.root = root
        self.admin = admin

        self.manager = ViolationManager()

        self.root.title(
            "Speed Detection - Review & Configuration"
        )

        self.root.geometry(
            "1250x720"
        )

        self.build_ui()
        self.refresh_records()

    def build_ui(self):

        notebook = ttk.Notebook(
            self.root
        )

        notebook.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.review_tab = ttk.Frame(
            notebook
        )

        self.config_tab = ttk.Frame(
            notebook
        )

        notebook.add(
            self.review_tab,
            text="Violation Review & Search"
        )

        notebook.add(
            self.config_tab,
            text="Admin Configuration"
        )

        self.build_review_tab()
        self.build_config_tab()

    # --------------------------------------------------------
    # FR-7 + FR-9
    # --------------------------------------------------------

    def build_review_tab(self):

        filters = ttk.LabelFrame(
            self.review_tab,
            text="Search & Filter Violations"
        )

        filters.pack(
            fill="x",
            padx=10,
            pady=10
        )

        self.plate_var = tk.StringVar()
        self.date_var = tk.StringVar()
        self.min_speed_var = tk.StringVar()
        self.max_speed_var = tk.StringVar()
        self.location_var = tk.StringVar()
        self.camera_var = tk.StringVar()

        fields = [
            ("Plate", self.plate_var),
            ("Date (YYYY-MM-DD)", self.date_var),
            ("Min Speed", self.min_speed_var),
            ("Max Speed", self.max_speed_var),
            ("Location", self.location_var),
            ("Camera ID", self.camera_var)
        ]

        for i, (label, var) in enumerate(
            fields
        ):

            ttk.Label(
                filters,
                text=label
            ).grid(
                row=0,
                column=i * 2,
                padx=5,
                pady=5
            )

            ttk.Entry(
                filters,
                textvariable=var,
                width=16
            ).grid(
                row=0,
                column=i * 2 + 1,
                padx=5,
                pady=5
            )

        ttk.Button(
            filters,
            text="Search / Filter",
            command=self.refresh_records
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            padx=5,
            pady=5
        )

        ttk.Button(
            filters,
            text="Show All",
            command=self.clear_filters
        ).grid(
            row=1,
            column=2,
            columnspan=2,
            padx=5,
            pady=5
        )

        columns = (
            "violation_id",
            "plate",
            "speed",
            "limit",
            "date",
            "camera",
            "status"
        )

        self.tree = ttk.Treeview(
            self.review_tab,
            columns=columns,
            show="headings"
        )

        headings = {
            "violation_id": "Violation ID",
            "plate": "Plate",
            "speed": "Speed km/h",
            "limit": "Limit",
            "date": "Date",
            "camera": "Camera",
            "status": "Review Status"
        }

        widths = {
            "violation_id": 220,
            "plate": 120,
            "speed": 100,
            "limit": 100,
            "date": 110,
            "camera": 100,
            "status": 120
        }

        for col in columns:

            self.tree.heading(
                col,
                text=headings[col]
            )

            self.tree.column(
                col,
                width=widths[col]
            )

        self.tree.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        buttons = ttk.Frame(
            self.review_tab
        )

        buttons.pack(
            fill="x",
            padx=10,
            pady=10
        )

        ttk.Button(
            buttons,
            text="Confirm Violation",
            command=self.confirm_selected
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Dismiss Violation",
            command=self.dismiss_selected
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Correct Plate",
            command=self.correct_selected_plate
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="View Details",
            command=self.view_details
        ).pack(
            side="left",
            padx=5
        )

    def get_selected_id(self):

        selection = self.tree.selection()

        if not selection:

            messagebox.showwarning(
                "Selection Required",
                "Please select a violation first."
            )

            return None

        values = self.tree.item(
            selection[0],
            "values"
        )

        return values[0]

    def refresh_records(self):

        if not hasattr(
            self,
            "tree"
        ):
            return

        for item in self.tree.get_children():

            self.tree.delete(
                item
            )

        def to_float(value):

            try:
                return float(value)

            except (
                ValueError,
                TypeError
            ):

                return None

        results = (
            self.manager.search_and_filter(
                plate=self.plate_var.get(),
                date_text=self.date_var.get(),
                min_speed=to_float(
                    self.min_speed_var.get()
                ),
                max_speed=to_float(
                    self.max_speed_var.get()
                ),
                location=self.location_var.get(),
                camera_id=self.camera_var.get()
            )
        )

        for r in results:

            self.tree.insert(
                "",
                "end",
                values=(
                    r.get(
                        "violation_id",
                        ""
                    ),

                    r.get(
                        "license_plate",
                        "UNKNOWN"
                    ),

                    r.get(
                        "measured_speed_kmh",
                        ""
                    ),

                    r.get(
                        "speed_limit_kmh",
                        ""
                    ),

                    r.get(
                        "date",
                        ""
                    ),

                    r.get(
                        "camera_id",
                        ""
                    ),

                    r.get(
                        "review_status",
                        "Pending"
                    )
                )
            )

    def clear_filters(self):

        self.plate_var.set("")
        self.date_var.set("")
        self.min_speed_var.set("")
        self.max_speed_var.set("")
        self.location_var.set("")
        self.camera_var.set("")

        self.refresh_records()

    def confirm_selected(self):

        violation_id = (
            self.get_selected_id()
        )

        if not violation_id:
            return

        self.manager.review_confirm(
            violation_id
        )

        self.refresh_records()

        messagebox.showinfo(
            "Review Updated",
            "Violation confirmed successfully."
        )

    def dismiss_selected(self):

        violation_id = (
            self.get_selected_id()
        )

        if not violation_id:
            return

        self.manager.review_dismiss(
            violation_id
        )

        self.refresh_records()

        messagebox.showinfo(
            "Review Updated",
            "Violation dismissed successfully."
        )

    def correct_selected_plate(self):

        violation_id = (
            self.get_selected_id()
        )

        if not violation_id:
            return

        new_plate = simpledialog.askstring(
            "Correct Plate",
            "Enter the correct license plate:"
        )

        if not new_plate:
            return

        self.manager.correct_plate(
            violation_id,
            new_plate
        )

        self.refresh_records()

        messagebox.showinfo(
            "Plate Updated",
            "License plate corrected successfully."
        )

    def view_details(self):

        violation_id = (
            self.get_selected_id()
        )

        if not violation_id:
            return

        records = (
            self.manager.load_records()
        )

        selected = next(
            (
                r
                for r in records
                if str(
                    r.get(
                        "violation_id"
                    )
                ) == str(
                    violation_id
                )
            ),
            None
        )

        if not selected:
            return

        details = json.dumps(
            selected,
            indent=4
        )

        messagebox.showinfo(
            "Violation Details",
            details
        )

    # --------------------------------------------------------
    # FR-8
    # --------------------------------------------------------

    def build_config_tab(self):

        frame = ttk.LabelFrame(
            self.config_tab,
            text="Admin Configuration"
        )

        frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        self.config_vars = {}

        config_fields = [
            ("camera_id", "Camera ID"),
            ("location_name", "Location"),
            ("video_source", "Video / Camera Source"),
            ("speed_limit_kmh", "Speed Limit (km/h)"),
            (
                "violation_threshold_kmh",
                "Violation Threshold (km/h)"
            ),
            (
                "min_ocr_confidence",
                "Minimum OCR Confidence"
            ),
            (
                "min_blur_threshold",
                "Minimum Blur Threshold"
            ),
            (
                "ocr_scale_width",
                "OCR Scale Width"
            ),
            (
                "ocr_sampling_frames",
                "OCR Sampling Frames"
            ),
            (
                "yolo_imgsz",
                "YOLO Image Size"
            )
        ]

        for row, (
            key,
            label
        ) in enumerate(
            config_fields
        ):

            ttk.Label(
                frame,
                text=label
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=10,
                pady=8
            )

            var = tk.StringVar(
                value=str(
                    self.admin.get(key)
                )
            )

            self.config_vars[
                key
            ] = var

            ttk.Entry(
                frame,
                textvariable=var,
                width=75
            ).grid(
                row=row,
                column=1,
                padx=10,
                pady=8
            )

        ttk.Button(
            frame,
            text="Save Configuration",
            command=self.save_configuration
        ).grid(
            row=len(config_fields),
            column=0,
            columnspan=2,
            pady=20
        )

        ttk.Label(
            frame,
            text=(
                "Changes are saved to admin_config.json. "
                "Restart processing after changing "
                "detection or OCR settings."
            )
        ).grid(
            row=len(config_fields) + 1,
            column=0,
            columnspan=2,
            pady=10
        )

    def save_configuration(self):

        try:

            self.admin.set(
                "camera_id",
                self.config_vars[
                    "camera_id"
                ].get().strip()
            )

            self.admin.set(
                "location_name",
                self.config_vars[
                    "location_name"
                ].get().strip()
            )

            self.admin.set(
                "video_source",
                self.config_vars[
                    "video_source"
                ].get().strip()
            )

            self.admin.set(
                "speed_limit_kmh",
                float(
                    self.config_vars[
                        "speed_limit_kmh"
                    ].get()
                )
            )

            self.admin.set(
                "violation_threshold_kmh",
                float(
                    self.config_vars[
                        "violation_threshold_kmh"
                    ].get()
                )
            )

            self.admin.set(
                "min_ocr_confidence",
                float(
                    self.config_vars[
                        "min_ocr_confidence"
                    ].get()
                )
            )

            self.admin.set(
                "min_blur_threshold",
                float(
                    self.config_vars[
                        "min_blur_threshold"
                    ].get()
                )
            )

            self.admin.set(
                "ocr_scale_width",
                int(
                    self.config_vars[
                        "ocr_scale_width"
                    ].get()
                )
            )

            self.admin.set(
                "ocr_sampling_frames",
                int(
                    self.config_vars[
                        "ocr_sampling_frames"
                    ].get()
                )
            )

            self.admin.set(
                "yolo_imgsz",
                int(
                    self.config_vars[
                        "yolo_imgsz"
                    ].get()
                )
            )

            messagebox.showinfo(
                "Configuration Saved",
                "Admin configuration saved successfully."
            )

        except ValueError:

            messagebox.showerror(
                "Invalid Value",
                "Please enter valid numeric values."
            )


# ============================================================
# BACKGROUND AI PROCESS
# ============================================================

def _background_ai_process(video_path, config_file, output_dir, model_path):
    """Run the complete YOLO + EasyOCR pipeline independently of playback."""
    try:
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        admin = AdminConfiguration(config_file)
        detector = SpeedAndPlateDetector(
            model_path=model_path,
            admin_config=admin,
            output_dir=output_dir
        )

        # This method performs AI only. It never opens an OpenCV window.
        detector._process_ai_video(video_path, display=False)
        print("[SUCCESS] Background YOLO + OCR processing completed.")

    except Exception as exc:
        print(f"[ERROR] Background AI process failed: {exc}")


# ============================================================
# LAUNCH DASHBOARD
# ============================================================

def launch_dashboard(admin):

    root = tk.Tk()

    ViolationDashboard(
        root,
        admin
    )

    root.mainloop()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Speed Detection & Plate Extraction System"
        )
    )

    parser.add_argument(
        "--review",
        action="store_true",
        help=(
            "Open violation review/search "
            "dashboard only"
        )
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help=(
            "Open admin configuration "
            "dashboard only"
        )
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Process video without displaying "
            "OpenCV window"
        )
    )

    args = parser.parse_args()

    admin = AdminConfiguration(
        "admin_config.json"
    )

    # Review/search/configuration only.
    if args.review or args.config:

        launch_dashboard(
            admin
        )

    else:

        video_path = admin.get(
            "video_source"
        )

        if not os.path.exists(
            video_path
        ):

            print(
                f"[ERROR] Video file not found:\n"
                f"{video_path}\n\n"
                f"Open the application with --config "
                f"to change the video source."
            )

        else:

            detector = (
                SpeedAndPlateDetector(
                    model_path="yolov8n.pt",
                    admin_config=admin,
                    output_dir=(
                        "violations_evidence"
                    )
                )
            )

            detector.process_video(
                video_path,
                display=not args.no_display
            )

            # Open review/search dashboard
            # after processing.
            launch_dashboard(
                admin
            )