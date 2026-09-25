from pathlib import Path

from workers.pipeline_adapter import PipelineAdapter


VIDEO_PATH = (
    r"C:\Users\Ainee\OneDrive\Desktop"
    r"\Speed_Detection_Project\videos"
    r"\257615344-1af57131-3ada-470a-b798-95fff00254e6.mp4"
)

OUTPUT_DIR = (
    r"C:\Users\Ainee\OneDrive\Desktop"
    r"\Speed_Detection_Project\output"
)


def show_progress(
    percent: float,
    message: str,
) -> None:
    print(
        f"[{percent:6.1f}%] {message}"
    )


video_path = Path(VIDEO_PATH)
output_path = Path(OUTPUT_DIR)

if not video_path.exists():
    raise FileNotFoundError(
        f"Video not found: {video_path}"
    )

output_path.mkdir(
    parents=True,
    exist_ok=True,
)

adapter = PipelineAdapter()

result = adapter.process(
    input_path=video_path,
    output_dir=output_path,
    speed_limit_kmh=50.0,
    on_progress=show_progress,
)

print()
print("=" * 70)
print("PIPELINE ADAPTER RESULT")
print("=" * 70)

print(
    f"Status: {result['status']}"
)

print(
    f"Input video: "
    f"{result['input_path']}"
)

print(
    f"Output directory: "
    f"{result['output_dir']}"
)

print(
    f"Speed limit: "
    f"{result['speed_limit_kmh']} km/h"
)

print(
    f"Violations: "
    f"{len(result['violations'])}"
)

print(
    f"Duration: "
    f"{result['duration_seconds']} seconds"
)

print(
    f"Quality warnings: "
    f"{result['quality_warnings']}"
)

print("=" * 70)

for violation in result["violations"]:
    print()
    print(
        f"Violation: "
        f"{violation['violation_id']}"
    )

    print(
        f"Speed: "
        f"{violation['measured_speed_kmh']} km/h"
    )

    print(
        f"Plate: "
        f"{violation['license_plate']}"
    )

    print(
        f"OCR confidence: "
        f"{violation['ocr_confidence']}"
    )

    print(
        f"Evidence: "
        f"{violation['evidence_image_path']}"
    )