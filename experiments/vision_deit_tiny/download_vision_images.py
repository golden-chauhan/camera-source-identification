
import csv
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit, unquote

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST = PROJECT_DIR / "data" / "raw" / "VISION" / "VISION_files.txt"
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "VISION"

# Download settings
MAX_WORKERS = 8
MAX_RETRIES = 4
TIMEOUT = 60
PROGRESS_INTERVAL = 100


def get_image_urls():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    with MANIFEST.open("r", encoding="utf-8") as file:
        urls = [
            line.strip()
            for line in file
            if line.strip().startswith("http")
            and urlsplit(line.strip()).path.lower().endswith(".jpg")
        ]

    # Remove duplicate URLs while preserving order
    urls = list(dict.fromkeys(urls))
    return urls


def download_one(url):
    parsed = urlsplit(url)

    if "/dataset/" not in parsed.path:
        return "failed", url, "Unexpected URL structure"

    relative_path = unquote(
        parsed.path.split("/dataset/", 1)[1]
    )

    destination = OUTPUT_DIR / Path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return "skipped", url, ""

    temporary_file = destination.with_suffix(
        destination.suffix + ".part"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "VISION-Dataset-Downloader/1.0"}
            )

            with urllib.request.urlopen(
                request, timeout=TIMEOUT
            ) as response:

                first_chunk = response.read(1024)

                # Basic JPEG file validation
                if not first_chunk.startswith(b"\xff\xd8\xff"):
                    raise ValueError("Downloaded file is not a valid JPEG")

                with temporary_file.open("wb") as output:
                    output.write(first_chunk)

                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)

            temporary_file.replace(destination)
            return "downloaded", url, ""

        except Exception as error:
            if temporary_file.exists():
                temporary_file.unlink()

            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
            else:
                return "failed", url, str(error)

    return "failed", url, "Unknown download error"


def main():
    print("=" * 60)
    print("VISION FULL IMAGE DATASET DOWNLOADER")
    print("=" * 60)

    urls = get_image_urls()
    total = len(urls)

    print(f"Manifest: {MANIFEST}")
    print(f"Image URLs: {total}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Workers: {MAX_WORKERS}")
    print("\nStarting download...\n")

    downloaded = 0
    skipped = 0
    failed = 0
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_one, url)
            for url in urls
        ]

        for completed, future in enumerate(
            as_completed(futures), start=1
        ):
            status, url, error = future.result()

            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append((url, error))

            if completed % PROGRESS_INTERVAL == 0 or completed == total:
                print(
                    f"Progress: {completed}/{total} | "
                    f"Downloaded: {downloaded} | "
                    f"Skipped: {skipped} | "
                    f"Failed: {failed}"
                )

    if failures:
        failure_file = OUTPUT_DIR / "download_failures.csv"

        with failure_file.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["URL", "Error"])
            writer.writerows(failures)

        print(f"\nFailure details saved to: {failure_file}")

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total URLs: {total}")
    print(f"Downloaded: {downloaded}")
    print(f"Already present: {skipped}")
    print(f"Failed: {failed}")
    print(f"Dataset folder: {OUTPUT_DIR}")

    if failed == 0:
        print("\nAll requested JPG downloads completed.")
    else:
        print("\nSome files failed. You can rerun this script safely.")


if __name__ == "__main__":
    main()
