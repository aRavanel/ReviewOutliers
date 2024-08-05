import os
import requests
from pathlib import Path
from src.utils.io.io import decompress_to_json

# ==========================================================================
# Utils functions
# ==========================================================================


def _get_data_urls(data_name: str) -> dict:
    """
    create urls from category name
    """
    base_url_reviews = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
    base_url_metadata = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/"

    link_review = f"{base_url_reviews}{data_name}.jsonl.gz"
    link_metadata = f"{base_url_metadata}meta_{data_name}.jsonl.gz"

    return {"review_url": link_review, "metadata_url": link_metadata}


def _download_and_convert_file(url, dest_folder, json_filename) -> None:
    """
    download a file based on its url
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    gz_filename = os.path.join(dest_folder, json_filename + ".gz")
    json_path = os.path.join(dest_folder, json_filename)

    # Download the file
    response = requests.get(url, stream=True)
    with open(gz_filename, "wb") as gz_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                gz_file.write(chunk)

    # decompress it and clean
    decompress_to_json(gz_filename, json_path)
    os.remove(gz_filename)  # Remove the .gz file after decompression
    print(f"Downloaded and converted {json_filename} to {dest_folder}")


# ==========================================================================
# Exported functions
# ==========================================================================


def download_data(data_name: str) -> None:
    """
    download reviews and metadata
    """
    urls = _get_data_urls(data_name)
    if not urls:
        print(f"No URLs found for {data_name}")
        return

    # Adjust the path to go up one level from the current working directory
    dest_folder = Path("../data/raw")

    print("-- download reviews --")
    _download_and_convert_file(urls["review_url"], dest_folder, f"{data_name}_review.json")

    print("-- download metadata --")
    _download_and_convert_file(urls["metadata_url"], dest_folder, f"{data_name}_metadata.json")
