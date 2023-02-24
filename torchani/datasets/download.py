r"""Utilities to download files"""
import urllib
import tarfile
import hashlib
from pathlib import Path

from torchani.utils import tqdm


_USER_AGENT = "torchani"
_MAX_HOPS = 3
_CHUNK_SIZE = 1024 * 32


def check_integrity(file_path: Path, md5: str) -> bool:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == md5


def download_and_extract_archive(
        base_url: str,
        file_name: str,
        dest_root: Path,
) -> None:
    file_path = dest_root / file_name
    url = f"{base_url}{file_name}"

    # download
    _download_file_from_url(url, file_path)

    if not str(file_path).endswith(".tar.gz"):
        raise ValueError("Incorrect file type for {file_path}, expected .tar.gz")

    # extract
    with tarfile.open(file_path, "r:gzip") as f:
        f.extractall(file_path.with_suffix(""))
    file_path.unlink()


def _download_file_from_url(url: str, file_path: Path) -> None:
    print(f"Downloading {url} to {str(file_path)}")
    url = _get_redirect_url(url)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})) as response:
        content = iter(lambda: response.read(_CHUNK_SIZE), b"")
        with open(file_path, "wb") as f, tqdm(total=response.length) as pbar:
            for chunk in content:
                # filter out keep-alive new chunks
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))


def _get_redirect_url(url: str) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": _USER_AGENT}
    for _ in range(_MAX_HOPS + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url
            url = response.url
    raise RecursionError(f"Request to {initial_url} exceeded {_MAX_HOPS} redirects. The last redirect points to {url}.")
