import hashlib
import json
import os
from typing import Optional
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

T = TypeVar("T")


def make_cache(location: Optional[Path] = None):
    if not os.path.exists(location):
        os.makedirs(location, exist_ok=True)
    def cache(key: Any, device, fallback: Callable[[], T]) -> T:
        # If there's no cache location, the cache is disabled.
        if location is None:
            return fallback()

        key_str = hashlib.sha256(json.dumps(key).encode("utf-8")).digest().hex()

        path = location + "/" + f"{key_str}.torch"
        try:
            # Attempt to load the cached item.
            key_loaded, value = torch.load(path, map_location=device)

            # If there was a hash collision and the keys don't actually match, throw an
            # error so that the fallback can be used.
            if key != key_loaded:
                raise ValueError("Keys did not match!")

            return value
        except (FileNotFoundError, ValueError):
            # Use the fallback to compute the value.
            value = fallback()

            # Cache the value.
            # path.parent.mkdir(exist_ok=True, parents=True)
            torch.save((key, value), path)

            return value

    return cache
