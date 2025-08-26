import io, re, hashlib, pathlib
from typing import Optional
import pandas as pd
try:
    import requests
except ImportError:
    requests = None

BLOB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$")

def to_raw(url: str) -> str:
    m = BLOB_RE.match(url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url

def read_csv_flexible(src: str, cache_dir: Optional[str] = "data/cache", **kwargs) -> pd.DataFrame:
    kwargs.setdefault("dtype", str); kwargs.setdefault("keep_default_na", False)
    if not src.startswith("http"):
        return pd.read_csv(src, **kwargs)

    raw_url = to_raw(src)
    cache_path = None
    if cache_dir:
        h = hashlib.sha1(raw_url.encode()).hexdigest()[:16]
        cache_path = pathlib.Path(cache_dir) / f"{h}_{pathlib.Path(raw_url).name}"
        if cache_path.exists(): return pd.read_csv(cache_path, **kwargs)
    
    try:
        df = pd.read_csv(raw_url, **kwargs)
    except Exception:
        if requests is None: raise ImportError("'requests' is required for URLs")
        r = requests.get(raw_url, timeout=30); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), **kwargs)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df
