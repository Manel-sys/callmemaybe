import os

__version__ = "1.0.0"
__author__ = "manferre"

_user: str | None = os.environ.get("USER")

if _user:
    _sgoinfre: str = f"/sgoinfre/{_user}"

    if os.path.isdir(_sgoinfre):
        os.environ.setdefault("HF_HOME", f"{_sgoinfre}/call_me_maybe/hf_cache")
