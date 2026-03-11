#!/usr/bin/env python3
from scripts import ai_ranker as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("_")
    }
)

if __name__ == "__main__":
    _impl.main()
