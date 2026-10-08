"""Stamp the published landing page's own files with their content: ``site.css`` becomes
``site.css?v=<sha1[:10]>``, and the same for the script, the logos, the clips and their posters.

The gateway once sent these with no Cache-Control, so a browser kept an old ``site.css`` under a new
``index.html`` (10-08: the section menu came out unstyled, the old white box behind the logos). A
stamped URL changes whenever the file does, so no cache can pair a page with yesterday's files.
Run by ``up.sh`` on the published copy (``$RSW_LIVE/www/index.html``), never on the repo's.

    python3 live/stamp.py ~/rss_ws/rsw_live/www
"""

import hashlib
import re
import sys
from pathlib import Path

ATTR = re.compile(r'\b(src|href|poster|srcset)="([^"#:?]+)(\?v=[0-9a-f]+)?"')


def stamp(www: Path) -> int:
    page = www / "index.html"
    html = page.read_text()
    done = 0

    def sub(m: re.Match) -> str:
        nonlocal done
        attr, path = m.group(1), m.group(2)
        f = www / path
        if not f.is_file():
            return m.group(0)
        done += 1
        return f'{attr}="{path}?v={hashlib.sha1(f.read_bytes()).hexdigest()[:10]}"'

    page.write_text(ATTR.sub(sub, html))
    return done


if __name__ == "__main__":
    print(f"stamp: {stamp(Path(sys.argv[1]))} file links")
