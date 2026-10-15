"""Generate a self-contained audit viewer with escaped, inert embedded data."""

import json
from pathlib import Path


def write_report(report, output):
    template = Path(__file__).with_name("report.html").read_text()
    # Escape HTML delimiters before embedding user-controlled corpus/query metadata.
    data = (
        json.dumps(report, ensure_ascii=True, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    Path(output).write_text(template.replace("__CAIRN_DATA__", data), encoding="utf-8")
