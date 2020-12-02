from typing import Any

import jsonpickle


def dump_object_as_human_readable_json(obj: Any, file_name: str):
    with open(file_name, "w") as out:
        out.write(jsonpickle.encode(obj, indent=4, separators=(",", ": ")))
