import json
from typing import Any

from pydantic import BaseModel, field_validator


def _parse_json_list(v: Any) -> list:
    if isinstance(v, str):
        return json.loads(v)
    return list(v) if v else []


class Theme(BaseModel):
    name: str
    state: str
    anchors: list[str]  # node ids

    @field_validator("anchors", mode="before")
    @classmethod
    def parse_json_list(cls, v: Any) -> list:
        return _parse_json_list(v)

    def render_block(self) -> str:
        anchors_str = ", ".join(self.anchors) if self.anchors else "none"
        return f"{self.name}\n{self.state}\n→ {anchors_str}"
