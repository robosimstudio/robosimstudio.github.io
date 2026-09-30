"""A small GLB writer: named pose nodes, one-primitive meshes, finish-exact PBR materials.

Material semantics are ``robosimstudio_studio/finish_gltf.glb_for``'s, restated for a
multi-node file (that function encodes one mesh per GLB):

* ``baseColorFactor`` is **linear** (the palette is display sRGB, decoded here), and
  white when the finish has a base-colour texture -- glTF multiplies, the tracer's
  ``UsdPreviewSurface`` replaces;
* a roughness map is packed red -> green (glTF's roughness channel) with
  ``roughnessFactor = 1``, so glTF's multiply lands on the map's own value;
* ``uv_scale``/``uv_offset`` are baked into the UVs (no KHR_texture_transform);
* ``alphaMode = BLEND`` below opacity 1, ``doubleSided = true``.

``tests/test_gltf.py`` checks the material JSON against ``glb_for``'s.
"""

from __future__ import annotations

import io
import json
import struct
from typing import Any

import numpy as np

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * (-len(b) % 4)


class Glb:
    def __init__(self) -> None:
        self.doc: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": "rsw"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [{"byteLength": 0}],
        }
        self.bin = bytearray()
        self._materials: dict[tuple, int] = {}
        self._textures: dict[tuple, int] = {}

    # --------------------------------------------------------------- buffers
    def _view(self, data: bytes, target: int | None = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.bin.extend(data)
        self.doc["bufferViews"].append(view)
        return len(self.doc["bufferViews"]) - 1

    def _accessor(self, arr: np.ndarray, kind: str, *, index: bool = False, minmax: bool = False) -> int:
        if index:
            data = np.ascontiguousarray(arr, dtype=np.uint32).reshape(-1)
            ctype = 5125
        else:
            data = np.ascontiguousarray(arr, dtype=np.float32)
            ctype = 5126
        view = self._view(data.tobytes(), _ELEMENT_ARRAY_BUFFER if index else _ARRAY_BUFFER)
        acc: dict[str, Any] = {"bufferView": view, "componentType": ctype, "count": int(len(data)), "type": kind}
        if minmax:
            acc["min"] = [float(v) for v in data.min(axis=0)]
            acc["max"] = [float(v) for v in data.max(axis=0)]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    # ------------------------------------------------------------- textures
    def _texture(self, key: tuple, image, mime: str) -> int:
        if key in self._textures:
            return self._textures[key]
        buf = io.BytesIO()
        img = image
        if max(img.size) > 1024:
            s = 1024 / max(img.size)
            img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))))
        if mime == "image/jpeg":
            img.convert("RGB").save(buf, format="JPEG", quality=90)
        else:
            img.save(buf, format="PNG", optimize=True)
        view = self._view(buf.getvalue())
        self.doc.setdefault("images", []).append({"bufferView": view, "mimeType": mime})
        self.doc.setdefault("samplers", [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])
        self.doc.setdefault("textures", []).append({"sampler": 0, "source": len(self.doc["images"]) - 1})
        idx = len(self.doc["textures"]) - 1
        self._textures[key] = idx
        return idx

    # ------------------------------------------------------------- materials
    def material(self, color, finish, *, textured: bool, name: str = "") -> int:
        """A material for ``(display colour, finish)``; ``textured`` if the mesh has UVs."""
        from robosimstudio.core.render.palette import srgb_to_linear
        from robosimstudio_studio import finish_gltf as fg

        image = fg._texture_image(finish.texture) if textured else None
        rough_map = fg._roughness_image(finish.roughness_texture) if textured else None
        key = (fg.material_key(color, finish), image is not None, rough_map is not None)
        if key in self._materials:
            return self._materials[key]
        linear = (1.0, 1.0, 1.0) if image is not None else tuple(float(v) for v in srgb_to_linear(tuple(color)[:3]))
        roughness = 1.0 if rough_map is not None else float(finish.roughness)
        opacity = float(finish.opacity)
        pbr: dict[str, Any] = {
            "baseColorFactor": [*linear, opacity],
            "metallicFactor": float(finish.metallic),
            "roughnessFactor": roughness,
        }
        if image is not None:
            pbr["baseColorTexture"] = {"index": self._texture(("c", finish.texture), image, "image/jpeg")}
        if rough_map is not None:
            pbr["metallicRoughnessTexture"] = {"index": self._texture(("r", finish.roughness_texture), rough_map, "image/png")}
        mat = {
            "name": name or f"m{len(self.doc['materials'])}",
            "pbrMetallicRoughness": pbr,
            "alphaMode": "BLEND" if opacity < 1.0 else "OPAQUE",
            "doubleSided": True,
        }
        self.doc["materials"].append(mat)
        self._materials[key] = len(self.doc["materials"]) - 1
        return self._materials[key]

    # ----------------------------------------------------------------- mesh
    def mesh(self, name: str, points, faces, *, material: int, normals=None, uvs=None) -> int:
        attrs = {"POSITION": self._accessor(np.asarray(points), "VEC3", minmax=True)}
        if normals is not None:
            attrs["NORMAL"] = self._accessor(np.asarray(normals), "VEC3")
        if uvs is not None:
            attrs["TEXCOORD_0"] = self._accessor(np.asarray(uvs), "VEC2")
        prim = {"attributes": attrs, "indices": self._accessor(np.asarray(faces), "SCALAR", index=True), "material": material}
        self.doc["meshes"].append({"name": name, "primitives": [prim]})
        return len(self.doc["meshes"]) - 1

    def node(self, name: str, *, mesh: int | None = None, children=(), pose7=None, extras=None, root: bool = False) -> int:
        n: dict[str, Any] = {"name": name}
        if mesh is not None:
            n["mesh"] = mesh
        if children:
            n["children"] = list(children)
        if pose7 is not None:
            p = [float(v) for v in pose7]
            n["translation"] = p[:3]
            n["rotation"] = p[3:7]  # glTF is xyzw too
        if extras:
            n["extras"] = extras
        self.doc["nodes"].append(n)
        idx = len(self.doc["nodes"]) - 1
        if root:
            self.doc["scenes"][0]["nodes"].append(idx)
        return idx

    # ----------------------------------------------------------------- write
    def to_bytes(self) -> bytes:
        doc = dict(self.doc)
        doc["buffers"] = [{"byteLength": len(self.bin)}]
        for k in ("images", "samplers", "textures"):
            if k in doc and not doc[k]:
                del doc[k]
        js = _pad4(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
        bn = _pad4(bytes(self.bin))
        body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(bn), 0x004E4942) + bn
        return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def read_json(glb: bytes) -> dict:
    """The JSON chunk of a GLB (tests and checks)."""
    size, kind = struct.unpack_from("<II", glb, 12)
    assert kind == 0x4E4F534A
    return json.loads(glb[20 : 20 + size].decode("utf-8"))
