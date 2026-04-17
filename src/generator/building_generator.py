from pathlib import Path

def generate_simple_building(output_dir: Path, floors: int = 5, color: str = "#aaaaaa"):
    height = floors * 1.5

    obj_content = f"""
o Building
v -1 0 -1
v 1 0 -1
v 1 {height} -1
v -1 {height} -1
v -1 0 1
v 1 0 1
v 1 {height} 1
v -1 {height} 1

f 1 2 3 4
f 5 6 7 8
f 1 5 8 4
f 2 6 7 3
f 4 3 7 8
f 1 2 6 5
"""

    obj_path = output_dir / "building.obj"
    with open(obj_path, "w") as f:
        f.write(obj_content)

    return {
        "obj_path": str(obj_path),
        "mtl_path": None
    }