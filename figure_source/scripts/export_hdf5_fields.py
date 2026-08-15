from pathlib import Path
import sys
import xml.etree.ElementTree as ET

LOCAL_DEPS = Path(r"E:\models\PFM_EBSD\IJSS_reconstruction\.pydeps")
if str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

import h5py
import numpy as np


ROOT = Path(r"E:\models\PFM_EBSD\IJSS_reconstruction")
OUT_DIR = ROOT / "source_data_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL = next(
    Path("D:/models").glob(
        "PFM*/PFM_H_GraphFracture/PFM_H_GraphFracture"
    )
)
RESULTS = MODEL / "results"


def xdmf_attribute_records(path):
    root = ET.parse(path).getroot()
    records = []
    for grid in root.findall(".//Grid"):
        attr = grid.find("./Attribute")
        if attr is None:
            continue
        time = grid.find("./Time")
        item = attr.find("./DataItem")
        if item is None or not (item.text or "").strip():
            continue
        reference = (item.text or "").strip()
        h5_name, dataset = reference.split(":", 1)
        records.append(
            {
                "name": attr.get("Name"),
                "center": attr.get("Center"),
                "time": float(time.get("Value")) if time is not None else 0.0,
                "h5": path.parent / h5_name,
                "dataset": dataset,
            }
        )
    return records


def read_mesh(h5_path):
    with h5py.File(h5_path, "r") as h5:
        geometry = np.asarray(h5["/Mesh/mesh/geometry"], dtype=float)
        topology = np.asarray(h5["/Mesh/mesh/topology"], dtype=np.int64)
    return geometry, topology


def read_record(record):
    with h5py.File(record["h5"], "r") as h5:
        return np.asarray(h5[record["dataset"]], dtype=float).reshape(-1)


def read_named_fields(case_dir, stem):
    records = xdmf_attribute_records(case_dir / f"{stem}.xdmf")
    fields = {record["name"]: read_record(record) for record in records}
    geometry, topology = read_mesh(records[0]["h5"])
    return geometry, topology, fields


def read_latest_damage(case_dir, xdmf_name="damage.xdmf"):
    records = xdmf_attribute_records(case_dir / xdmf_name)
    record = max(records, key=lambda item: item["time"])
    geometry, topology = read_mesh(record["h5"])
    return geometry, topology, record["time"], read_record(record)


def read_graph_resume_damage(case_dir):
    records = []
    for name in ("damage.resume_001.xdmf", "damage.resume_002.xdmf"):
        records.extend(xdmf_attribute_records(case_dir / name))
    records.sort(key=lambda item: item["time"])
    geometry, topology = read_mesh(records[0]["h5"])
    stack = np.vstack([read_record(record) for record in records])
    times = np.asarray([record["time"] for record in records], dtype=float)
    return geometry, topology, times, stack


def main():
    graph_h = RESULTS / "ijss_ban_factorial_graph_H"
    hom_h = RESULTS / "ijss_ban_factorial_homogeneous_H"
    hom_noh = RESULTS / "ijss_ban_factorial_homogeneous_noH"
    pilot = RESULTS / "ebsd_ban_confidence1_onset_pilot"

    geometry, topology, material = read_named_fields(graph_h, "material")
    h_geometry, h_topology, hydrogen = read_named_fields(graph_h, "hydrogen")
    g_geometry, g_topology, graph_times, graph_damage = read_graph_resume_damage(graph_h)
    noh_geometry, noh_topology, noh_time, noh_damage = read_latest_damage(hom_noh)
    hh_geometry, hh_topology, hh_time, hh_damage = read_latest_damage(hom_h)
    p_geometry, p_topology, p_time, p_damage = read_latest_damage(pilot)

    np.savez_compressed(
        OUT_DIR / "spatial_fields_export.npz",
        geometry=geometry,
        topology=topology,
        base_fracture_toughness=material["base_fracture_toughness"],
        effective_fracture_toughness=material["effective_fracture_toughness"],
        hydrogen_diffusivity=material["hydrogen_diffusivity"],
        trap_density=material["trap_density"],
        hydrogen_geometry=h_geometry,
        hydrogen_topology=h_topology,
        lattice_hydrogen=hydrogen["lattice_hydrogen"],
        hydrogen_coverage=hydrogen["hydrogen_coverage"],
        trapped_hydrogen=hydrogen["trapped_hydrogen"],
        graph_damage_geometry=g_geometry,
        graph_damage_topology=g_topology,
        graph_damage_times=graph_times,
        graph_damage=graph_damage,
        homogeneous_noh_geometry=noh_geometry,
        homogeneous_noh_topology=noh_topology,
        homogeneous_noh_time=np.asarray([noh_time]),
        homogeneous_noh_damage=noh_damage,
        homogeneous_h_geometry=hh_geometry,
        homogeneous_h_topology=hh_topology,
        homogeneous_h_time=np.asarray([hh_time]),
        homogeneous_h_damage=hh_damage,
        pilot_geometry=p_geometry,
        pilot_topology=p_topology,
        pilot_time=np.asarray([p_time]),
        pilot_damage=p_damage,
    )
    print(OUT_DIR / "spatial_fields_export.npz")


if __name__ == "__main__":
    main()

