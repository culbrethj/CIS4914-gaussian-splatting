import math
import os
import struct
import sys

SH_C0 = 0.28209479177387814

def read_ply_header(fileobj, max_lines=50000):
    header_lines = []
    for _ in range(max_lines):
        line_bytes = fileobj.readline()
        if not line_bytes:
            raise ValueError("Unexpected EOF while searching for end_header.")

        line_str = line_bytes.decode("ascii", errors="replace")
        header_lines.append(line_str)

        if "end_header" in line_str:
            break
    else:
        raise ValueError(f"No 'end_header' found in the first {max_lines} header lines")

    return header_lines

def parse_ply_header_lines(header_lines):
    vertex_count = 0
    prop_list = []
    is_binary_le = False

    in_vertex_element = False

    for line in header_lines:
        line = line.strip()

        if line.startswith("format "):
            if "binary_little_endian" in line:
                is_binary_le = True
            elif "ascii" in line:
                raise ValueError("ASCII PLY is not supported by this example.")
            else:
                raise ValueError("Only binary_little_endian PLY is handled here.")

        elif line.startswith("element vertex"):
            parts = line.split()
            vertex_count = int(parts[2])
            in_vertex_element = True

        elif line.startswith("element ") and not line.startswith("element vertex"):
            in_vertex_element = False

        elif in_vertex_element and line.startswith("property "):
            parts = line.split()
            prop_type = parts[1]
            prop_name = parts[2]
            prop_list.append((prop_name, prop_type))

    return vertex_count, prop_list, is_binary_le

def ply_to_splat(ply_path):
    if not ply_path.endswith(".ply"):
        raise ValueError("Input file must be a .ply file")

    with open(ply_path, "rb") as f:
        header_lines = read_ply_header(f)

        vertex_count, properties, is_bin_le = parse_ply_header_lines(header_lines)
        if not is_bin_le:
            raise ValueError("PLY is not binary_little_endian; cannot continue.")

        type_map = {
            "float":  ("f", 4),
            "float32":("f", 4),
            "double": ("d", 8),
            "float64":("d", 8),
            "int":    ("i", 4),
            "int32":  ("i", 4),
            "uint":   ("I", 4),
            "uint32": ("I", 4),
            "short":  ("h", 2),
            "ushort": ("H", 2),
            "int16":  ("h", 2),
            "uint16": ("H", 2),
            "char":   ("b", 1),
            "uchar":  ("B", 1),
            "int8":   ("b", 1),
            "uint8":  ("B", 1),
        }
        prop_structs = []
        row_size = 0
        for (name, ptype) in properties:
            if ptype not in type_map:
                raise ValueError(f"Unsupported property type: {ptype}")
            fmt, size = type_map[ptype]
            prop_structs.append((name, fmt, size))
            row_size += size

        splat_data = bytearray(32 * vertex_count)

        def clamp_byte(x):
            return max(0, min(255, int(round(x))))

        def write_float(offset, value):
            struct.pack_into("<f", splat_data, offset, value)

        total_vertex_bytes = row_size * vertex_count
        vertex_block = f.read(total_vertex_bytes)
        if len(vertex_block) < total_vertex_bytes:
            raise ValueError("Not enough data for all vertices in the file.")

        idx = 0
        for i in range(vertex_count):
            out_offset = i * 32
            px, py, pz = 0.0, 0.0, 0.0
            sx, sy, sz = 1.0, 1.0, 1.0
            cr, cg, cb, ca = 255, 255, 255, 255
            rw, rx, ry, rz = 1.0, 0.0, 0.0, 0.0

            prop_offset = 0
            for (prop_name, prop_fmt, prop_size) in prop_structs:
                raw_value = struct.unpack_from("<" + prop_fmt, vertex_block, idx + prop_offset)[0]
                prop_offset += prop_size

                if prop_name == "x":
                    px = float(raw_value)
                elif prop_name == "y":
                    py = float(raw_value)
                elif prop_name == "z":
                    pz = float(raw_value)

                elif prop_name in ["red", "r"]:
                    cr = clamp_byte(raw_value)
                elif prop_name in ["green", "g"]:
                    cg = clamp_byte(raw_value)
                elif prop_name in ["blue", "b"]:
                    cb = clamp_byte(raw_value)
                elif prop_name in ["alpha", "a"]:
                    ca = clamp_byte(raw_value)
                
                elif prop_name in ["f_dc_0", "features_0"]:
                    cr = clamp_byte((0.5 + SH_C0 * raw_value) * 255)
                elif prop_name in ["f_dc_1", "features_1"]:
                    cg = clamp_byte((0.5 + SH_C0 * raw_value) * 255)
                elif prop_name in ["f_dc_2", "features_2"]:
                    cb = clamp_byte((0.5 + SH_C0 * raw_value) * 255)
                elif prop_name == "f_dc_3":
                    ca = clamp_byte((0.5 + SH_C0 * raw_value) * 255)

                elif prop_name in ["scale_0","scaling_0"]:
                    sx = math.exp(raw_value)
                elif prop_name in ["scale_1","scaling_1"]:
                    sy = math.exp(raw_value)
                elif prop_name in ["scale_2","scaling_2"]:
                    sz = math.exp(raw_value)

                elif prop_name in ["opacity","opacity_0"]:
                    val = 1.0 / (1.0 + math.exp(-float(raw_value)))
                    ca = clamp_byte(val*255.0)

                elif prop_name in ["rot_0","rotation_0"]:
                    rw = float(raw_value)
                elif prop_name in ["rot_1","rotation_1"]:
                    rx = float(raw_value)
                elif prop_name in ["rot_2","rotation_2"]:
                    ry = float(raw_value)
                elif prop_name in ["rot_3","rotation_3"]:
                    rz = float(raw_value)

            idx += row_size

            length = math.sqrt(rw*rw + rx*rx + ry*ry + rz*rz)
            if length > 1e-9:
                rw /= length
                rx /= length
                ry /= length
                rz /= length

            rot0 = clamp_byte(rw * 128 + 128)
            rot1 = clamp_byte(rx * 128 + 128)
            rot2 = clamp_byte(ry * 128 + 128)
            rot3 = clamp_byte(rz * 128 + 128)

            write_float(out_offset + 0, px)
            write_float(out_offset + 4, py)
            write_float(out_offset + 8, pz)

            write_float(out_offset + 12, sx)
            write_float(out_offset + 16, sy)
            write_float(out_offset + 20, sz)

            splat_data[out_offset + 24] = cr
            splat_data[out_offset + 25] = cg
            splat_data[out_offset + 26] = cb
            splat_data[out_offset + 27] = ca

            splat_data[out_offset + 28] = rot0
            splat_data[out_offset + 29] = rot1
            splat_data[out_offset + 30] = rot2
            splat_data[out_offset + 31] = rot3

    base_dir = os.path.dirname(ply_path)
    filename = os.path.basename(ply_path)
    splat_path = os.path.join(base_dir, filename.replace(".ply", ".splat"))
    with open(splat_path, "wb") as out:
        out.write(splat_data)

    return splat_path

if __name__ == "__main__":
    ply_to_splat(sys.argv[1])
