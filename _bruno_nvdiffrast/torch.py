import torch
import torch.nn.functional as F
import math


class RasterizeCudaContext:
    def __init__(self, device='cuda'):
        self.device = device


class DepthPeeler:
    def __init__(self, ctx, vertices, faces, resolution):
        self.ctx = ctx
        self.device = vertices.device
        B = vertices.shape[0]
        H, W = (resolution, resolution) if isinstance(resolution, int) else (resolution[0], resolution[1])
        self.vertices = vertices
        self.faces = faces
        self.resolution = (H, W)
        self.depth_peeled = torch.full((B, H, W), torch.finfo(torch.float32).max, device=self.device)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def rasterize_next_layer(self):
        rast, rast_db = rasterize(self.ctx, self.vertices, self.faces, self.resolution)
        B, H, W, _ = rast.shape
        new_depth = rast[..., 2]
        prev_depth = self.depth_peeled
        mask = (rast[..., 3] > 0) & (new_depth > prev_depth + 1e-6)
        rast_out = rast.clone()
        rast_out[..., 3] = torch.where(mask, rast[..., 3], torch.zeros_like(rast[..., 3]))
        self.depth_peeled = torch.where(mask, new_depth, prev_depth)
        return rast_out, rast_db


def _clip_to_screen(vertices, resolution):
    B, V, D = vertices.shape
    H, W = resolution
    xy = vertices[..., :2] / vertices[..., 3:4].clamp(min=1e-8)
    z = vertices[..., 2:3] / vertices[..., 3:4].clamp(min=1e-8)
    px = (xy[..., 0] + 1.0) * 0.5 * (W - 1)
    py = (1.0 - xy[..., 1]) * 0.5 * (H - 1)
    return px, py, z.squeeze(-1)


def rasterize(ctx, vertices, faces, resolution):
    B = vertices.shape[0]
    if isinstance(resolution, int):
        H, W = resolution, resolution
    else:
        H, W = resolution[0], resolution[1]
    device = vertices.device

    if faces.dim() == 2:
        faces = faces.unsqueeze(0).expand(B, -1, -1)

    rast_out = torch.zeros(B, H, W, 4, device=device)
    rast_out[:, :, :, 2] = torch.finfo(rast_out.dtype).max
    rast_db_out = torch.zeros(B, H, W, 4, device=device)

    for b in range(B):
        verts = vertices[b]
        face = faces[b]
        F_tri = face.shape[0]

        px, py, z = _clip_to_screen(verts.unsqueeze(0), (H, W))
        px = px[0]; py = py[0]; z = z[0]

        f0 = face[:, 0].long()
        f1 = face[:, 1].long()
        f2 = face[:, 2].long()
        v0x, v0y, v0z = px[f0], py[f0], z[f0]
        v1x, v1y, v1z = px[f1], py[f1], z[f1]
        v2x, v2y, v2z = px[f2], py[f2], z[f2]

        bb_xmin = torch.floor(torch.min(torch.min(v0x, v1x), v2x)).long().clamp(0, W - 1)
        bb_ymin = torch.floor(torch.min(torch.min(v0y, v1y), v2y)).long().clamp(0, H - 1)
        bb_xmax = torch.ceil(torch.max(torch.max(v0x, v1x), v2x)).long().clamp(0, W - 1)
        bb_ymax = torch.ceil(torch.max(torch.max(v0y, v1y), v2y)).long().clamp(0, H - 1)

        denom_all = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y)
        degenerate = denom_all.abs() < 1e-8
        priority = denom_all.abs()

        BLOCK = 32
        for sy in range(0, H, BLOCK):
            for sx in range(0, W, BLOCK):
                ey = min(sy + BLOCK, H)
                ex = min(sx + BLOCK, W)

                overlap = (bb_xmax >= sx) & (bb_xmin < ex) & \
                          (bb_ymax >= sy) & (bb_ymin < ey) & ~degenerate
                overlap_idx = torch.where(overlap)[0]
                K = overlap_idx.shape[0]
                if K == 0:
                    continue

                bh = ey - sy
                bw = ex - sx
                pgy, pgx = torch.meshgrid(
                    torch.arange(sy, ey, device=device, dtype=torch.float32),
                    torch.arange(sx, ex, device=device, dtype=torch.float32),
                    indexing='ij',
                )
                flat_px = pgx.reshape(-1)
                flat_py = pgy.reshape(-1)
                N = flat_px.shape[0]

                for k in range(K):
                    idx = overlap_idx[k]
                    _v0x, _v0y, _v0z = v0x[idx], v0y[idx], v0z[idx]
                    _v1x, _v1y, _v1z = v1x[idx], v1y[idx], v1z[idx]
                    _v2x, _v2y, _v2z = v2x[idx], v2y[idx], v2z[idx]

                    denom = (_v1y - _v2y) * (_v0x - _v2x) + (_v2x - _v1x) * (_v0y - _v2y)
                    if abs(denom) < 1e-8:
                        continue
                    inv_denom = 1.0 / denom

                    l0 = ((_v1y - _v2y) * (flat_px - _v2x) + (_v2x - _v1x) * (flat_py - _v2y)) * inv_denom
                    l1 = ((_v2y - _v0y) * (flat_px - _v2x) + (_v0x - _v2x) * (flat_py - _v2y)) * inv_denom
                    l2 = 1.0 - l0 - l1

                    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0) & (~torch.isnan(l0)) & (~torch.isnan(l1)) & (~torch.isnan(l2))
                    if not inside.any():
                        del l0, l1, l2, inside
                        continue

                    depth = l0 * _v0z + l1 * _v1z + l2 * _v2z
                    px_i = flat_px[inside].long()
                    py_i = flat_py[inside].long()
                    d_i = depth[inside]
                    l0_i = l0[inside]
                    l1_i = l1[inside]

                    cur_depth = rast_out[b, py_i, px_i, 2]
                    closer = d_i < cur_depth
                    if closer.any():
                        c_px = px_i[closer]
                        c_py = py_i[closer]
                        rast_out[b, c_py, c_px, 0] = l0_i[closer]
                        rast_out[b, c_py, c_px, 1] = l1_i[closer]
                        rast_out[b, c_py, c_px, 2] = d_i[closer]
                        rast_out[b, c_py, c_px, 3] = float(idx + 1)

                    del l0, l1, l2, inside, depth, px_i, py_i, d_i, l0_i, l1_i, closer

    return rast_out, rast_db_out


def interpolate(attr, rast, faces, rast_db=None, diff_attrs=None):
    if rast is None:
        return (None, None) if rast_db is not None else None

    B = rast.shape[0]
    C = attr.shape[-1]
    device = rast.device

    if faces.dim() == 2:
        faces_b = faces.unsqueeze(0).expand(B, -1, -1)
    else:
        faces_b = faces

    bary0 = rast[..., 0]
    bary1 = rast[..., 1]
    bary2 = 1.0 - bary0 - bary1
    tri_id = rast[..., 3].long() - 1

    valid = tri_id >= 0
    if not valid.any():
        output = torch.zeros(B, *rast.shape[1:3], C, device=device)
        return (output, None) if rast_db is not None else output

    tri_id_clamped = tri_id.clamp(min=0)
    v0_idx = torch.gather(faces_b[:, :, 0], 1, tri_id_clamped.reshape(B, -1)).reshape(B, *rast.shape[1:3])
    v1_idx = torch.gather(faces_b[:, :, 1], 1, tri_id_clamped.reshape(B, -1)).reshape(B, *rast.shape[1:3])
    v2_idx = torch.gather(faces_b[:, :, 2], 1, tri_id_clamped.reshape(B, -1)).reshape(B, *rast.shape[1:3])

    if attr.dim() == 3:
        attr_a0 = torch.gather(attr, 1, v0_idx.unsqueeze(-1).expand(-1, -1, -1, C).reshape(B, -1, C))
        attr_a1 = torch.gather(attr, 1, v1_idx.unsqueeze(-1).expand(-1, -1, -1, C).reshape(B, -1, C))
        attr_a2 = torch.gather(attr, 1, v2_idx.unsqueeze(-1).expand(-1, -1, -1, C).reshape(B, -1, C))
        attr_a0 = attr_a0.reshape(B, *rast.shape[1:3], C)
        attr_a1 = attr_a1.reshape(B, *rast.shape[1:3], C)
        attr_a2 = attr_a2.reshape(B, *rast.shape[1:3], C)
    else:
        return (rast[..., :3], None) if rast_db is not None else rast[..., :3]

    result = bary0.unsqueeze(-1) * attr_a0 + bary1.unsqueeze(-1) * attr_a1 + bary2.unsqueeze(-1) * attr_a2
    result[~valid.unsqueeze(-1).expand_as(result)] = 0

    if rast_db is not None:
        return (result, None)
    return result


def antialias(img, rast, vertices, faces):
    return F.avg_pool2d(img.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=1).permute(0, 2, 3, 1)


def texture(tex, texc, texd=None, filter_mode='linear', boundary_mode='clamp'):
    B = tex.shape[0]
    C = tex.shape[-1]

    if texc.dim() == 3:
        N = texc.shape[1]
    else:
        N = texc.shape[0]
        texc = texc.unsqueeze(0)
        B = 1

    grid = texc.view(B, 1, N, 2)

    padding_mode = 'border' if boundary_mode in ('clamp', 'CLAMP_TO_EDGE') else 'reflection'
    if boundary_mode == 'cube':
        grid = grid / grid.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        u = torch.atan2(grid[..., 0], -grid[..., 2]) / (2 * math.pi) + 0.5
        v = torch.acos(grid[..., 1].clamp(-1, 1)) / math.pi
        grid = torch.stack([u, v], dim=-1)
        padding_mode = 'reflection'

    mode = 'bilinear' if 'linear' in filter_mode else 'nearest'
    sampled = F.grid_sample(
        tex.permute(0, 3, 1, 2),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    sampled = sampled.permute(0, 2, 3, 1).reshape(B, N, C)

    if texd is not None:
        return (sampled, None)
    return sampled
