#include "water/solvers/boundary_sampler.h"
#include <cmath>

namespace water::solvers {

std::vector<Vec3f> sample_aabb_boundary(Vec3f lo, Vec3f hi, f32 spacing, int n_layers) {
    std::vector<Vec3f> pts;
    auto emit_face = [&](int axis, f32 face_coord, f32 inward_step) {
        const f32 a_lo[3] = {lo.x, lo.y, lo.z};
        const f32 a_hi[3] = {hi.x, hi.y, hi.z};
        const int u_axis = (axis + 1) % 3;
        const int v_axis = (axis + 2) % 3;
        for (int layer = 0; layer < n_layers; ++layer) {
            const f32 axis_pos = face_coord + layer * inward_step;
            for (f32 u = a_lo[u_axis]; u <= a_hi[u_axis] + 0.5f * spacing; u += spacing)
            for (f32 v = a_lo[v_axis]; v <= a_hi[v_axis] + 0.5f * spacing; v += spacing) {
                f32 p[3];
                p[axis]   = axis_pos;
                p[u_axis] = u;
                p[v_axis] = v;
                pts.push_back({p[0], p[1], p[2]});
            }
        }
    };
    emit_face(0, lo.x,  spacing);   // -X face, layers go +X (inward)
    emit_face(0, hi.x, -spacing);   // +X face, layers go -X
    emit_face(1, lo.y,  spacing);   // -Y face (floor), layers go +Y
    emit_face(1, hi.y, -spacing);   // +Y face (ceiling), layers go -Y
    emit_face(2, lo.z,  spacing);   // -Z face
    emit_face(2, hi.z, -spacing);   // +Z face
    return pts;
}

} // namespace water::solvers
