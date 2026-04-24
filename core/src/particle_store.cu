#include "water/core/particle_store.h"

namespace water {

ParticleStore::ParticleStore(std::size_t capacity, cudaStream_t stream)
    : capacity_(capacity), stream_(stream) {
    position_ = register_attribute<Vec3f>("position", AttribType::F32x3);
    velocity_ = register_attribute<Vec3f>("velocity", AttribType::F32x3);
}

void ParticleStore::resize(std::size_t count) {
    if (count > capacity_) {
        throw std::out_of_range(
            "ParticleStore::resize: count " + std::to_string(count)
            + " exceeds capacity " + std::to_string(capacity_));
    }
    count_ = count;
}

bool ParticleStore::has_attribute(std::string_view name) const {
    return name_to_index_.find(std::string{name}) != name_to_index_.end();
}

} // namespace water
