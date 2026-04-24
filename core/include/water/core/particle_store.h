#pragma once

#include "water/core/device_buffer.h"
#include "water/core/types.h"
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace water {

enum class AttribType { F32, F32x3, U32 };

inline std::size_t attrib_element_size(AttribType t) {
    switch (t) {
        case AttribType::F32:   return sizeof(float);
        case AttribType::F32x3: return sizeof(float) * 3;
        case AttribType::U32:   return sizeof(unsigned);
    }
    return 0;
}

// Type-erased attribute storage. Internal — use AttribHandle<T> for typed access.
struct AttribStorage {
    std::string  name;
    AttribType   type;
    std::size_t  element_size;     // bytes per element (e.g. 12 for F32x3)
    std::unique_ptr<DeviceBuffer<unsigned char>> buffer;
};

// Typed handle: solver code holds these and uses them to look up the typed
// device pointer at kernel-launch time.
template<typename T>
class AttribHandle {
public:
    AttribHandle() = default;
    explicit AttribHandle(std::size_t index) : index_(index) {}
    std::size_t index() const noexcept { return index_; }
    bool valid() const noexcept { return index_ != npos; }
private:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
    std::size_t index_ = npos;
};

class ParticleStore {
public:
    explicit ParticleStore(std::size_t capacity, cudaStream_t stream = 0);

    ParticleStore(const ParticleStore&) = delete;
    ParticleStore& operator=(const ParticleStore&) = delete;
    ParticleStore(ParticleStore&&) = default;
    ParticleStore& operator=(ParticleStore&&) = default;

    // ---- Always-present attributes ----
    AttribHandle<Vec3f> position_handle() const noexcept { return position_; }
    AttribHandle<Vec3f> velocity_handle() const noexcept { return velocity_; }

    Vec3f*       positions()       { return typed<Vec3f>(position_); }
    Vec3f*       velocities()      { return typed<Vec3f>(velocity_); }
    const Vec3f* positions()  const { return typed_const<Vec3f>(position_); }
    const Vec3f* velocities() const { return typed_const<Vec3f>(velocity_); }

    // ---- Custom attributes ----
    template<typename T>
    AttribHandle<T> register_attribute(std::string_view name, AttribType type);

    template<typename T>
    T* attribute_data(AttribHandle<T> h) { return typed<T>(h); }
    template<typename T>
    const T* attribute_data(AttribHandle<T> h) const { return typed_const<T>(h); }

    bool has_attribute(std::string_view name) const;

    // ---- Lifecycle ----
    void resize(std::size_t count);
    std::size_t count()    const noexcept { return count_; }
    std::size_t capacity() const noexcept { return capacity_; }

    cudaStream_t stream() const noexcept { return stream_; }

private:
    template<typename T>
    T* typed(AttribHandle<T> h) {
        if (!h.valid() || h.index() >= attribs_.size())
            throw std::out_of_range("ParticleStore: invalid handle");
        return reinterpret_cast<T*>(attribs_[h.index()].buffer->data());
    }
    template<typename T>
    const T* typed_const(AttribHandle<T> h) const {
        if (!h.valid() || h.index() >= attribs_.size())
            throw std::out_of_range("ParticleStore: invalid handle");
        return reinterpret_cast<const T*>(attribs_[h.index()].buffer->data());
    }

    std::size_t  capacity_;
    std::size_t  count_ = 0;
    cudaStream_t stream_;

    std::vector<AttribStorage>                       attribs_;
    std::unordered_map<std::string, std::size_t>     name_to_index_;

    AttribHandle<Vec3f> position_;
    AttribHandle<Vec3f> velocity_;
};

template<typename T>
AttribHandle<T> ParticleStore::register_attribute(std::string_view name, AttribType type) {
    std::string sname{name};
    if (name_to_index_.count(sname)) {
        throw std::runtime_error("ParticleStore: attribute already registered: " + sname);
    }
    AttribStorage storage;
    storage.name         = sname;
    storage.type         = type;
    storage.element_size = attrib_element_size(type);
    storage.buffer = std::make_unique<DeviceBuffer<unsigned char>>(
        capacity_ * storage.element_size, stream_);
    storage.buffer->fill_zero();

    std::size_t idx = attribs_.size();
    attribs_.push_back(std::move(storage));
    name_to_index_[sname] = idx;
    return AttribHandle<T>(idx);
}

} // namespace water
