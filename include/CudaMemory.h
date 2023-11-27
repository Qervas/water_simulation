template <typename T>
class CudaMemory {
public:
    CudaMemory(size_t length) : _length(length), _devicePtr(nullptr) {
        if (length > 0) {
            cudaError_t err = cudaMalloc(&_devicePtr, sizeof(T) * length);
            if (err != cudaSuccess) {
                // Handle allocation failure
                throw std::runtime_error("Failed to allocate CUDA memory");
            }
        }
    }

    ~CudaMemory() {
        if (_devicePtr) {
            cudaFree(_devicePtr);
        }
    }

    T* get() const { return _devicePtr; }

    // Disable copy semantics
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    // Enable move semantics
    CudaMemory(CudaMemory&& other) noexcept : _devicePtr(other._devicePtr), _length(other._length) {
        other._devicePtr = nullptr;
        other._length = 0;
    }
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            _devicePtr = other._devicePtr;
            _length = other._length;
            other._devicePtr = nullptr;
            other._length = 0;
        }
        return *this;
    }

private:
    T* _devicePtr;
    size_t _length;
};
