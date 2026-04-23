// Minimal Slang compute shader. Adds 1 to every element of a buffer.
// Phase 1 only verifies that slangc compiles this to SPIR-V; the pipeline
// to actually run it is created in Phase 4.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          uniform RWStructuredBuffer<float> buf,
          uniform uint count)
{
    if (tid.x >= count) return;
    buf[tid.x] += 1.0;
}
