// Fixed-geometry Lanczos-6 gather and conjugate-transpose scatter.
// Compile without fast math / FMA contraction to match the eager float32 map.
struct Complex { float re, im; };

__device__ float sinc_pi(float x) {
    if (x == 0.0f) return 1.0f;
    float a = 3.14159265358979323846f * x;
    return sinf(a) / a;
}

extern "C" __global__ void lanczos_tilt(
    const Complex* input, Complex* output, const float* fx_axis, const float* fy_axis,
    int n, int rows, float m2, float k, float k2, float k_floor,
    float cosine, float sine_over_m, float sine_times_m, float n_pitch, int adjoint)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned int)n * rows) return;
    int idx = (int)tid;
    int column = idx % n;
    int row = idx / n;
    float fx = fx_axis[column], fy = fy_axis[row];
    float transverse = m2 * (fx * fx + fy * fy);
    float kz = sqrtf(fmaxf(k2 - transverse, 0.0f));
    float delta = -transverse / (kz + k);
    float q = cosine * fx + sine_over_m * delta;
    float jacobian = cosine - sine_times_m * fx / fmaxf(kz, k_floor);
    if (!(transverse < k2 && jacobian > 0.0f)) {
        if (!adjoint) output[idx] = {0.0f, 0.0f};
        return;
    }
    float sample = q * n_pitch + n / 2;
    int base = (int)floorf(sample);
    Complex value = {0.0f, 0.0f};
    if (adjoint) {
        value.re = input[idx].re * jacobian;
        value.im = input[idx].im * jacobian;
    }
    Complex sum = {0.0f, 0.0f};
    for (int offset = -5; offset <= 6; ++offset) {
        int index = base + offset;
        float distance = sample - index;
        if (index < 0 || index >= n || fabsf(distance) >= 6.0f) continue;
        float weight = sinc_pi(distance) * sinc_pi(distance / 6.0f);
        int address = row * n + index;
        if (adjoint) {
            atomicAdd(&output[address].re, value.re * weight);
            atomicAdd(&output[address].im, value.im * weight);
        } else {
            sum.re += input[address].re * weight;
            sum.im += input[address].im * weight;
        }
    }
    if (!adjoint) output[idx] = {sum.re * jacobian, sum.im * jacobian};
}
