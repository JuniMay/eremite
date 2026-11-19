// ---- LFM3-VL's projector: the vision tower's patches as the language model's tokens ----
//
// Rows of FP32 in memory. The products (mm_bf16) read them as BF16 and add up
// in FP32; the norm, the biases and the activation run in FP32.

// LayerNorm of each row of n values: y = w (x - mean) / sqrt(var + eps) + b,
// the variance taken around the row's mean. One threadgroup per row.
kernel void lfm3_layer_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                            device const float *b [[buffer(2)]], device float *y [[buffer(3)]],
                            constant Lfm3Norm &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[32];
  x += row * p.n, y += row * p.n;
  float total = 0;
  for (uint i = t; i < p.n; i += nt) total += x[i];
  float mean = tg_sum(total, tmp, t, nt) / p.n, squares = 0;
  for (uint i = t; i < p.n; i += nt) squares += (x[i] - mean) * (x[i] - mean);
  float r = rsqrt(tg_sum(squares, tmp, t, nt) / p.n + p.eps);
  for (uint i = t; i < p.n; i += nt) y[i] = w[i] * ((x[i] - mean) * r) + b[i];
}

// erf(x), by Abramowitz and Stegun's formula 7.1.26: its error is below
// 1.5e-7.
inline float lfm3_erf(float x) {
  float a = fabs(x), t = 1.0f / (1.0f + 0.3275911f * a);
  float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
  return copysign(1.0f - poly * exp(-a * a), x);
}

// GELU, with erf, of each value plus its column's bias, in rows of n values:
// z = x + b[column], x = z (1 + erf(z / sqrt(2))) / 2.
kernel void lfm3_bias_gelu(device float *x [[buffer(0)]], device const float *b [[buffer(1)]],
                           constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float z = x[i] + b[i % n];
  x[i] = 0.5f * z * (1.0f + lfm3_erf(z * M_SQRT1_2_F));
}

// Each value plus its column's bias, in rows of n values: x += b[column].
kernel void lfm3_add_bias(device float *x [[buffer(0)]], device const float *b [[buffer(1)]],
                          constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  x[i] += b[i % n];
}
