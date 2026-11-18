// ---- LFM3: the language model of LiquidAI's d1 decision models ----
//
// A pass's rows are FP32 in memory. The products (mm_bf16) read them as BF16
// and add up in FP32. The norms, the short convolution and the activation run
// in FP32.

// An RMSNorm's row length and epsilon.
struct Lfm3Norm {
  uint n;
  float eps;
};

// The rows of a short convolution: dim channels, the filter's width, and the
// rows of in_proj's outputs before the block's first row.
struct Lfm3Conv {
  uint dim, width, before;
};

// Embedding: row r of x gets token ids[r]'s BF16 row of the table as FP32,
// for each row whose id is 0 or more. Thread x: one of the row's dim values.
kernel void lfm3_embed(device const ushort *table [[buffer(0)]], device const int *ids [[buffer(1)]],
                       device float *x [[buffer(2)]], constant uint &dim [[buffer(3)]],
                       uint2 i [[thread_position_in_grid]]) {
  int id = ids[i.y];
  if (id >= 0) x[i.y * dim + i.x] = bf16f(table[uint(id) * dim + i.x]);
}

// RMSNorm of each row of n values: y = w (x / sqrt(mean(x^2) + eps)). One
// threadgroup per row.
kernel void lfm3_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                      device float *y [[buffer(2)]], constant Lfm3Norm &p [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                      uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[32];
  x += row * p.n, y += row * p.n;
  float squares = 0;
  for (uint i = t; i < p.n; i += nt) squares += x[i] * x[i];
  float r = rsqrt(tg_sum(squares, tmp, t, nt) / p.n + p.eps);
  for (uint i = t; i < p.n; i += nt) y[i] = w[i] * (x[i] * r);
}

// A sandwich norm's residual: x += w (y / sqrt(mean(y^2) + eps)) for each
// row of n values. One threadgroup per row.
kernel void lfm3_norm_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                          device const float *w [[buffer(2)]], constant Lfm3Norm &p [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[32];
  x += row * p.n, y += row * p.n;
  float squares = 0;
  for (uint i = t; i < p.n; i += nt) squares += y[i] * y[i];
  float r = rsqrt(tg_sum(squares, tmp, t, nt) / p.n + p.eps);
  for (uint i = t; i < p.n; i += nt) x[i] += w[i] * (y[i] * r);
}

// Copies count floats from x to y.
kernel void lfm3_copy(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count) y[i] = x[i];
}

// The gated short convolution of a block of rows. Each row of bcx holds
// in_proj's outputs B, C and x, dim values each, for the block's rows and the
// `before` rows ahead of them. With u = B x, channel c of output row t is
// C[t][c] times the sum over j < width of w[c][j] u[t - width + 1 + j][c].
// Rows before the sequence's start count as zero. Thread x: the channel;
// thread y: the block's row.
kernel void lfm3_conv(device const float *bcx [[buffer(0)]], device const float *w [[buffer(1)]],
                      device float *y [[buffer(2)]], constant Lfm3Conv &p [[buffer(3)]],
                      uint2 i [[thread_position_in_grid]]) {
  uint c = i.x, t = i.y + p.before;
  float sum = 0;
  for (uint j = 0; j < p.width; j++) {
    if (t + j + 1 < p.width) continue;
    device const float *row = bcx + (t + j + 1 - p.width) * 3 * p.dim;
    sum += w[c * p.width + j] * (row[c] * row[2 * p.dim + c]);
  }
  y[i.y * p.dim + c] = bcx[t * 3 * p.dim + p.dim + c] * sum;
}

// The feed-forward's activation: y = silu(g) u.
kernel void lfm3_swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],
                        device float *y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float a = g[i];
  y[i] = a / (1.0f + exp(-a)) * u[i];
}
