#include "model/lfm3/model.h"
#include "model/lfm3/image.h"
#include "base/clock.h"
#include "base/error.h"
#include "base/file.h"
#include "base/json.h"
#include "engine/memory.h"
#include "engine/pass.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// LFM3's language model in a d1 checkpoint (model.language_model.*), and its
// decision pass. A d1 checkpoint's config.json has model_type lfm3_vl and
// d1_attention "set": a sequence holds a state, then its questions. On the
// global attention layers, the state's rows see every row, and a question's
// rows see the state and their own question, both ways. The sliding
// attention layers and the short convolutions see only rows before.
//
// The pass runs one layer at a time over all of the sequence's rows, in
// blocks of rows. Each layer's work is one command buffer.
//
// LFM3-VL's vision tower (model.vision_tower.*) and projector
// (model.multi_modal_projector.*) turn an image's patches into rows of the
// language model's width, one for each square of 2 x 2 patches. Their weights
// load at the first image.

// The RoPE bases (rope_theta) of the global and the sliding layers.
static constexpr double GLOBAL_THETA = 1e6, SLIDING_THETA = 1e4;
// The value pairs that RoPE turns in each head: half of them on the global
// layers (partial_rotary_factor 0.5), all of them on the sliding ones.
static constexpr int GLOBAL_ROTATED = LFM3_HEAD_DIM / 4,
                     SLIDING_ROTATED = LFM3_HEAD_DIM / 2;
// Rows of a pass's block: each layer takes its rows in blocks of this many,
// so that a long sequence's temporary buffers stay near 480 MiB, 120 KiB a
// row. Block boundaries change no result.
static constexpr int BLOCK_ROWS = 4096;
// The rows that the attention kernel takes in one threadgroup, and its key
// blocks: the BF16 attention arrays hold whole tiles of them, and as many
// zero rows after the sequence's rows.
static constexpr int ATTEND_ROWS = 64;
// The threads of a row's RMSNorm threadgroup, and of the other kernels'
// threadgroups over a row's values.
static constexpr int ROW_THREADS = 256;
// The threads of lfm3_heads's threadgroup: one pair of a head's values each.
static constexpr int HEAD_THREADS = LFM3_HEAD_DIM / 2;
// The threadgroups of lfm3_heads for one row: the query heads, then the key
// heads, then the value heads.
static constexpr int HEAD_GROUPS = LFM3_HEADS + 2 * LFM3_KV_HEADS;
// The threads of lfm3_attend's threadgroup: its 4 simdgroups.
static constexpr int ATTEND_THREADS = 128;
// The rows of the convolution's filter that reach back before a row.
static constexpr int CONV_BEFORE = LFM3_CONV_WIDTH - 1;
// The stack of a pass: it holds little more than one row of FP32 values.
static constexpr size_t PASS_STACK_BYTES = 1u << 20;
// The vision tower's RoPE base (rope_theta) and its norms' epsilon
// (norm_eps), and the projector LayerNorm's epsilon
// (projector_layer_norm_eps).
static constexpr double VISION_THETA = 1e6;
static constexpr float VISION_NORM_EPS = 1e-5f, PROJECTOR_EPS = 1e-6f;
// The pairs of a vision head that RoPE turns by a patch's row; as many turn by
// its column.
static constexpr int VISION_ROW_PAIRS = LFM3_VISION_HEAD_DIM / 4;
// The side of the vision convolution's filter (conv_kernel_size).
static constexpr int VISION_FILTER = 3;
// The threads of lfm3_vision_heads's threadgroup, one pair of a head's values
// each, and its threadgroups for one patch: the query heads, then the key
// heads, then the value heads.
static constexpr int VISION_HEAD_THREADS = LFM3_VISION_HEAD_DIM / 2,
                     VISION_HEAD_GROUPS =
                         LFM3_VISION_HEADS + 2 * LFM3_VISION_KV_HEADS;
// The width of a square's patches joined: the projector's first product reads
// rows of this many values.
static constexpr int JOINED_WIDTH = LFM3_VISION_WIDTH * LFM3_MERGE * LFM3_MERGE;

// The GPU that the model runs on.
static Gpu *gpu;
// The model's workspace slot, from its loader.
static MemorySlot *work_slot;
static TensorIndex *checkpoint;
static WeightGroup *text_weights;
// The vision tower's and the projector's weights, from the first image on.
static WeightGroup *vision_weights;
// The number of the last command buffer that the running pass committed; 0
// for none. Each pass has its own.
static uint64_t last_committed;
// The running pass's logits from its last decision, and their room. Each pass
// has its own.
static float *results;
static int results_room;
// The rows of the decisions since the model loaded.
static uint64_t decision_rows;
// The least memory that a pass over one full block needs: the weights and
// the pass's buffers.
static uint64_t required_bytes;
static int K_lfm3_embed, K_lfm3_norm, K_lfm3_norm_add, K_lfm3_copy, K_lfm3_conv,
    K_lfm3_swiglu, K_lfm3_heads, K_lfm3_attend, K_mm_bf16, K_lfm3_vision_conv,
    K_lfm3_vision_heads, K_lfm3_vision_attend, K_lfm3_vision_add,
    K_lfm3_layer_norm, K_lfm3_bias_gelu, K_lfm3_add_bias;

// A BF16 matrix of n rows of k values.
typedef struct {
  Buf w;
  int n, k;
} Lin;

// A layer's kind (layer_types).
typedef enum {
  // A gated short convolution over the rows before.
  LAYER_CONV,
  // Attention over a window of the rows before.
  LAYER_SLIDING,
  // Attention under the decision layout: the state sees every row, and a
  // question sees the state and itself.
  LAYER_GLOBAL
} LayerKind;

typedef struct {
  LayerKind kind;
  // The norms around the operator and the feed-forward, as FP32 weights.
  Buf operator_norm, post_operator_norm, ffn_norm, post_ffn_norm;
  // The short convolution: in_proj's B, C and x, the filter (LFM3_DIM rows of
  // LFM3_CONV_WIDTH), and out_proj.
  Lin in_proj, out_proj;
  Buf conv;
  // Attention: the projections, the per-head norms and the output gate.
  Lin q, k, v, gate, o;
  Buf q_norm, k_norm;
  // The feed-forward: w1 (gate), w3 (up) and w2 (down).
  Lin w1, w3, w2;
} Layer;

static Layer layers[LFM3_LAYERS];
// The embedding table (BF16, tied to the output head), the final norm, and
// RoPE's inverse frequencies on the global and the sliding layers.
static Buf embed, final_norm, global_inv_freq, sliding_inv_freq;

// A layer of the vision tower.
typedef struct {
  // Whether the layer attends (full_attention); else it convolves (conv).
  int attends;
  // The norms before the operator and the feed-forward, as FP32 weights.
  Buf operator_norm, ffn_norm;
  // The operator's output product (out_proj), the convolution's or attention's.
  Lin out_proj;
  // The convolution: in_proj's B, C and x, and the filter (LFM3_VISION_WIDTH
  // rows of 3 x 3, FP32).
  Lin in_proj;
  Buf filter;
  // Attention: the projections and the per-head norms.
  Lin q, k, v;
  Buf q_norm, k_norm;
  // The feed-forward: w1 (gate), w3 (up) and w2 (down).
  Lin w1, w3, w2;
} VisionLayer;

static VisionLayer vision_layers[LFM3_VISION_LAYERS];
// The patchifier and its norm, RoPE's inverse frequencies, and the projector:
// its LayerNorm's weights and bias, and its two products with their biases.
static Lin patchifier, projector_in, projector_out;
static Buf patch_norm, vision_inv_freq, projector_norm, projector_norm_bias,
    projector_in_bias, projector_out_bias;

// The parameters of the kernels, as src/kernels/lfm3.metal and mm_bf16 take
// them.
typedef struct {
  uint32_t n;
  float eps;
} NormParams;

typedef struct {
  uint32_t dim, width, before;
} ConvParams;

typedef struct {
  uint32_t rows, first, padded, rotated;
  float eps;
} HeadsParams;

typedef struct {
  uint32_t rows, first, block_rows, padded, state, window;
} AttendParams;

typedef struct {
  uint32_t m, n, k, round, lda, ldy;
} ProductParams;

typedef struct {
  uint32_t dim, grid_w, grid_h;
} VisionConvParams;

typedef struct {
  uint32_t rows, padded, grid_w;
  float eps;
} VisionHeadsParams;

typedef struct {
  uint32_t rows, padded;
} VisionAttendParams;

// The buffers of a pass over a sequence: its rows' and one block's.
typedef struct {
  // The sequence's rows: the residual stream (FP32), each row's token and
  // question (0 for the state's rows), the questions' spans, and attention's
  // BF16 inputs (queries of 16 heads, keys and values of 4) with its gates.
  Buf x, ids, groups, spans, qh, kh, vh, gates;
  // A block's rows: normed inputs (after the convolution's rows before the
  // block), in_proj's outputs, an operator's output and its projection, the
  // queries, keys and values, and the feed-forward's two halves.
  Buf normed, bcx, out, projected, q, k, v, up, gated;
} PassBuffers;

static constexpr int PASS_BUFFERS = sizeof(PassBuffers) / sizeof(Buf);

// ---- weights ----

// Returns the number at key of the JSON object. Ends the program when it is
// missing or not a number.
static double read_number(const Json *object, const char *key,
                          const char *path) {
  const Json *value = json_get(object, key);
  if (!value || value->type != J_NUM) die("%s: no number %s", path, key);
  return value->num;
}

// Returns whether the JSON value is true.
static int is_true(const Json *value) {
  return value && value->type == J_BOOL && value->num;
}

// Checks that config.json describes the d1 model that the kernels run: its
// shape, its layers' kinds, RoPE, the norms and the decision layout, and the
// vision tower's and the projector's. Ends the program if not.
static void check_config(const char *dir) {
  char path[1024];
  snprintf(path, sizeof path, "%s/config.json", dir);
  char *text = read_file(path, 0);
  Json *root = json_parse(text);
  const Json *layout = json_get(root, "d1_attention");
  if (!layout || layout->type != J_STR || strcmp(layout->s, "set"))
    die("%s: not a d1 decision model with d1_attention \"set\"", path);
  const Json *config = json_get(root, "text_config");
  if (!config) die("%s: no text_config", path);
  const Json *vision = json_get(root, "vision_config"),
             *vision_rope = json_get(vision, "rope_parameters");
  if (!vision || !vision_rope)
    die("%s: no vision_config with rope_parameters", path);
  const Json *rope = json_get(config, "rope_parameters"),
             *global = json_get(rope, "full_attention"),
             *sliding = json_get(rope, "sliding_attention");
  if (!global || !sliding)
    die("%s: no rope_parameters for both layer kinds", path);

  const struct {
    const Json *object;
    const char *key;
    double value;
  } shape[] = {{config, "hidden_size", LFM3_DIM},
               {config, "num_hidden_layers", LFM3_LAYERS},
               {config, "vocab_size", LFM3_VOCAB},
               {config, "intermediate_size", LFM3_HIDDEN},
               {config, "num_attention_heads", LFM3_HEADS},
               {config, "num_key_value_heads", LFM3_KV_HEADS},
               {config, "head_dim", LFM3_HEAD_DIM},
               {config, "conv_L_cache", LFM3_CONV_WIDTH},
               {config, "sliding_window", LFM3_WINDOW},
               {config, "max_position_embeddings", LFM3_POSITIONS},
               {config, "norm_eps", LFM3_NORM_EPS},
               {global, "rope_theta", GLOBAL_THETA},
               {global, "partial_rotary_factor",
                (double)GLOBAL_ROTATED / SLIDING_ROTATED},
               {sliding, "rope_theta", SLIDING_THETA},
               {vision, "hidden_size", LFM3_VISION_WIDTH},
               {vision, "num_hidden_layers", LFM3_VISION_LAYERS},
               {vision, "intermediate_size", LFM3_VISION_HIDDEN},
               {vision, "num_attention_heads", LFM3_VISION_HEADS},
               {vision, "num_key_value_heads", LFM3_VISION_KV_HEADS},
               {vision, "in_channels", 3},
               {vision, "patch_size", LFM3_PATCH},
               {vision, "spatial_merge_size", LFM3_MERGE},
               {vision, "temporal_patch_size", LFM3_FRAMES},
               {vision, "conv_kernel_size", VISION_FILTER},
               {vision, "norm_eps", VISION_NORM_EPS},
               {vision_rope, "rope_theta", VISION_THETA},
               {root, "downsample_factor", LFM3_MERGE},
               {root, "projector_hidden_size", LFM3_PROJECTOR_HIDDEN},
               {root, "projector_layer_norm_eps", PROJECTOR_EPS},
               {root, "d1_vision_max_pixels", LFM3_IMAGE_MAX_PIXELS}};

  for (size_t i = 0; i < sizeof shape / sizeof *shape; i++)
    if ((float)read_number(shape[i].object, shape[i].key, path) !=
        (float)shape[i].value)
      die("%s: %s is not %g", path, shape[i].key, shape[i].value);
  if (!is_true(json_get(config, "use_sandwich_norm")) ||
      !is_true(json_get(config, "tie_word_embeddings")) ||
      is_true(json_get(config, "conv_bias")))
    die("%s: not sandwich norms, a tied output head and convolutions without "
        "bias",
        path);
  const Json *rope_type = json_get(vision_rope, "rope_type"),
             *activation = json_get(root, "projector_hidden_act");
  if (!rope_type || rope_type->type != J_STR || strcmp(rope_type->s, "axial") ||
      !activation || activation->type != J_STR ||
      strcmp(activation->s, "gelu") ||
      !is_true(json_get(root, "projector_bias")))
    die("%s: not the vision tower's axial RoPE and a projector with biases "
        "and GELU",
        path);
  const Json *vision_kinds = json_get(vision, "layer_types");
  if (!vision_kinds || vision_kinds->type != J_ARR ||
      vision_kinds->n != LFM3_VISION_LAYERS)
    die("%s: vision_config's layer_types does not list %d layers", path,
        LFM3_VISION_LAYERS);
  for (int l = 0; l < LFM3_VISION_LAYERS; l++) {
    const Json *kind = &vision_kinds->kid[l];
    if (kind->type != J_STR ||
        (strcmp(kind->s, "conv") && strcmp(kind->s, "full_attention")))
      die("%s: vision layer type %s", path,
          kind->type == J_STR ? kind->s : "that is no string");
    vision_layers[l].attends = !strcmp(kind->s, "full_attention");
  }
  const Json *kinds = json_get(config, "layer_types");
  if (!kinds || kinds->type != J_ARR || kinds->n != LFM3_LAYERS)
    die("%s: layer_types does not list %d layers", path, LFM3_LAYERS);
  for (int l = 0; l < LFM3_LAYERS; l++) {
    const Json *kind = &kinds->kid[l];
    if (kind->type != J_STR) die("%s: layer_types holds a non-string", path);
    if (!strcmp(kind->s, "conv"))
      layers[l].kind = LAYER_CONV;
    else if (!strcmp(kind->s, "sliding_attention"))
      layers[l].kind = LAYER_SLIDING;
    else if (!strcmp(kind->s, "full_attention"))
      layers[l].kind = LAYER_GLOBAL;
    else
      die("%s: layer type %s", path, kind->s);
  }
  json_free(root);
  free(text);
}

// Adds the BF16 matrix named model.name.weight to the weight group.
static void add_matrix(WeightGroup *group, Lin *line, const char *name) {
  Tensor tensor = tensor_index_get(checkpoint, "model.%s.weight", name);
  if (tensor.dtype != 'b' || tensor.ndim != 2)
    die("%s: not a BF16 matrix", name);
  *line = (Lin){.n = (int)tensor.shape[0], .k = (int)tensor.shape[1]};
  weight_group_add_tensor(group, &line->w, tensor, 1);
}

// Adds the weights named model.name.weight to the weight group, as FP32.
static void add_vector(WeightGroup *group, Buf *target, const char *name) {
  weight_group_add_vector(group, target,
                          tensor_index_get(checkpoint, "model.%s.weight", name),
                          0, 0);
}

// Adds the bias named model.name.bias to the weight group, as FP32.
static void add_bias(WeightGroup *group, Buf *target, const char *name) {
  weight_group_add_vector(
      group, target, tensor_index_get(checkpoint, "model.%s.bias", name), 0, 0);
}

// RoPE's inverse frequencies: count of them, for a base theta and a length
// of dim values.
typedef struct {
  double theta;
  int count, dim;
} Frequencies;

// Writes RoPE's inverse frequencies theta^(-2 i / dim), as the reference
// computes them in FP32.
static void fill_inv_freq(Buf *buffer, const void *parameters) {
  const Frequencies *f = parameters;
  float *out = (float *)buffer->ptr;
  for (int i = 0; i < f->count; i++)
    out[i] = 1.0f / powf((float)f->theta, (float)(2 * i) / (float)f->dim);
}

// Adds every weight of the language model to the weight group.
static void add_text_weights(void) {
  char name[128];
  WeightGroup *g = text_weights;
  for (int l = 0; l < LFM3_LAYERS; l++) {
    Layer *y = &layers[l];
#define NAME(part)                                                             \
  (snprintf(name, sizeof name, "language_model.layers.%d.%s", l, part), name)
    add_vector(g, &y->operator_norm, NAME("operator_norm"));
    add_vector(g, &y->post_operator_norm, NAME("post_operator_norm"));
    add_vector(g, &y->ffn_norm, NAME("ffn_norm"));
    add_vector(g, &y->post_ffn_norm, NAME("post_ffn_norm"));
    if (y->kind == LAYER_CONV) {
      add_matrix(g, &y->in_proj, NAME("conv.in_proj"));
      add_matrix(g, &y->out_proj, NAME("conv.out_proj"));
      add_vector(g, &y->conv, NAME("conv.conv"));
    } else {
      add_matrix(g, &y->q, NAME("self_attn.q_proj"));
      add_matrix(g, &y->k, NAME("self_attn.k_proj"));
      add_matrix(g, &y->v, NAME("self_attn.v_proj"));
      add_matrix(g, &y->gate, NAME("self_attn.gate_proj"));
      add_matrix(g, &y->o, NAME("self_attn.o_proj"));
      add_vector(g, &y->q_norm, NAME("self_attn.q_norm"));
      add_vector(g, &y->k_norm, NAME("self_attn.k_norm"));
    }
    add_matrix(g, &y->w1, NAME("feed_forward.w1"));
    add_matrix(g, &y->w3, NAME("feed_forward.w3"));
    add_matrix(g, &y->w2, NAME("feed_forward.w2"));
#undef NAME
  }
  Tensor table =
      tensor_index_get(checkpoint, "model.language_model.embed_tokens.weight");
  if (table.dtype != 'b' || table.shape[0] != LFM3_VOCAB ||
      table.shape[1] != LFM3_DIM)
    die("the embedding table is not %d BF16 rows of %d", LFM3_VOCAB, LFM3_DIM);
  weight_group_add_tensor(g, &embed, table, 1);
  add_vector(g, &final_norm, "language_model.embedding_norm");
  const int pairs = LFM3_HEAD_DIM / 2;
  weight_group_add_generated(
      g, &global_inv_freq, pairs * sizeof(float), fill_inv_freq,
      &(Frequencies){GLOBAL_THETA, pairs, LFM3_HEAD_DIM}, sizeof(Frequencies));
  weight_group_add_generated(
      g, &sliding_inv_freq, pairs * sizeof(float), fill_inv_freq,
      &(Frequencies){SLIDING_THETA, pairs, LFM3_HEAD_DIM}, sizeof(Frequencies));
}

// Adds every weight of the vision tower and the projector to the weight
// group. RoPE's frequencies are those of half a head's values, for a patch's
// row or column (Lfm3VlVisionRotaryEmbedding).
static void add_vision_weights(void) {
  char name[128];
  WeightGroup *g = vision_weights;
  for (int l = 0; l < LFM3_VISION_LAYERS; l++) {
    VisionLayer *y = &vision_layers[l];
#define NAME(part)                                                             \
  (snprintf(name, sizeof name, "vision_tower.layers.%d.%s", l, part), name)
    add_vector(g, &y->operator_norm, NAME("operator_norm"));
    add_vector(g, &y->ffn_norm, NAME("ffn_norm"));
    if (y->attends) {
      add_matrix(g, &y->q, NAME("self_attn.q_proj"));
      add_matrix(g, &y->k, NAME("self_attn.k_proj"));
      add_matrix(g, &y->v, NAME("self_attn.v_proj"));
      add_matrix(g, &y->out_proj, NAME("self_attn.out_proj"));
      add_vector(g, &y->q_norm, NAME("self_attn.q_layernorm"));
      add_vector(g, &y->k_norm, NAME("self_attn.k_layernorm"));
    } else {
      add_matrix(g, &y->in_proj, NAME("conv.in_proj"));
      add_matrix(g, &y->out_proj, NAME("conv.out_proj"));
      add_vector(g, &y->filter, NAME("conv.spatial_conv"));
    }
    add_matrix(g, &y->w1, NAME("feed_forward.w1"));
    add_matrix(g, &y->w3, NAME("feed_forward.w3"));
    add_matrix(g, &y->w2, NAME("feed_forward.w2"));
#undef NAME
  }
  add_matrix(g, &patchifier, "vision_tower.patchifier");
  add_vector(g, &patch_norm, "vision_tower.embedding_norm");
  add_vector(g, &projector_norm, "multi_modal_projector.layer_norm");
  add_bias(g, &projector_norm_bias, "multi_modal_projector.layer_norm");
  add_matrix(g, &projector_in, "multi_modal_projector.linear_1");
  add_bias(g, &projector_in_bias, "multi_modal_projector.linear_1");
  add_matrix(g, &projector_out, "multi_modal_projector.linear_2");
  add_bias(g, &projector_out_bias, "multi_modal_projector.linear_2");
  const int half = LFM3_VISION_HEAD_DIM / 2;
  weight_group_add_generated(
      g, &vision_inv_freq, VISION_ROW_PAIRS * sizeof(float), fill_inv_freq,
      &(Frequencies){VISION_THETA, VISION_ROW_PAIRS, half},
      sizeof(Frequencies));
}

// ---- the pass's buffers ----

// Rounds n up to a multiple of unit.
static int round_up(int n, int unit) { return (n + unit - 1) / unit * unit; }

// The rows of each head in the BF16 attention arrays for n rows.
static int attention_rows(int n) {
  return round_up(n, ATTEND_ROWS) + ATTEND_ROWS;
}

// Writes the bytes of each buffer of a pass over n rows with that many
// questions, in PassBuffers's order.
static void count_buffer_bytes(int n, int questions,
                               size_t bytes[PASS_BUFFERS]) {
  size_t rows = (size_t)n, block = (size_t)(n < BLOCK_ROWS ? n : BLOCK_ROWS),
         padded = (size_t)attention_rows(n), wide = block + CONV_BEFORE;
  size_t sizes[PASS_BUFFERS] = {rows * LFM3_DIM * 4,
                                rows * 4,
                                rows * 4,
                                (size_t)(questions + 1) * 8,
                                LFM3_HEADS * padded * LFM3_HEAD_DIM * 2,
                                LFM3_KV_HEADS * padded * LFM3_HEAD_DIM * 2,
                                LFM3_KV_HEADS * padded * LFM3_HEAD_DIM * 2,
                                rows * LFM3_HEADS * 4,
                                wide * LFM3_DIM * 4,
                                wide * 3 * LFM3_DIM * 4,
                                block * LFM3_DIM * 4,
                                block * LFM3_DIM * 4,
                                block * LFM3_HEADS * LFM3_HEAD_DIM * 4,
                                block * LFM3_KV_HEADS * LFM3_HEAD_DIM * 4,
                                block * LFM3_KV_HEADS * LFM3_HEAD_DIM * 4,
                                block * LFM3_HIDDEN * 4,
                                block * LFM3_HIDDEN * 4};
  memcpy(bytes, sizes, sizeof sizes);
}

// Returns the bytes that the buffers of a pass over n rows with that many
// questions take.
static uint64_t count_pass_bytes(int n, int questions) {
  size_t bytes[PASS_BUFFERS];
  count_buffer_bytes(n, questions, bytes);
  uint64_t total = 0;
  for (int i = 0; i < PASS_BUFFERS; i++)
    total += gpu_round_up_to_pages(bytes[i]);
  return total;
}

// Frees the pass's buffers once the GPU has finished with them.
static void free_pass_buffers(PassBuffers *b) {
  Buf *all = &b->x;
  for (int i = 0; i < PASS_BUFFERS; i++)
    if (all[i].size) gpu_mark_to_free(&all[i]);
}

// Allocates the buffers of a pass over n rows with that many questions, in
// the workspace slot, and zeroes the attention arrays' rows after the
// sequence's. Returns 0, with none allocated, when they do not fit.
static int allocate_pass_buffers(PassBuffers *b, int n, int questions) {
  size_t bytes[PASS_BUFFERS];
  count_buffer_bytes(n, questions, bytes);
  *b = (PassBuffers){0};
  Buf *all = &b->x;
  for (int i = 0; i < PASS_BUFFERS; i++) {
    all[i] = gpu_try_alloc_required(gpu, work_slot, bytes[i]);
    if (!all[i].size) {
      free_pass_buffers(b);
      return 0;
    }
    gpu_set_resident(&all[i], 1);
  }
  gpu_commit_residency(gpu);
  size_t padded = (size_t)attention_rows(n), row = LFM3_HEAD_DIM * 2,
         after = (padded - (size_t)n) * row;
  for (int h = 0; h < LFM3_HEADS; h++) {
    memset(b->qh.ptr + ((size_t)h * padded + (size_t)n) * row, 0, after);
    if (h >= LFM3_KV_HEADS) continue;
    memset(b->kh.ptr + ((size_t)h * padded + (size_t)n) * row, 0, after);
    memset(b->vh.ptr + ((size_t)h * padded + (size_t)n) * row, 0, after);
  }
  return 1;
}

// ---- the pass's work ----

// Commits the command buffer, and records it as the running pass's last one.
static void commit_work(GpuCommandBuffer *c) {
  last_committed = c->number;
  gpu_commit(c);
}

static int committed_finished(void *number) {
  return gpu_has_finished(gpu, (uint64_t)(uintptr_t)number);
}

static void committed_wait(void *number) {
  gpu_wait_until_finished(gpu, (uint64_t)(uintptr_t)number);
}

// Waits for the GPU work that the running pass committed: inside a pass, by
// yielding; outside passes, for all GPU work.
static void wait_for_work(void) {
  if (!pass_current()) {
    gpu_sync(gpu);
    return;
  }
  if (last_committed)
    pass_wait(&(PassWait){committed_finished, committed_wait,
                          (void *)(uintptr_t)last_committed, PASS_WAIT_GPU,
                          NULL});
  gpu_check_finished(gpu);
}

// The bytes before row r of rows of LFM3_DIM floats.
static size_t row_bytes(int r) { return (size_t)r * LFM3_DIM * 4; }

// y = x W^T for m rows of x (row stride lda floats, from byte x_off) into rows
// of y (row stride ldy floats, from byte y_off).
static void multiply(GpuCommandBuffer *c, const Lin *w, int m, const Buf *x,
                     size_t x_off, int lda, const Buf *y, size_t y_off,
                     int ldy) {
  ProductParams p = {(uint32_t)m, (uint32_t)w->n, (uint32_t)w->k,
                     0,           (uint32_t)lda,  (uint32_t)ldy};
  gpu_select_kernel(c, K_mm_bf16);
  gpu_bind_buffer(c, 0, &w->w, 0), gpu_bind_buffer(c, 2, x, x_off);
  gpu_bind_buffer(c, 3, y, y_off), gpu_copy_bytes(c, 4, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)(w->n + 63) / 64, (uint32_t)(m + 63) / 64, 1, 128,
                 1, 1);
}

// y = RMSNorm(x) w for m rows of LFM3_DIM values, from bytes x_off and y_off.
static void normalize(GpuCommandBuffer *c, const Buf *x, size_t x_off,
                      const Buf *w, const Buf *y, size_t y_off, int m) {
  NormParams p = {LFM3_DIM, LFM3_NORM_EPS};
  gpu_select_kernel(c, K_lfm3_norm);
  gpu_bind_buffer(c, 0, x, x_off), gpu_bind_buffer(c, 1, w, 0);
  gpu_bind_buffer(c, 2, y, y_off), gpu_copy_bytes(c, 3, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)m, 1, 1, ROW_THREADS, 1, 1);
}

// x += RMSNorm(y) w for m rows of LFM3_DIM values from byte x_off of x: a
// sandwich norm's residual.
static void add_normalized(GpuCommandBuffer *c, const Buf *x, size_t x_off,
                           const Buf *y, const Buf *w, int m) {
  NormParams p = {LFM3_DIM, LFM3_NORM_EPS};
  gpu_select_kernel(c, K_lfm3_norm_add);
  gpu_bind_buffer(c, 0, x, x_off), gpu_bind_buffer(c, 1, y, 0);
  gpu_bind_buffer(c, 2, w, 0), gpu_copy_bytes(c, 3, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)m, 1, 1, ROW_THREADS, 1, 1);
}

// The rows of the block that starts at row first of n.
static int block_rows(int first, int n) {
  return n - first < BLOCK_ROWS ? n - first : BLOCK_ROWS;
}

// Layer y's short convolution over the sequence's n rows, block by block. A
// block's in_proj also takes the CONV_BEFORE rows before it, normed from the
// layer's input. Those rows stay in the normed buffer from the block before,
// a full block, since the residual has changed the stream's rows by then:
// they move to its front.
static void run_conv(GpuCommandBuffer *c, const Layer *y, PassBuffers *b,
                     int n) {
  int before = 0;
  for (int first = 0; first < n; first += BLOCK_ROWS) {
    int m = block_rows(first, n),
        kept = first < CONV_BEFORE ? first : CONV_BEFORE;
    if (kept) {
      uint32_t count = (uint32_t)kept * LFM3_DIM;
      gpu_select_kernel(c, K_lfm3_copy);
      gpu_bind_buffer(c, 0, &b->normed, row_bytes(before + BLOCK_ROWS - kept));
      gpu_bind_buffer(c, 1, &b->normed, 0);
      gpu_copy_bytes(c, 2, &count, sizeof count);
      gpu_run_kernel(c, (count + ROW_THREADS - 1) / ROW_THREADS, 1, 1,
                     ROW_THREADS, 1, 1);
    }
    before = kept;
    normalize(c, &b->x, row_bytes(first), &y->operator_norm, &b->normed,
              row_bytes(before), m);
    multiply(c, &y->in_proj, before + m, &b->normed, 0, LFM3_DIM, &b->bcx, 0,
             3 * LFM3_DIM);
    ConvParams p = {LFM3_DIM, LFM3_CONV_WIDTH, (uint32_t)before};
    gpu_select_kernel(c, K_lfm3_conv);
    gpu_bind_buffer(c, 0, &b->bcx, 0), gpu_bind_buffer(c, 1, &y->conv, 0);
    gpu_bind_buffer(c, 2, &b->out, 0), gpu_copy_bytes(c, 3, &p, sizeof p);
    gpu_run_kernel(c, LFM3_DIM / ROW_THREADS, (uint32_t)m, 1, ROW_THREADS, 1,
                   1);
    multiply(c, &y->out_proj, m, &b->out, 0, LFM3_DIM, &b->projected, 0,
             LFM3_DIM);
    add_normalized(c, &b->x, row_bytes(first), &b->projected,
                   &y->post_operator_norm, m);
  }
}

// Layer y's attention over the sequence's n rows, whose first `state` rows
// are the state. First each block's queries, keys, values and gates; then,
// once every row's keys are there, each block's attention, its projection and
// its residual.
static void run_attention(GpuCommandBuffer *c, const Layer *y, PassBuffers *b,
                          int n, int state) {
  int sliding = y->kind == LAYER_SLIDING;
  uint32_t padded = (uint32_t)attention_rows(n);
  for (int first = 0; first < n; first += BLOCK_ROWS) {
    int m = block_rows(first, n);
    normalize(c, &b->x, row_bytes(first), &y->operator_norm, &b->normed, 0, m);
    multiply(c, &y->q, m, &b->normed, 0, LFM3_DIM, &b->q, 0,
             LFM3_HEADS * LFM3_HEAD_DIM);
    multiply(c, &y->k, m, &b->normed, 0, LFM3_DIM, &b->k, 0,
             LFM3_KV_HEADS * LFM3_HEAD_DIM);
    multiply(c, &y->v, m, &b->normed, 0, LFM3_DIM, &b->v, 0,
             LFM3_KV_HEADS * LFM3_HEAD_DIM);
    multiply(c, &y->gate, m, &b->normed, 0, LFM3_DIM, &b->gates,
             (size_t)first * LFM3_HEADS * 4, LFM3_HEADS);
    HeadsParams h = {(uint32_t)m, (uint32_t)first, padded,
                     sliding ? SLIDING_ROTATED : GLOBAL_ROTATED, LFM3_NORM_EPS};
    gpu_select_kernel(c, K_lfm3_heads);
    gpu_bind_buffer(c, 0, &b->q, 0), gpu_bind_buffer(c, 1, &b->k, 0);
    gpu_bind_buffer(c, 2, &b->v, 0), gpu_bind_buffer(c, 3, &y->q_norm, 0);
    gpu_bind_buffer(c, 4, &y->k_norm, 0);
    gpu_bind_buffer(c, 5, sliding ? &sliding_inv_freq : &global_inv_freq, 0);
    gpu_bind_buffer(c, 6, &b->qh, 0), gpu_bind_buffer(c, 7, &b->kh, 0);
    gpu_bind_buffer(c, 8, &b->vh, 0), gpu_copy_bytes(c, 9, &h, sizeof h);
    gpu_run_kernel(c, HEAD_GROUPS, (uint32_t)m, 1, HEAD_THREADS, 1, 1);
  }
  for (int first = 0; first < n; first += BLOCK_ROWS) {
    int m = block_rows(first, n);
    AttendParams a = {(uint32_t)n, (uint32_t)first, (uint32_t)m,
                      padded,      (uint32_t)state, sliding ? LFM3_WINDOW : 0};
    gpu_select_kernel(c, K_lfm3_attend);
    gpu_bind_buffer(c, 0, &b->qh, 0), gpu_bind_buffer(c, 1, &b->kh, 0);
    gpu_bind_buffer(c, 2, &b->vh, 0);
    gpu_bind_buffer(c, 3, &b->gates, (size_t)first * LFM3_HEADS * 4);
    gpu_bind_buffer(c, 4, &b->groups, 0), gpu_bind_buffer(c, 5, &b->spans, 0);
    gpu_bind_buffer(c, 6, &b->out, 0), gpu_copy_bytes(c, 7, &a, sizeof a);
    gpu_run_kernel(c, (uint32_t)(m + ATTEND_ROWS - 1) / ATTEND_ROWS, LFM3_HEADS,
                   1, ATTEND_THREADS, 1, 1);
    multiply(c, &y->o, m, &b->out, 0, LFM3_DIM, &b->projected, 0, LFM3_DIM);
    add_normalized(c, &b->x, row_bytes(first), &b->projected,
                   &y->post_operator_norm, m);
  }
}

// Layer y's feed-forward over the sequence's n rows, block by block.
static void run_feed_forward(GpuCommandBuffer *c, const Layer *y,
                             PassBuffers *b, int n) {
  for (int first = 0; first < n; first += BLOCK_ROWS) {
    int m = block_rows(first, n);
    normalize(c, &b->x, row_bytes(first), &y->ffn_norm, &b->normed, 0, m);
    multiply(c, &y->w1, m, &b->normed, 0, LFM3_DIM, &b->gated, 0, LFM3_HIDDEN);
    multiply(c, &y->w3, m, &b->normed, 0, LFM3_DIM, &b->up, 0, LFM3_HIDDEN);
    gpu_select_kernel(c, K_lfm3_swiglu);
    gpu_bind_buffer(c, 0, &b->gated, 0), gpu_bind_buffer(c, 1, &b->up, 0);
    gpu_bind_buffer(c, 2, &b->gated, 0);
    gpu_run_kernel(c, (uint32_t)m * LFM3_HIDDEN / ROW_THREADS, 1, 1,
                   ROW_THREADS, 1, 1);
    multiply(c, &y->w2, m, &b->gated, 0, LFM3_HIDDEN, &b->projected, 0,
             LFM3_DIM);
    add_normalized(c, &b->x, row_bytes(first), &b->projected, &y->post_ffn_norm,
                   m);
  }
}

// Writes the sequence's token ids, each row's question and the questions'
// spans into the pass's buffers. Returns 0 if the questions do not end in
// order after the state, the last at the sequence's end, or a token is out of
// the vocabulary.
static int write_layout(PassBuffers *b, const int *tokens, int n, int state,
                        const int *question_ends, int questions) {
  if (state < 0 || questions < 1 || question_ends[questions - 1] != n) return 0;
  int32_t *ids = (int32_t *)b->ids.ptr;
  uint32_t *groups = (uint32_t *)b->groups.ptr,
           *spans = (uint32_t *)b->spans.ptr;
  spans[0] = 0, spans[1] = (uint32_t)state;
  for (int r = 0; r < state && r < n; r++) groups[r] = 0;
  for (int q = 0, start = state; q < questions; q++) {
    if (question_ends[q] <= start) return 0;
    spans[2 * (q + 1)] = (uint32_t)start;
    spans[2 * (q + 1) + 1] = (uint32_t)question_ends[q];
    for (int r = start; r < question_ends[q]; r++) groups[r] = (uint32_t)q + 1;
    start = question_ends[q];
  }
  for (int r = 0; r < n; r++) {
    if (tokens[r] < 0 || tokens[r] >= LFM3_VOCAB) return 0;
    ids[r] = tokens[r];
  }
  return 1;
}

// Writes the logit of token ids[i] at row rows[i] for each i < count into
// out: the final norm of the row, times the token's embedding row.
static void read_logits(const PassBuffers *b, const int *rows, const int *ids,
                        int count, float *out) {
  const float *x = (const float *)b->x.ptr, *w = (const float *)final_norm.ptr;
  // The table's tensor starts off bytes into its buffer.
  const uint16_t *table =
      (const uint16_t *)((const uint8_t *)embed.ptr + embed.off);
  float normed[LFM3_DIM];
  int normed_row = -1;
  for (int i = 0; i < count; i++) {
    if (rows[i] != normed_row) {
      const float *row = x + (size_t)rows[i] * LFM3_DIM;
      double squares = 0;
      for (int d = 0; d < LFM3_DIM; d++) squares += (double)row[d] * row[d];
      float r = 1.0f / sqrtf((float)(squares / LFM3_DIM) + LFM3_NORM_EPS);
      for (int d = 0; d < LFM3_DIM; d++) normed[d] = w[d] * (row[d] * r);
      normed_row = rows[i];
    }
    const uint16_t *e = table + (size_t)ids[i] * LFM3_DIM;
    double sum = 0;
    for (int d = 0; d < LFM3_DIM; d++) {
      uint32_t bits = (uint32_t)e[d] << 16;
      float value;
      memcpy(&value, &bits, sizeof value);
      sum += (double)normed[d] * value;
    }
    out[i] = (float)sum;
  }
}

static const float *lfm3_decide(const int *tokens, int n, int state,
                                const int *question_ends, int questions,
                                const int *rows, const int *ids, int count,
                                const PassWait *wait) {
  if (n < 1 || n > LFM3_POSITIONS)
    die("a decision takes 1 to %d rows", LFM3_POSITIONS);
  for (int i = 0; i < count; i++)
    if (rows[i] < 0 || rows[i] >= n || ids[i] < 0 || ids[i] >= LFM3_VOCAB)
      die("a decision's logit at row %d of token %d", rows[i], ids[i]);
  PassBuffers b;
  if (!allocate_pass_buffers(&b, n, questions)) return NULL;
  if (!write_layout(&b, tokens, n, state, question_ends, questions))
    die("a decision's questions must follow its state in order up to its end, "
        "with tokens of the vocabulary");
  GpuCommandBuffer c = gpu_begin(gpu);
  uint32_t dim = LFM3_DIM;
  gpu_select_kernel(&c, K_lfm3_embed);
  gpu_bind_buffer(&c, 0, &embed, 0), gpu_bind_buffer(&c, 1, &b.ids, 0);
  gpu_bind_buffer(&c, 2, &b.x, 0), gpu_copy_bytes(&c, 3, &dim, sizeof dim);
  gpu_run_kernel(&c, LFM3_DIM / ROW_THREADS, (uint32_t)n, 1, ROW_THREADS, 1, 1);
  commit_work(&c);
  for (int l = 0; l < LFM3_LAYERS; l++) {
    if (wait) pass_wait(wait);
    const Layer *y = &layers[l];
    c = gpu_begin(gpu);
    if (y->kind == LAYER_CONV)
      run_conv(&c, y, &b, n);
    else
      run_attention(&c, y, &b, n, state);
    run_feed_forward(&c, y, &b, n);
    commit_work(&c);
  }
  wait_for_work();
  if (count > results_room) {
    free(results);
    results = xmalloc((size_t)count * sizeof *results);
    results_room = count;
  }
  read_logits(&b, rows, ids, count, results);
  free_pass_buffers(&b);
  decision_rows += (uint64_t)n;
  return results;
}

// ---- the vision tower and the projector ----

// The buffers of the vision tower's work on an image's patches.
typedef struct {
  // The patches' pixels (FP32 rows of LFM3_PATCH_VALUES), the residual
  // stream, a normed copy, the convolution's in_proj outputs, an operator's
  // output and its projection, the queries, keys and values with attention's
  // BF16 inputs, the feed-forward's two halves, the projector's hidden rows
  // and its output rows.
  Buf pixels, x, normed, bcx, out, projected, q, k, v, qh, kh, vh, up, gated,
      hidden, rows;
} VisionBuffers;

static constexpr int VISION_BUFFERS = sizeof(VisionBuffers) / sizeof(Buf);

// Writes the bytes of each buffer of the vision tower's work on n patches, in
// VisionBuffers's order.
static void count_vision_bytes(int n, size_t bytes[VISION_BUFFERS]) {
  size_t rows = (size_t)n, tokens = rows / (LFM3_MERGE * LFM3_MERGE),
         padded = (size_t)attention_rows(n), width = LFM3_VISION_WIDTH,
         heads = LFM3_VISION_HEADS * LFM3_VISION_HEAD_DIM,
         kv = LFM3_VISION_KV_HEADS * LFM3_VISION_HEAD_DIM;
  size_t sizes[VISION_BUFFERS] = {rows * LFM3_PATCH_VALUES * 4,
                                  rows * width * 4,
                                  rows * width * 4,
                                  rows * 3 * width * 4,
                                  rows * width * 4,
                                  rows * width * 4,
                                  rows * heads * 4,
                                  rows * kv * 4,
                                  rows * kv * 4,
                                  padded * heads * 2,
                                  padded * kv * 2,
                                  padded * kv * 2,
                                  rows * LFM3_VISION_HIDDEN * 4,
                                  rows * LFM3_VISION_HIDDEN * 4,
                                  tokens * LFM3_PROJECTOR_HIDDEN * 4,
                                  tokens * LFM3_DIM * 4};
  memcpy(bytes, sizes, sizeof sizes);
}

// Frees the vision tower's buffers once the GPU has finished with them.
static void free_vision_buffers(VisionBuffers *b) {
  Buf *all = &b->pixels;
  for (int i = 0; i < VISION_BUFFERS; i++)
    if (all[i].size) gpu_mark_to_free(&all[i]);
}

// Allocates the buffers of the vision tower's work on n patches, in the
// workspace slot, and zeroes the attention arrays' rows after the patches'.
// Returns 0, with none allocated, when they do not fit.
static int allocate_vision_buffers(VisionBuffers *b, int n) {
  size_t bytes[VISION_BUFFERS];
  count_vision_bytes(n, bytes);
  *b = (VisionBuffers){0};
  Buf *all = &b->pixels;
  for (int i = 0; i < VISION_BUFFERS; i++) {
    all[i] = gpu_try_alloc_required(gpu, work_slot, bytes[i]);
    if (!all[i].size) {
      free_vision_buffers(b);
      return 0;
    }
    gpu_set_resident(&all[i], 1);
  }
  gpu_commit_residency(gpu);
  size_t padded = (size_t)attention_rows(n), row = LFM3_VISION_HEAD_DIM * 2,
         after = (padded - (size_t)n) * row;
  for (int h = 0; h < LFM3_VISION_HEADS; h++) {
    memset(b->qh.ptr + ((size_t)h * padded + (size_t)n) * row, 0, after);
    if (h >= LFM3_VISION_KV_HEADS) continue;
    memset(b->kh.ptr + ((size_t)h * padded + (size_t)n) * row, 0, after);
    memset(b->vh.ptr + ((size_t)h * padded + (size_t)n) * row, 0, after);
  }
  return 1;
}

// y = RMSNorm(x) w for n rows of LFM3_VISION_WIDTH values.
static void normalize_patches(GpuCommandBuffer *c, const Buf *x, const Buf *w,
                              const Buf *y, int n) {
  NormParams p = {LFM3_VISION_WIDTH, VISION_NORM_EPS};
  gpu_select_kernel(c, K_lfm3_norm);
  gpu_bind_buffer(c, 0, x, 0), gpu_bind_buffer(c, 1, w, 0);
  gpu_bind_buffer(c, 2, y, 0), gpu_copy_bytes(c, 3, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)n, 1, 1, ROW_THREADS, 1, 1);
}

// The residual x += y over n rows of LFM3_VISION_WIDTH values.
static void add_patches(GpuCommandBuffer *c, const Buf *x, const Buf *y,
                        int n) {
  gpu_select_kernel(c, K_lfm3_vision_add);
  gpu_bind_buffer(c, 0, x, 0), gpu_bind_buffer(c, 1, y, 0);
  gpu_run_kernel(c, (uint32_t)n * LFM3_VISION_WIDTH / ROW_THREADS, 1, 1,
                 ROW_THREADS, 1, 1);
}

// Vision layer y over the n patches of a grid of grid_w x grid_h: its
// convolution or its attention, then its feed-forward, each with its residual.
static void run_vision_layer(GpuCommandBuffer *c, const VisionLayer *y,
                             VisionBuffers *b, int n, int grid_w, int grid_h) {
  const int width = LFM3_VISION_WIDTH;
  normalize_patches(c, &b->x, &y->operator_norm, &b->normed, n);
  if (y->attends) {
    uint32_t padded = (uint32_t)attention_rows(n);
    multiply(c, &y->q, n, &b->normed, 0, width, &b->q, 0,
             LFM3_VISION_HEADS * LFM3_VISION_HEAD_DIM);
    multiply(c, &y->k, n, &b->normed, 0, width, &b->k, 0,
             LFM3_VISION_KV_HEADS * LFM3_VISION_HEAD_DIM);
    multiply(c, &y->v, n, &b->normed, 0, width, &b->v, 0,
             LFM3_VISION_KV_HEADS * LFM3_VISION_HEAD_DIM);
    VisionHeadsParams h = {(uint32_t)n, padded, (uint32_t)grid_w,
                           VISION_NORM_EPS};
    gpu_select_kernel(c, K_lfm3_vision_heads);
    gpu_bind_buffer(c, 0, &b->q, 0), gpu_bind_buffer(c, 1, &b->k, 0);
    gpu_bind_buffer(c, 2, &b->v, 0), gpu_bind_buffer(c, 3, &y->q_norm, 0);
    gpu_bind_buffer(c, 4, &y->k_norm, 0);
    gpu_bind_buffer(c, 5, &vision_inv_freq, 0);
    gpu_bind_buffer(c, 6, &b->qh, 0), gpu_bind_buffer(c, 7, &b->kh, 0);
    gpu_bind_buffer(c, 8, &b->vh, 0), gpu_copy_bytes(c, 9, &h, sizeof h);
    gpu_run_kernel(c, VISION_HEAD_GROUPS, (uint32_t)n, 1, VISION_HEAD_THREADS,
                   1, 1);
    VisionAttendParams a = {(uint32_t)n, padded};
    gpu_select_kernel(c, K_lfm3_vision_attend);
    gpu_bind_buffer(c, 0, &b->qh, 0), gpu_bind_buffer(c, 1, &b->kh, 0);
    gpu_bind_buffer(c, 2, &b->vh, 0), gpu_bind_buffer(c, 3, &b->out, 0);
    gpu_copy_bytes(c, 4, &a, sizeof a);
    gpu_run_kernel(c, (uint32_t)(n + ATTEND_ROWS - 1) / ATTEND_ROWS,
                   LFM3_VISION_HEADS, 1, ATTEND_THREADS, 1, 1);
  } else {
    multiply(c, &y->in_proj, n, &b->normed, 0, width, &b->bcx, 0, 3 * width);
    VisionConvParams p = {(uint32_t)width, (uint32_t)grid_w, (uint32_t)grid_h};
    gpu_select_kernel(c, K_lfm3_vision_conv);
    gpu_bind_buffer(c, 0, &b->bcx, 0), gpu_bind_buffer(c, 1, &y->filter, 0);
    gpu_bind_buffer(c, 2, &b->out, 0), gpu_copy_bytes(c, 3, &p, sizeof p);
    gpu_run_kernel(c, width / ROW_THREADS, (uint32_t)n, 1, ROW_THREADS, 1, 1);
  }
  multiply(c, &y->out_proj, n, &b->out, 0, width, &b->projected, 0, width);
  add_patches(c, &b->x, &b->projected, n);
  normalize_patches(c, &b->x, &y->ffn_norm, &b->normed, n);
  multiply(c, &y->w1, n, &b->normed, 0, width, &b->gated, 0,
           LFM3_VISION_HIDDEN);
  multiply(c, &y->w3, n, &b->normed, 0, width, &b->up, 0, LFM3_VISION_HIDDEN);
  gpu_select_kernel(c, K_lfm3_swiglu);
  gpu_bind_buffer(c, 0, &b->gated, 0), gpu_bind_buffer(c, 1, &b->up, 0);
  gpu_bind_buffer(c, 2, &b->gated, 0);
  gpu_run_kernel(c, (uint32_t)n * LFM3_VISION_HIDDEN / ROW_THREADS, 1, 1,
                 ROW_THREADS, 1, 1);
  multiply(c, &y->w2, n, &b->gated, 0, LFM3_VISION_HIDDEN, &b->projected, 0,
           width);
  add_patches(c, &b->x, &b->projected, n);
}

// The projector over the n patches' rows of x, into rows of LFM3_DIM: a
// LayerNorm of each patch, the 4 patches of each square joined into one row,
// the first product with its bias and GELU, then the second with its bias.
static void run_projector(GpuCommandBuffer *c, VisionBuffers *b, int n) {
  uint32_t tokens = (uint32_t)n / (LFM3_MERGE * LFM3_MERGE),
           hidden = LFM3_PROJECTOR_HIDDEN, dim = LFM3_DIM;
  NormParams p = {LFM3_VISION_WIDTH, PROJECTOR_EPS};
  gpu_select_kernel(c, K_lfm3_layer_norm);
  gpu_bind_buffer(c, 0, &b->x, 0), gpu_bind_buffer(c, 1, &projector_norm, 0);
  gpu_bind_buffer(c, 2, &projector_norm_bias, 0);
  gpu_bind_buffer(c, 3, &b->normed, 0), gpu_copy_bytes(c, 4, &p, sizeof p);
  gpu_run_kernel(c, (uint32_t)n, 1, 1, ROW_THREADS, 1, 1);
  // A square's patches are consecutive rows: as rows of JOINED_WIDTH values,
  // the normed patches are the joined squares.
  multiply(c, &projector_in, (int)tokens, &b->normed, 0, JOINED_WIDTH,
           &b->hidden, 0, LFM3_PROJECTOR_HIDDEN);
  gpu_select_kernel(c, K_lfm3_bias_gelu);
  gpu_bind_buffer(c, 0, &b->hidden, 0);
  gpu_bind_buffer(c, 1, &projector_in_bias, 0);
  gpu_copy_bytes(c, 2, &hidden, sizeof hidden);
  gpu_run_kernel(c, tokens * hidden / ROW_THREADS, 1, 1, ROW_THREADS, 1, 1);
  multiply(c, &projector_out, (int)tokens, &b->hidden, 0, LFM3_PROJECTOR_HIDDEN,
           &b->rows, 0, LFM3_DIM);
  gpu_select_kernel(c, K_lfm3_add_bias);
  gpu_bind_buffer(c, 0, &b->rows, 0);
  gpu_bind_buffer(c, 1, &projector_out_bias, 0);
  gpu_copy_bytes(c, 2, &dim, sizeof dim);
  gpu_run_kernel(c, tokens * dim / ROW_THREADS, 1, 1, ROW_THREADS, 1, 1);
}

// Returns f rounded to BF16, to the nearest with ties to even.
static uint16_t round_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Runs the vision tower and the projector on the patches of im with the loaded
// vision weights, and writes the image's rows of LFM3_DIM values as BF16 to
// im->rows, counted in the workspace. Each layer's work is one command buffer,
// so that one long buffer does not hold up the display. Returns 0 when the
// work's buffers do not fit.
static int encode_patches(Media *im) {
  int n = im->vh * im->vw, tokens = im->th * im->tw;
  VisionBuffers b;
  if (!allocate_vision_buffers(&b, n)) return 0;
  uint32_t *pixels = (uint32_t *)b.pixels.ptr;
  for (size_t i = 0; i < (size_t)n * LFM3_PATCH_VALUES; i++)
    pixels[i] = (uint32_t)im->patch[i] << 16;
  GpuCommandBuffer c = gpu_begin(gpu);
  multiply(&c, &patchifier, n, &b.pixels, 0, LFM3_PATCH_VALUES, &b.normed, 0,
           LFM3_VISION_WIDTH);
  normalize_patches(&c, &b.normed, &patch_norm, &b.x, n);
  gpu_commit(&c);
  for (int l = 0; l < LFM3_VISION_LAYERS; l++) {
    c = gpu_begin(gpu);
    run_vision_layer(&c, &vision_layers[l], &b, n, im->vw, im->vh);
    gpu_commit(&c);
  }
  c = gpu_begin(gpu);
  run_projector(&c, &b, n);
  gpu_commit(&c);
  gpu_sync(gpu);
  im->rows =
      memory_host_alloc_required(work_slot, (size_t)tokens * LFM3_DIM * 2);
  im->memory = work_slot;
  const float *rows = (const float *)b.rows.ptr;
  for (size_t i = 0; i < (size_t)tokens * LFM3_DIM; i++)
    im->rows[i] = round_to_bf16(rows[i]);
  free_vision_buffers(&b);
  return 1;
}

// Encodes im once: its rows of LFM3_DIM values, one for each square of
// patches. The vision weights load at the first image, and stay pinned while
// they work. Returns 0 when the model has not loaded, or the weights or the
// work's buffers do not fit.
static int lfm3_encode_image(Media *im) {
  if (im->rows) return 1;
  if (!text_weights) return 0;
  if (!vision_weights) {
    vision_weights =
        weight_group_new_dependent(text_weights, "LFM3-VL vision weights");
    add_vision_weights();
  }
  MemorySlot *slot = weight_group_slot(vision_weights);
  memory_pin(slot, 1);
  int encoded = weight_group_load(vision_weights) && encode_patches(im);
  memory_pin(slot, 0);
  return encoded;
}

// Prepares an image file's bytes for the vision tower (model/lfm3/image.c).
static int lfm3_prepare_image(const char *dir, Media *im, const void *data,
                              size_t len, const char **error) {
  (void)dir;
  return lfm3_image_open(im, data, len, error);
}

// ---- loading, counters and passes ----

static int lfm3_load(Gpu *model_gpu, MemorySlot *work, const char *dir,
                     const Options *o) {
  (void)o;
  if (!checkpoint) {
    gpu = model_gpu, work_slot = work;
    check_config(dir);
#define K(name) K_##name = gpu_find_kernel(gpu, #name)
    K(lfm3_embed), K(lfm3_norm), K(lfm3_norm_add), K(lfm3_copy), K(lfm3_conv),
        K(lfm3_swiglu), K(lfm3_heads), K(lfm3_attend), K(mm_bf16),
        K(lfm3_vision_conv), K(lfm3_vision_heads), K(lfm3_vision_attend),
        K(lfm3_vision_add), K(lfm3_layer_norm), K(lfm3_bias_gelu),
        K(lfm3_add_bias);
#undef K
    checkpoint = tensor_index_open(dir);
    text_weights =
        weight_group_new(gpu, work->account, "LFM3 language model weights");
    add_text_weights();
    required_bytes = weight_group_required_bytes(text_weights) +
                     count_pass_bytes(BLOCK_ROWS, 1);
  }
  MemorySlot *slot = weight_group_slot(text_weights);
  memory_pin(slot, 1);
  if (!memory_wait_for_minimum(slot->account,
                               weight_group_required_bytes(text_weights) +
                                   count_pass_bytes(1, 1)) ||
      !weight_group_load(text_weights)) {
    memory_pin(slot, 0);
    return 0;
  }
  return 1;
}

static MemorySlot *lfm3_weight_slot(void) {
  return text_weights ? weight_group_slot(text_weights) : NULL;
}

static void lfm3_read_counters(ModelCounters *c) {
  *c = (ModelCounters){.prefill_rows = decision_rows,
                       .layers = LFM3_LAYERS,
                       .dense_bytes = weight_group_resident_bytes(text_weights),
                       .required_bytes = required_bytes};
}

static int lfm3_vocab_size(void) { return LFM3_VOCAB; }

static int lfm3_run_idle_work(void) { return 0; }

// A pass's own values: its last command buffer and its logits.
static Pass *lfm3_pass_new(void (*body)(void *), void *argument) {
  const PassVariable variables[] = {{&last_committed, sizeof last_committed},
                                    {&results, sizeof results},
                                    {&results_room, sizeof results_room}};
  Pass *pass = pass_new(body, argument, PASS_STACK_BYTES);
  pass_set_variables(pass, variables, sizeof variables / sizeof *variables);
  return pass;
}

static void lfm3_pass_begin(void) {
  last_committed = 0;
  results = NULL, results_room = 0;
}

static void lfm3_pass_end(void) {
  wait_for_work();
  free(results);
  results = NULL, results_room = 0;
}

static const char *const lfm3_types[] = {"lfm3_vl", 0};

const ModelOps lfm3_ops = {.family = "LFM3 d1",
                           .model_types = lfm3_types,
                           .media_token = "<|image|>",
                           .load = lfm3_load,
                           .dense_weight_slot = lfm3_weight_slot,
                           .read_counters = lfm3_read_counters,
                           .vocab_size = lfm3_vocab_size,
                           .decide = lfm3_decide,
                           .run_idle_work = lfm3_run_idle_work,
                           .pass_new = lfm3_pass_new,
                           .pass_begin = lfm3_pass_begin,
                           .pass_end = lfm3_pass_end,
                           .prepare_media = lfm3_prepare_image,
                           .encode_media = lfm3_encode_image};
