#include "model/lfm3/model.h"
#include "base/error.h"
#include "base/file.h"
#include "base/json.h"
#include "engine/memory.h"
#include "engine/safetensors.h"
#include "engine/weights.h"
#include "gpu/gpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// LFM3's language model in a d1 checkpoint (model.language_model.*): its
// shape, its layers and their weights. A d1 checkpoint's config.json has
// model_type lfm3_vl and d1_attention "set". The vision tower's weights do
// not load with the language model.

// The RoPE bases (rope_theta) of the global and the sliding layers.
static constexpr double GLOBAL_THETA = 1e6, SLIDING_THETA = 1e4;
// The value pairs that RoPE turns in each head: half of them on the global
// layers (partial_rotary_factor 0.5), all of them on the sliding ones.
static constexpr int GLOBAL_ROTATED = LFM3_HEAD_DIM / 4,
                     SLIDING_ROTATED = LFM3_HEAD_DIM / 2;

// The GPU that the model runs on.
static Gpu *gpu;
static TensorIndex *checkpoint;
static WeightGroup *text_weights;

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
// shape, its layers' kinds, RoPE, the norms and the decision layout. Ends the
// program if not.
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
               {sliding, "rope_theta", SLIDING_THETA}};

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

// Adds the BF16 matrix named model.language_model.name.weight to the weights.
static void add_matrix(Lin *line, const char *name) {
  Tensor tensor =
      tensor_index_get(checkpoint, "model.language_model.%s.weight", name);
  if (tensor.dtype != 'b' || tensor.ndim != 2)
    die("%s: not a BF16 matrix", name);
  *line = (Lin){.n = (int)tensor.shape[0], .k = (int)tensor.shape[1]};
  weight_group_add_tensor(text_weights, &line->w, tensor, 1);
}

// Adds the vector named model.language_model.name.weight to the weights, as
// FP32.
static void add_vector(Buf *target, const char *name) {
  weight_group_add_vector(
      text_weights, target,
      tensor_index_get(checkpoint, "model.language_model.%s.weight", name), 0,
      0);
}

// The base of RoPE's inverse frequencies.
typedef struct {
  double theta;
} Frequencies;

// Writes RoPE's inverse frequencies for heads of LFM3_HEAD_DIM values, as the
// reference computes them in FP32.
static void fill_inv_freq(Buf *buffer, const void *parameters) {
  const Frequencies *f = parameters;
  float *out = (float *)buffer->ptr;
  for (int i = 0; i < LFM3_HEAD_DIM / 2; i++)
    out[i] =
        1.0f / powf((float)f->theta, (float)(2 * i) / (float)LFM3_HEAD_DIM);
}

// Adds every weight of the language model to the weight group.
static void add_text_weights(void) {
  char name[128];
  for (int l = 0; l < LFM3_LAYERS; l++) {
    Layer *y = &layers[l];
#define NAME(part) (snprintf(name, sizeof name, "layers.%d.%s", l, part), name)
    add_vector(&y->operator_norm, NAME("operator_norm"));
    add_vector(&y->post_operator_norm, NAME("post_operator_norm"));
    add_vector(&y->ffn_norm, NAME("ffn_norm"));
    add_vector(&y->post_ffn_norm, NAME("post_ffn_norm"));
    if (y->kind == LAYER_CONV) {
      add_matrix(&y->in_proj, NAME("conv.in_proj"));
      add_matrix(&y->out_proj, NAME("conv.out_proj"));
      add_vector(&y->conv, NAME("conv.conv"));
    } else {
      add_matrix(&y->q, NAME("self_attn.q_proj"));
      add_matrix(&y->k, NAME("self_attn.k_proj"));
      add_matrix(&y->v, NAME("self_attn.v_proj"));
      add_matrix(&y->gate, NAME("self_attn.gate_proj"));
      add_matrix(&y->o, NAME("self_attn.o_proj"));
      add_vector(&y->q_norm, NAME("self_attn.q_norm"));
      add_vector(&y->k_norm, NAME("self_attn.k_norm"));
    }
    add_matrix(&y->w1, NAME("feed_forward.w1"));
    add_matrix(&y->w3, NAME("feed_forward.w3"));
    add_matrix(&y->w2, NAME("feed_forward.w2"));
#undef NAME
  }
  Tensor table =
      tensor_index_get(checkpoint, "model.language_model.embed_tokens.weight");
  if (table.dtype != 'b' || table.shape[0] != LFM3_VOCAB ||
      table.shape[1] != LFM3_DIM)
    die("the embedding table is not %d BF16 rows of %d", LFM3_VOCAB, LFM3_DIM);
  weight_group_add_tensor(text_weights, &embed, table, 1);
  add_vector(&final_norm, "embedding_norm");
  const size_t frequency_bytes = LFM3_HEAD_DIM / 2 * sizeof(float);
  weight_group_add_generated(text_weights, &global_inv_freq, frequency_bytes,
                             fill_inv_freq, &(Frequencies){GLOBAL_THETA},
                             sizeof(Frequencies));
  weight_group_add_generated(text_weights, &sliding_inv_freq, frequency_bytes,
                             fill_inv_freq, &(Frequencies){SLIDING_THETA},
                             sizeof(Frequencies));
}

// ---- loading and counters ----

static int lfm3_load(Gpu *model_gpu, MemorySlot *work, const char *dir,
                     const Options *o) {
  (void)o;
  if (!checkpoint) {
    gpu = model_gpu;
    check_config(dir);
    checkpoint = tensor_index_open(dir);
    text_weights =
        weight_group_new(gpu, work->account, "LFM3 language model weights");
    add_text_weights();
  }
  MemorySlot *slot = weight_group_slot(text_weights);
  memory_pin(slot, 1);
  if (!memory_wait_for_minimum(slot->account,
                               weight_group_required_bytes(text_weights)) ||
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
  *c =
      (ModelCounters){.layers = LFM3_LAYERS,
                      .dense_bytes = weight_group_resident_bytes(text_weights)};
}

static int lfm3_vocab_size(void) { return LFM3_VOCAB; }

static int lfm3_run_idle_work(void) { return 0; }

static const char *const lfm3_types[] = {"lfm3_vl", 0};

const ModelOps lfm3_ops = {.family = "LFM3 d1",
                           .model_types = lfm3_types,
                           .load = lfm3_load,
                           .dense_weight_slot = lfm3_weight_slot,
                           .read_counters = lfm3_read_counters,
                           .vocab_size = lfm3_vocab_size,
                           .run_idle_work = lfm3_run_idle_work};
