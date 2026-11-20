#include "model/model.h"
#include "base/error.h"
#include "base/file.h"
#include "base/json.h"
#include "engine/weights.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const ModelOps *const families[] = {&deepseek_ops, &lfm3_ops,
                                           &minimax_h3_image_ops};

const ModelOps *model_find_family(const char *dir) {
  char path[1024];
  snprintf(path, sizeof path, "%s/config.json", dir);
  const char *key = "model_type";
  if (access(path, R_OK) != 0) {
    snprintf(path, sizeof path, "%s/model_index.json", dir);
    if (access(path, R_OK) == 0)
      key = "_class_name";
    else
      snprintf(path, sizeof path, "%s/config.json", dir);
  }
  char *text = read_file(path, 0);
  Json *root = json_parse(text);
  const Json *type = json_get(root, key);
  if (!type || type->type != J_STR) die("%s: no %s", path, key);
  for (size_t f = 0; f < sizeof families / sizeof *families; f++)
    for (const char *const *t = families[f]->model_types; *t; t++)
      if (!strcmp(*t, type->s)) {
        json_free(root);
        free(text);
        return families[f];
      }
  die("%s: %s %s is not one this engine runs", path, key, type->s);
}

Model *model_create(Memory *memory, Gpu *gpu, const char *dir, uint64_t limit) {
  Model *m = xmalloc(sizeof *m);
  *m = (Model){.ops = model_find_family(dir), .dir = dir, .gpu = gpu};
  memory_account_init(memory, &m->memory, dir, limit);
  memory_slot_init(&m->work, &m->memory, MEMORY_MODEL, NULL, "model workspace",
                   NULL, NULL);
  m->work.automatic = 1;
  return m;
}

int model_load(Model *m, const Options *o) {
  weight_group_set_stop_on_quit(1);  // quitting stops a model's load
  int loaded = m->ops->load(m->gpu, &m->work, m->dir, o);
  weight_group_set_stop_on_quit(0);
  return loaded;
}

void model_set_request_running(Model *m, int active) {
  if (m->ops->dense_weight_slot && m->ops->dense_weight_slot())
    memory_pin(m->ops->dense_weight_slot(), active);
  if (m->ops->set_request_running) m->ops->set_request_running(active);
  if (!active) memory_set_minimum(&m->memory, 0, MEMORY_REQUIRED);
}

Model *model_open(Memory *memory, Gpu *gpu, const char *dir, const Options *o) {
  Model *m = model_create(memory, gpu, dir, 0);
  if (!model_load(m, o)) return NULL;
  model_set_request_running(m, 1);
  return m;
}
