#include "base/json.h"
#include "base/error.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <yyjson.h>

// The tree's root, with the yyjson document whose strings its nodes point to.
typedef struct {
  Json root;
  yyjson_doc *source;
} JsonDocument;

static void json_value(Json *out, yyjson_val *value) {
  if (yyjson_is_obj(value) || yyjson_is_arr(value)) {
    int object = yyjson_is_obj(value);
    size_t count = object ? yyjson_obj_size(value) : yyjson_arr_size(value);
    if (count > INT_MAX) die("JSON container is too large");
    out->type = object ? J_OBJ : J_ARR;
    out->n = (int)count;
    out->kid = xmalloc(count * sizeof *out->kid);
    size_t i, maximum;
    yyjson_val *key, *child;
    if (object) {
      yyjson_obj_foreach(value, i, maximum, key, child) {
        json_value(&out->kid[i], child);
        out->kid[i].key = yyjson_get_str(key);
      }
    } else {
      yyjson_arr_foreach(value, i, maximum, child) json_value(&out->kid[i], child);
    }
  } else if (yyjson_is_str(value)) {
    size_t length = yyjson_get_len(value);
    if (length > INT_MAX) die("JSON string is too large");
    out->type = J_STR, out->n = (int)length, out->s = yyjson_get_str(value);
  } else if (yyjson_is_bool(value)) {
    out->type = J_BOOL, out->num = yyjson_get_bool(value);
  } else if (yyjson_is_num(value)) {
    out->type = J_NUM, out->num = yyjson_get_num(value);
  }
}

Json *json_parse(char *text) {
  yyjson_read_err error;
  yyjson_doc *source = yyjson_read_opts(text, strlen(text), 0, NULL, &error);
  if (!source) die("invalid JSON at byte %zu: %s", error.pos, error.msg);
  JsonDocument *doc = xmalloc(sizeof *doc);
  doc->source = source;
  json_value(&doc->root, yyjson_doc_get_root(source));
  return &doc->root;
}

Json *json_get(const Json *object, const char *key) {
  if (!object || object->type != J_OBJ) return NULL;
  for (int i = 0; i < object->n; i++)
    if (!strcmp(object->kid[i].key, key)) return &object->kid[i];
  return NULL;
}

static void json_children_free(Json *value) {
  if (value->type == J_OBJ || value->type == J_ARR)
    for (int i = 0; i < value->n; i++) json_children_free(&value->kid[i]);
  free(value->kid);
}

void json_free(Json *root) {
  if (!root) return;
  JsonDocument *doc = (JsonDocument *)root;
  json_children_free(root);
  yyjson_doc_free(doc->source);
  free(doc);
}
