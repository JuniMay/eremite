#ifndef EREMITE_BASE_JSON_H
#define EREMITE_BASE_JSON_H

// JSON documents, parsed into trees.

typedef struct Json Json;
struct Json {
  enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } type;
  int n;  // children of arrays and objects, bytes of strings
  const char *key;  // set on object members
  const char *s;
  double num;
  Json *kid;
};

// Parses text into a tree that owns its strings; dies on invalid JSON.
Json *json_parse(char *text);

// Returns the object's member named key, or null.
Json *json_get(const Json *object, const char *key);

// Frees the tree and its strings.
void json_free(Json *root);

#endif
