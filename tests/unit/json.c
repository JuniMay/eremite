#include "base/json.h"
#include "check.h"
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests that a document becomes a tree of typed nodes, that members are found
// by name, and that invalid JSON is fatal.

int main(void) {
  char text[] = "{\"layers\": 40, \"experts\": [true, null, \"iq1_m\"], "
                "\"rope\": {\"theta\": 2.5}}";
  Json *root = json_parse(text);
  check(root->type == J_OBJ && root->n == 3,
        "the root is an object of 3 members");
  Json *layers = json_get(root, "layers");
  check(layers && layers->type == J_NUM && layers->num == 40,
        "a number member");
  Json *experts = json_get(root, "experts");
  check(experts && experts->type == J_ARR && experts->n == 3 &&
            experts->kid[0].type == J_BOOL && experts->kid[0].num == 1 &&
            experts->kid[1].type == J_NULL,
        "an array of a boolean and a null");
  check(experts->kid[2].type == J_STR && experts->kid[2].n == 5 &&
            !strcmp(experts->kid[2].s, "iq1_m"),
        "a string with its length");
  Json *theta = json_get(json_get(root, "rope"), "theta");
  check(theta && theta->num == 2.5, "a member of a nested object");
  check(!json_get(root, "missing") && !json_get(layers, "layers"),
        "no member for a missing name or a non-object");
  json_free(root);

  fflush(stdout);
  pid_t pid = fork();
  if (!pid) {
    close(2);
    char bad[] = "{\"layers\": ";
    json_parse(bad);
    _exit(0);
  }
  int status;
  waitpid(pid, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 1, "invalid JSON is fatal");
  return failures != 0;
}
