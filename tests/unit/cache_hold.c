#include "check.h"
#include "residency_stand_ins.h"
#include "engine/expert_store.c"
#include "store_layout.h"
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

// Tests holds: a held expert is no victim while passes hold it, a layer's
// hold keeps all its experts, and a resize leaves held slots. A pass that
// finds every slot held ends the program when it runs alone, and waits for a
// slot while other passes run.

static int needed = 9, requested;

static void request_needed(void *argument) {
  (void)argument;
  expert_store_start_reads(st, 0, &needed, 1);
  requested = 1;
}

int main(void) {
  st = store_create();
  Slot fixture[4] = {0};
  double priorities[4] = {1, 2, 3, 4};
  int free_slots[4];
  st->slots = fixture, st->score = priorities,
  st->max_slots = st->cap = st->nslots = 4, st->free_list = free_slots;
  memset(st->where, 0xff, WHERE_BYTES);
  st->head = 3, st->tail = 0, st->pass_layer = st->serving = -1;
  st->token = 10, st->weight = 1;
  for (int i = 0; i < 4; i++) {
    fixture[i].layer = 0, fixture[i].expert = i, fixture[i].mapped = 1;
    fixture[i].prev = i == 3 ? -1 : i + 1, fixture[i].next = i ? i - 1 : -1;
    WHERE(0, i) = i, st->escore[i] = priorities[i];
  }
  int ids[] = {0, 1}, second[] = {1, 2};
  check(expert_store_hold(st, 0, ids, 2) == 0,
        "holding resident experts starts no read");
  st->serving = 3;
  check(victim(st) == 2, "the lowest unheld priority is the victim");
  check(expert_store_hold(st, 0, second, 2) == 0,
        "a second hold of a held expert starts no read");
  st->serving = 3;
  check(fixture[1].holds == 2 && victim(st) == 3, "holds of one expert add up");
  expert_store_release(st, 0, ids, 2);
  check(fixture[0].holds == 0 && fixture[1].holds == 1 && victim(st) == 0,
        "an expert stays held until its last hold goes");
  check(WHERE(0, 0) == 0 && WHERE(0, 1) == 1 && WHERE(0, 2) == 2,
        "holds move no expert");
  expert_store_release(st, 0, second, 2);
  check(fixture[1].holds == 0 && fixture[2].holds == 0 &&
            st->held_references == 0,
        "every hold is released");
  check(st->stats.misses == 0 && st->stats.evictions == 0,
        "holds of resident experts count no miss or eviction");

  fixture[3].loading = threads_group_new();
  threads_group_enter(fixture[3].loading);
  int pending[] = {3};
  check(expert_store_hold(st, 0, pending, 1) == 1 &&
            expert_store_hold(st, 0, pending, 1) == 1,
        "holds of an expert being read report its read");
  check(fixture[3].holds == 2 && WHERE(0, 3) == 3 && st->stats.misses == 0,
        "two holds share the read");
  expert_store_release(st, 0, pending, 1);
  check(fixture[3].holds == 1 && !expert_store_is_ready(st, 0, 3),
        "the expert is not ready while it is read");
  threads_group_leave(fixture[3].loading);
  expert_store_wait_for_reads(st, 0, pending, 1);
  check(expert_store_is_ready(st, 0, 3) && fixture[3].holds == 1,
        "the read finishes while the expert is held");
  expert_store_release(st, 0, pending, 1);
  check(fixture[3].holds == 0 && st->held_references == 0,
        "the last hold is released");

  expert_store_hold_layer(st, 0), expert_store_hold_layer(st, 0);
  st->serving = 3;  // no slot counts as being made resident now
  int no_victim = victim(st) < 0;
  expert_store_release_layer(st, 0);
  no_victim &= victim(st) < 0;
  expert_store_release_layer(st, 0);
  check(no_victim && victim(st) >= 0,
        "a layer's holds keep all its experts until the last one goes");

  int oldest[] = {0, 1};
  check(expert_store_hold(st, 0, oldest, 2) == 0,
        "holding the two oldest experts starts no read");
  st->cap = 2;
  expert_store_resize_in_prompt(st, 2);
  check(st->nslots == 2 && fixture[0].holds == 1 && fixture[1].holds == 1 &&
            st->held_references == 2 && WHERE(0, 0) == 0 && WHERE(0, 1) == 1 &&
            WHERE(0, 2) < 0 && WHERE(0, 3) < 0,
        "a resize gives back the unheld slots and leaves the held ones, "
        "whatever their age");
  expert_store_release(st, 0, oldest, 2);

  int zero = open("/dev/zero", O_RDONLY);
  check(zero >= 0, "/dev/zero opens");
  for (int i = 0; i < 4; i++)
    fixture[i].b.ptr = xmalloc(65536), fixture[i].b.size = 65536;
  expert_spans[0] = 1;
  expert_span[0][needed][0] = (FileSpan){zero, 0, 4096};
  for (int i = 0; i < 4; i++)
    WHERE(0, i) = i, fixture[i].layer = 0, fixture[i].expert = i;
  st->nslots = st->cap = 4, st->nfree = 0;
  int all[] = {0, 1, 2, 3};
  check(expert_store_hold(st, 0, all, 4) == 0, "every slot is held");
  fflush(stdout);
  pid_t child = fork();
  if (!child) {
    pass_resume(pass_new(request_needed, NULL, 1 << 20));
    _exit(0);
  }
  int status = 0;
  check(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
            WEXITSTATUS(status) == 1,
        "a pass alone that needs a slot while every slot is held ends the "
        "program");
  // The test stands in for another pass that holds the slots.
  expert_store_set_other_passes(st, 1);
  Pass *waiting = pass_new(request_needed, NULL, 1 << 20);
  int yielded = !pass_resume(waiting);
  check(yielded && !requested && !pass_ready(waiting),
        "a pass that needs a slot while every slot is held waits");
  expert_store_release(st, 0, all + 3, 1);
  check(pass_ready(waiting), "the pass can continue once a hold goes");
  check(pass_resume(waiting) && requested && WHERE(0, needed) == 3 &&
            fixture[3].expert == needed,
        "the released slot holds the pass's expert");
  expert_store_wait_for_reads(st, 0, &needed, 1);
  expert_store_release(st, 0, all, 3);
  pass_free(waiting);
  close(zero);
  return failures != 0;
}
