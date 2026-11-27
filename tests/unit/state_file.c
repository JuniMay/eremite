#include "check.h"
#include "base/shutdown.h"
#include "engine/kv_pages.h"
#include "engine/state_file.h"
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

// Tests saved states' files without a model: a fake family keeps its rows
// and snapshot parts in plain memory. A saved state's parts must give back,
// for every prefix, the rows a state_copy of the live state gives, also
// across appends and a part that starts earlier (a rewind); damaged, short or
// differently shaped files must be refused; a snapshot, and a live state's
// end, must come back byte for byte; a second stop request must end a write;
// and a checkpoint's identity must follow its weights and nothing else.
// CRC32C must give its standard check value, also in pieces. The temporary
// folder is removed before exit.

static char directory[4096];

static void remove_tree(const char *path) {
  DIR *d = opendir(path);
  if (d) {
    for (struct dirent *e; (e = readdir(d));) {
      if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
      char sub[8192];
      snprintf(sub, sizeof sub, "%s/%s", path, e->d_name);
      struct stat st;
      if (!lstat(sub, &st) && S_ISDIR(st.st_mode))
        remove_tree(sub);
      else
        unlink(sub);
    }
    closedir(d);
  }
  rmdir(path);
}

static void cleanup(void) {
  if (*directory) remove_tree(directory);
}

// ---- a fake family ----

// Four regions like the real ones: plain rows (24 bytes), compressed pairs
// with a guard row (1 KiB, more than one 4 MiB block at these lengths), one
// index row per four positions (512 bytes), and rows a position late (64
// bytes; Qwen's MTP keys).
enum { REGIONS = 4, ROLL = 100000, VOCAB = 250000 };

static const uint32_t UNIT[REGIONS] = {24, 1024, 512, 64},
                      RATIO[REGIONS] = {1, 2, 4, 1},
                      GUARD[REGIONS] = {0, 1, 0, 1},
                      LAG[REGIONS] = {0, 0, 0, 1};
static uint32_t unit_of[REGIONS] = {
    24, 1024, 512, 64};  // the shape a "build" lists; one test changes it

// Region i's shape, its rows nowhere.
static StateRegion shape(int i) {
  return (StateRegion){NULL, UNIT[i], RATIO[i], GUARD[i], LAG[i], 0, 0};
}

typedef struct {
  uint8_t *rows[REGIONS];
  int cap, pos, generation;
  uint8_t roll[ROLL];
} Fake;

typedef struct {
  int pos;
  uint8_t roll[ROLL];
  float *logits;
} FakeSnap;

static size_t region_bytes(int i, int cap) {
  return (size_t)(cap / RATIO[i] + GUARD[i] + 1) * UNIT[i];
}

static void reserve(Fake *s, int positions) {
  if (positions < 1)
    positions = 1;  // a new state has room, as the real ones do
  if (positions <= s->cap) return;
  for (int i = 0; i < REGIONS; i++) {
    size_t old = s->cap ? region_bytes(i, s->cap) : 0,
           now = region_bytes(i, positions);
    s->rows[i] = realloc(s->rows[i], now);
    memset(s->rows[i] + old, 0xEE, now - old);
  }
  s->cap = positions;
}

static int fake_regions(ModelState *state, int keep, int positions, int fill,
                        StateRegion *out) {
  (void)keep;
  Fake *s = (Fake *)state;
  if (fill) reserve(s, positions), s->pos = positions;
  for (int i = 0; i < REGIONS; i++)
    out[i] =
        (StateRegion){s->rows[i], unit_of[i], RATIO[i], GUARD[i], LAG[i], 0, 0};
  return REGIONS;
}

static int fake_parts(ModelSnapshot *snap, ModelState *state, int pos,
                      SnapshotPart *out, int32_t *meta, int *nmeta) {
  FakeSnap *x = (FakeSnap *)snap;
  Fake *s = (Fake *)state;
  meta[0] = x ? x->pos : pos, *nmeta = 1;
  int n = 0;
  out[n++] = (SnapshotPart){x ? x->roll : s->roll, ROLL};
  if (x && x->logits) out[n++] = (SnapshotPart){x->logits, VOCAB * 4};
  return n;
}

static ModelSnapshot *fake_alloc(const int32_t *meta, int nmeta,
                                 int with_logits) {
  if (nmeta != 1) return NULL;
  FakeSnap *x = calloc(1, sizeof *x);
  x->pos = meta[0];
  if (with_logits) x->logits = calloc(VOCAB, 4);
  return (ModelSnapshot *)x;
}

static void fake_free(ModelSnapshot *snap) {
  FakeSnap *x = (FakeSnap *)snap;
  if (x) free(x->logits), free(x);
}

static const float *fake_logits(const ModelSnapshot *snap) {
  return ((const FakeSnap *)snap)->logits;
}

static const ModelOps fake = {.family = "fake",
                              .snapshot_free = fake_free,
                              .snapshot_logits = fake_logits,
                              .state_version = 1,
                              .write_state_regions = fake_regions,
                              .write_snapshot_parts = fake_parts,
                              .snapshot_new_empty = fake_alloc};

static uint8_t byte_of(int region, int64_t row, int generation, size_t k) {
  uint64_t x = (uint64_t)region * 0x9e3779b97f4a7c15ull ^
               (uint64_t)row * 0xbf58476d1ce4e5b9ull ^
               (uint64_t)generation * 0x94d049bb133111ebull ^
               k * 0x2545f4914f6cdd1dull;
  x ^= x >> 31, x *= 0xd6e8feb86659fd93ull, x ^= x >> 32;
  return (uint8_t)x;
}

// The live state runs positions from .. to - 1: every row from those that
// `from` positions leave unchanged is written again, as a prefill that
// continues there writes it (a guard row, and rows after a rewind).
static void run(Fake *s, int from, int to) {
  reserve(s, to);
  s->generation++;
  for (int i = 0; i < REGIONS; i++) {
    StateRegion r0 = shape(i);
    for (int64_t r = region_stable(&r0, from); r < region_rows(&r0, to); r++)
      for (size_t k = 0; k < UNIT[i]; k++)
        s->rows[i][r * UNIT[i] + k] = byte_of(i, r, s->generation, k);
  }
  for (int k = 0; k < ROLL; k++)
    s->roll[k] = byte_of(9, to, s->generation, (size_t)k);
  s->pos = to;
}

// Whether d holds s's rows before `positions`, as a state_copy gives them.
static int same_rows(const Fake *d, const Fake *s, int positions) {
  for (int i = 0; i < REGIONS; i++) {
    StateRegion r0 = shape(i);
    if (memcmp(d->rows[i], s->rows[i],
               (size_t)region_rows(&r0, positions) * UNIT[i]))
      return 0;
  }
  return 1;
}

static void fake_release(Fake *s) {
  for (int i = 0; i < REGIONS; i++) free(s->rows[i]);
  free(s);
}

static int open_new(const char *name) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, name);
  return open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
}

// Reads the first `count` of the files' rows prefix into a new state and
// compares it with s's.
static int prefix_ok(const int *fds, int count, int positions, const Fake *s) {
  Fake *d = calloc(1, sizeof *d);
  int ok = !state_file_read(&fake, (ModelState *)d, positions, fds, count) &&
           d->pos == positions && same_rows(d, s, positions);
  fake_release(d);
  return ok;
}

// Whether a prefix reads at all (its rows not compared).
static int reads(const int *fds, int count, int positions) {
  Fake *d = calloc(1, sizeof *d);
  int ok = !state_file_read(&fake, (ModelState *)d, positions, fds, count);
  fake_release(d);
  return ok;
}

static void flip(int fd, off_t at) {
  uint8_t b;
  if (pread(fd, &b, 1, at) == 1) {
    b ^= 0x40;
    if (pwrite(fd, &b, 1, at) != 1) perror("pwrite");
  }
}

// ---- identity ----

static void write_file(const char *name, const char *text, size_t size) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, name);
  FILE *f = fopen(path, "wb");
  if (!f) {
    perror(path);
    exit(1);
  }
  for (size_t i = 0; i < size; i++) fputc(text[i % strlen(text)], f);
  fclose(f);
}

static void folder_identity(const char *folder, uint8_t out[32]) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, folder);
  CC_SHA256_CTX h;
  CC_SHA256_Init(&h);
  identity_add_folder(&h, path);
  CC_SHA256_Final(out, &h);
}

static void set_identity(const char *first, uint8_t out[32]) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, first);
  CC_SHA256_CTX h;
  CC_SHA256_Init(&h);
  identity_add_gguf_set(&h, path);
  CC_SHA256_Final(out, &h);
}

static void identity_checks(void) {
  char path[8192];
  snprintf(path, sizeof path, "%s/model", directory), mkdir(path, 0700);
  snprintf(path, sizeof path, "%s/model/sub", directory), mkdir(path, 0700);
  snprintf(path, sizeof path, "%s/model/.cache", directory), mkdir(path, 0700);
  write_file("model/config.json", "{\"a\": 1}", 8);
  write_file("model/a.safetensors", "weights-a", 300000);
  write_file("model/sub/b.safetensors", "weights-b", 1000);
  write_file("model/README.md", "read me", 7);
  write_file("model/.cache/c.safetensors", "hidden", 6);
  uint8_t base[32], again[32];
  folder_identity("model", base);
  write_file("model/README.md", "read me again", 13);
  write_file("model/.cache/c.safetensors", "hidden, changed", 15);
  write_file("model/.DS_Store", "finder", 6);
  folder_identity("model", again);
  check(!memcmp(base, again, 32),
        "documents and dot files leave the identity as it is");
  write_file("model/a.safetensors", "weights-A",
             300000);  // the first 64 KiB differ
  folder_identity("model", again);
  check(memcmp(base, again, 32),
        "a weights file's first bytes change the identity");
  write_file("model/a.safetensors", "weights-a", 300000);
  folder_identity("model", again);
  check(!memcmp(base, again, 32),
        "the same weights give the same identity again");
  write_file("model/sub/b.safetensors", "weights-b", 1001);  // its size
  folder_identity("model", again);
  check(memcmp(base, again, 32), "a weights file's size changes the identity");
  write_file("model/sub/b.safetensors", "weights-b", 1000);
  write_file("model/config.json", "{\"a\": 2}", 8);
  folder_identity("model", again);
  check(memcmp(base, again, 32), "config.json changes the identity");
  write_file("x-00001-of-00002.gguf", "first", 5000);
  write_file("x-00002-of-00002.gguf", "second", 5000);
  set_identity("x-00001-of-00002.gguf", base);
  write_file("x-00002-of-00002.gguf", "SECOND", 5000);
  set_identity("x-00001-of-00002.gguf", again);
  check(memcmp(base, again, 32), "every file of a GGUF set counts");
}

// ---- paged regions ----

enum { PAGED_SLICES = 2, PAGED_ROWS = 256, PAGED_UNIT = 1024 };

typedef struct {
  void **pages;
  int npages;
} PagedState;

static void paged_reserve(PagedState *s, int positions) {
  int need = positions > 0 ? (positions + PAGED_ROWS - 1) / PAGED_ROWS : 0;
  if (need <= s->npages) return;
  s->pages = realloc(s->pages, (size_t)need * sizeof *s->pages);
  size_t page_bytes = (size_t)PAGED_SLICES * PAGED_ROWS * PAGED_UNIT;
  for (int p = s->npages; p < need; p++) {
    s->pages[p] = malloc(page_bytes);
    memset(s->pages[p], 0xCC, page_bytes);
  }
  s->npages = need;
}

static void paged_free(PagedState *s) {
  for (int p = 0; p < s->npages; p++) free(s->pages[p]);
  free(s->pages);
  free(s);
}

static int paged_regions(ModelState *state, int keep, int positions, int fill,
                         StateRegion *out) {
  (void)keep;
  PagedState *s = (PagedState *)state;
  if (fill) paged_reserve(s, positions);
  size_t slice_bytes = (size_t)PAGED_ROWS * PAGED_UNIT;
  for (int i = 0; i < PAGED_SLICES; i++) {
    out[i] = (StateRegion){
        .data = s->pages,
        .unit = PAGED_UNIT,
        .ratio = 1,
        .guard = 0,
        .lag = 0,
        .page_rows = PAGED_ROWS,
        .page_offset = (size_t)i * slice_bytes,
    };
  }
  return PAGED_SLICES;
}

static const ModelOps paged_ops = {.family = "paged",
                                   .state_version = 1,
                                   .write_state_regions = paged_regions};

static uint8_t *paged_row(const PagedState *s, int slice, int pos) {
  int p = pos / PAGED_ROWS, in_p = pos % PAGED_ROWS;
  return (uint8_t *)s->pages[p] + (size_t)slice * PAGED_ROWS * PAGED_UNIT +
         (size_t)in_p * PAGED_UNIT;
}

static void paged_checks(void) {
  // 5000 rows x 1024 B = 5.12 MB per slice: crosses 4 MiB block and 256-row
  // page boundaries across two parts with an overlapping branch/rewind.
  PagedState *src = calloc(1, sizeof *src);
  paged_reserve(src, 5000);
  for (int pos = 0; pos < 5000; pos++) {
    for (int sl = 0; sl < PAGED_SLICES; sl++) {
      uint8_t *dst = paged_row(src, sl, pos);
      for (int k = 0; k < PAGED_UNIT; k++)
        dst[k] = byte_of(sl + 10, pos, 1, (size_t)k);
    }
  }
  int pfd[2] = {open_new("paged-0"), open_new("paged-1")};
  check(state_file_write(&paged_ops, (ModelState *)src, 0, 5000, pfd[0]) > 0,
        "paged state writes first part spanning 4 MiB blocks");
  // Overwrite suffix from 3000..4600 (mid-page 11 through page 17) as branch.
  for (int pos = 3000; pos < 4600; pos++) {
    for (int sl = 0; sl < PAGED_SLICES; sl++) {
      uint8_t *dst = paged_row(src, sl, pos);
      for (int k = 0; k < PAGED_UNIT; k++)
        dst[k] = byte_of(sl + 10, pos, 2, (size_t)k);
    }
  }
  check(state_file_write(&paged_ops, (ModelState *)src, 3000, 4600, pfd[1]) > 0,
        "paged state writes overlapping branch suffix part");

  PagedState *got = calloc(1, sizeof *got);
  int read_ok = !state_file_read(&paged_ops, (ModelState *)got, 4600, pfd, 2);
  int match = read_ok;
  for (int pos = 0; match && pos < 4600; pos++) {
    for (int sl = 0; match && sl < PAGED_SLICES; sl++) {
      if (memcmp(paged_row(got, sl, pos), paged_row(src, sl, pos), PAGED_UNIT))
        match = 0;
    }
  }
  check(match, "paged state reads back byte for byte across parts and pages");
  close(pfd[0]), close(pfd[1]);
  paged_free(got);
  paged_free(src);
}

// ---- copy-on-write paged load sharing ----

static Buf cow_test_alloc(void *ctx, MemorySlot *slot, size_t bytes) {
  (void)ctx;
  (void)slot;
  void *ptr = malloc(bytes ? bytes : 1);
  if (!ptr) return (Buf){0};
  memset(ptr, 0xA5, bytes);
  return (Buf){
      .ptr = ptr, .size = bytes, .allocated = bytes, .address = (uint64_t)ptr};
}

static void cow_test_release(void *ctx, Buf *buf) {
  (void)ctx;
  free(buf->ptr);
  *buf = (Buf){0};
}

static size_t cow_test_round(size_t bytes) { return bytes; }

static int cow_regions(ModelState *state, int keep, int positions, int fill,
                       StateRegion *out) {
  KvPageTable *t = (KvPageTable *)state;
  if (fill) {
    if (keep == 0) {
      if (!kv_page_table_reset_writable(t, positions)) return 0;
    } else if (positions > keep) {
      if (!kv_page_table_prepare_write(t, keep, positions - keep)) return 0;
    }
  }
  size_t slice_bytes = t->layout.slice_bytes;
  for (int i = 0; i < PAGED_SLICES; i++) {
    out[i] = (StateRegion){
        .data = t->host_ptrs,
        .unit = PAGED_UNIT,
        .ratio = 1,
        .guard = 0,
        .lag = 0,
        .page_rows = PAGED_ROWS,
        .page_offset = (size_t)i * slice_bytes,
    };
  }
  return PAGED_SLICES;
}

static const ModelOps cow_ops = {
    .family = "cow", .state_version = 1, .write_state_regions = cow_regions};

static uint8_t *cow_row(const KvPageTable *t, int slice, int pos) {
  int p = pos / PAGED_ROWS, in_p = pos % PAGED_ROWS;
  return (uint8_t *)t->host_ptrs[p] + (size_t)slice * t->layout.slice_bytes +
         (size_t)in_p * PAGED_UNIT;
}

static void cow_shared_load_checks(void) {
  KvPageBackend backend = {.alloc = cow_test_alloc,
                           .release = cow_test_release,
                           .round_bytes = cow_test_round};
  StateGrowthPolicy tg = {
      .start_cap = 8, .factor = 2, .max_step = 16, .align = 8, .max_cap = 256};
  KvPageLayout layout = kv_page_layout_init(
      PAGED_ROWS, (size_t)PAGED_ROWS * PAGED_UNIT, PAGED_SLICES, tg);

  KvPageTable parent, branch, loaded, loaded_mid, loaded_exact;
  kv_page_table_init(&parent, backend, &layout);
  kv_page_table_init(&branch, backend, &layout);
  kv_page_table_init(&loaded, backend, &layout);
  kv_page_table_init(&loaded_mid, backend, &layout);
  kv_page_table_init(&loaded_exact, backend, &layout);

  check(kv_page_table_prepare_write(&parent, 0, 5000),
        "CoW parent allocates 5000 rows");
  for (int pos = 0; pos < 5000; pos++) {
    for (int sl = 0; sl < PAGED_SLICES; sl++) {
      uint8_t *dst = cow_row(&parent, sl, pos);
      for (int k = 0; k < PAGED_UNIT; k++)
        dst[k] = byte_of(sl + 20, pos, 1, (size_t)k);
    }
  }
  int pfd[2] = {open_new("cow-0"), open_new("cow-1")};
  check(state_file_write(&cow_ops, (ModelState *)&parent, 0, 5000, pfd[0]) > 0,
        "CoW parent writes part 0..5000");

  // Branch forks at 3000 (11 full pages 0..10 shared + partial page 11), then
  // generates 3000..4600 and saves only its suffix part 3000..4600.
  check(kv_page_table_fork(&branch, &parent, 3000),
        "CoW branch forks parent at 3000");
  check(kv_page_table_prepare_write(&branch, 3000, 1600),
        "CoW branch allocates suffix 3000..4600");
  for (int pos = 3000; pos < 4600; pos++) {
    for (int sl = 0; sl < PAGED_SLICES; sl++) {
      uint8_t *dst = cow_row(&branch, sl, pos);
      for (int k = 0; k < PAGED_UNIT; k++)
        dst[k] = byte_of(sl + 20, pos, 2, (size_t)k);
    }
  }
  check(state_file_write(&cow_ops, (ModelState *)&branch, 3000, 4600, pfd[1]) >
            0,
        "CoW branch writes suffix part 3000..4600");

  // 1) Reload branch from disk while parent is in RAM: full prefix pages 0..10
  // stay pointer-shared with parent, page 11 stays private, rows match branch.
  check(kv_page_table_fork(&loaded, &parent, 3000),
        "reload forks prefix 0..3000 from live parent");
  check(!state_file_read_from(&cow_ops, (ModelState *)&loaded, 3000, 4600, pfd,
                              2),
        "state_file_read_from reads suffix 3000..4600 without resetting pages");
  int full_shared = 1;
  for (int p = 0; p < 3000 / PAGED_ROWS; p++) {
    if (loaded.pages[p] != parent.pages[p] || parent.pages[p]->refs != 3)
      full_shared = 0;
  }
  check(full_shared, "reloaded branch shares full prefix pages 0..10 in RAM");
  check(loaded.pages[3000 / PAGED_ROWS] != parent.pages[3000 / PAGED_ROWS] &&
            loaded.pages[3000 / PAGED_ROWS]->refs == 1,
        "reloaded branch keeps partial boundary page 11 private");
  int match = 1;
  for (int pos = 0; match && pos < 4600; pos++) {
    for (int sl = 0; match && sl < PAGED_SLICES; sl++) {
      if (memcmp(cow_row(&loaded, sl, pos), cow_row(&branch, sl, pos),
                 PAGED_UNIT))
        match = 0;
    }
  }
  check(match, "reloaded branch matches original branch byte for byte");

  // 2) Mid-part keep crossing a 4 MiB block boundary (4200 * 1024 B > 4 MiB).
  check(kv_page_table_fork(&loaded_mid, &parent, 4200),
        "mid-part reload forks prefix 0..4200 from live parent");
  check(!state_file_read_from(&cow_ops, (ModelState *)&loaded_mid, 4200, 5000,
                              pfd, 1),
        "state_file_read_from reads suffix across 4 MiB block boundary");
  int mid_shared = 1;
  for (int p = 0; p < 4200 / PAGED_ROWS; p++) {
    if (loaded_mid.pages[p] != parent.pages[p]) mid_shared = 0;
  }
  check(mid_shared, "mid-part reload shares full prefix pages 0..15 in RAM");
  int mid_match = 1;
  for (int pos = 0; mid_match && pos < 5000; pos++) {
    for (int sl = 0; mid_match && sl < PAGED_SLICES; sl++) {
      if (memcmp(cow_row(&loaded_mid, sl, pos), cow_row(&parent, sl, pos),
                 PAGED_UNIT))
        mid_match = 0;
    }
  }
  check(mid_match, "mid-part reload matches parent byte for byte");

  // 3) Exact prefix match (keep == positions): reads zero row blocks and keeps
  // all full prefix pages shared.
  check(kv_page_table_fork(&loaded_exact, &parent, 3000),
        "exact-prefix reload forks 3000 positions");
  check(!state_file_read_from(&cow_ops, (ModelState *)&loaded_exact, 3000, 3000,
                              pfd, 2),
        "state_file_read_from succeeds with zero data reads when keep == pos");
  check(loaded_exact.pages[0] == parent.pages[0],
        "exact-prefix reload preserves shared prefix pages");

  close(pfd[0]), close(pfd[1]);
  kv_page_table_free(&loaded_exact);
  kv_page_table_free(&loaded_mid);
  kv_page_table_free(&loaded);
  kv_page_table_free(&branch);
  kv_page_table_free(&parent);
}

int main(void) {
  const char *base = getenv("TMPDIR");
  if (!base || !*base) base = "/tmp";
  if (snprintf(directory, sizeof directory,
               "%s/eremite-state-file-check.XXXXXX",
               base) >= (int)sizeof directory ||
      !mkdtemp(directory)) {
    perror("mkdtemp");
    return 1;
  }
  atexit(cleanup);

  // one part, every prefix
  Fake *s = calloc(1, sizeof *s);
  run(s, 0, 9000);
  int fds[3] = {open_new("rows-0"), open_new("rows-1"), open_new("rows-2")};
  int64_t written = state_file_write(&fake, (ModelState *)s, 0, 9000, fds[0]);
  check(written > 0 && written % 16384 == 0 &&
            lseek(fds[0], 0, SEEK_END) == written,
        "a part is written in whole 16 KiB spans");
  int prefixes[] = {0, 1, 2, 3, 4, 5, 4095, 4096, 8191, 8999, 9000};
  int all = 1;
  for (size_t k = 0; k < sizeof prefixes / sizeof *prefixes; k++)
    all &= prefix_ok(fds, 1, prefixes[k], s);
  check(all, "every prefix of one part reads back as state_copy gives it");
  check(!prefix_ok(fds, 1, 9001, s), "a prefix past the part is refused");

  // an append from where the part ended: its guard rows come from the append
  run(s, 9000, 12001);
  state_file_write(&fake, (ModelState *)s, 9000, 12001, fds[1]);
  int appended[] = {0, 5000, 8999, 9000, 9001, 9002, 11999, 12000, 12001};
  all = 1;
  for (size_t k = 0; k < sizeof appended / sizeof *appended; k++)
    all &= prefix_ok(fds, 2, appended[k], s);
  check(all, "an append gives every prefix of the state that continued");

  // a rewind: the state continues from 10,000 again, and a part from there
  run(s, 10000, 11003);
  state_file_write(&fake, (ModelState *)s, 10000, 11003, fds[2]);
  int rewound[] = {0, 8999, 9500, 9999, 10000, 10001, 11002, 11003};
  all = 1;
  for (size_t k = 0; k < sizeof rewound / sizeof *rewound; k++)
    all &= prefix_ok(fds, 3, rewound[k], s);
  check(all, "a part from an earlier position replaces the rows after it");
  check(!prefix_ok(fds, 3, 11004, s),
        "a prefix past the latest part is refused");

  // Partial read with odd keep on ratio-2 guard-1 and lag-1 flat regions:
  // keeps stable rows before keep and refreshes guard/lag rows from disk.
  Fake *partial = calloc(1, sizeof *partial);
  run(partial, 0, 8999);
  // Corrupt rows past region_stable(8999) to prove disk overwrites guard/lag.
  for (int i = 0; i < REGIONS; i++) {
    StateRegion r = {partial->rows[i], unit_of[i], RATIO[i], GUARD[i],
                     LAG[i],           0,          0};
    int64_t stable = region_stable(&r, 8999);
    int64_t total = region_rows(&r, 8999);
    if (total > stable)
      memset(partial->rows[i] + (size_t)stable * unit_of[i], 0xFF,
             (size_t)(total - stable) * unit_of[i]);
  }
  check(!state_file_read_from(&fake, (ModelState *)partial, 8999, 11003, fds,
                              3) &&
            same_rows(partial, s, 11003),
        "flat partial read preserves stable prefix and refreshes guard/lag "
        "rows");
  fake_release(partial);
  int reordered[3] = {fds[1], fds[0], fds[2]};
  check(!prefix_ok(reordered, 3, 100, s), "parts out of order are refused");

  // damage: a block the prefix needs, and one it does not. The first part's
  // compressed pairs (4,501 rows of 1 KiB) fill two blocks; the second holds
  // the rows from 4,096 on, those of positions 8,192 and later.
  off_t region1 = 0;
  {
    uint8_t h[16384];
    if (pread(fds[0], h, sizeof h, 0) == (ssize_t)sizeof h) {
      uint64_t at;
      memcpy(&at, h + 40 + 48 * 1 + 40, 8);  // region 1's offset
      region1 = (off_t)at;
    }
  }
  check(reads(fds, 1, 9000), "the first part reads before any damage");
  flip(fds[0], region1 + (4 << 20) + 1000);
  check(!reads(fds, 1, 9000), "a prefix that needs a damaged block is refused");
  check(prefix_ok(fds, 1, 4000, s),
        "a prefix before the damaged block still reads");
  flip(fds[0], region1 + (4 << 20) + 1000);
  flip(fds[0], 100);  // the header
  check(!reads(fds, 1, 10), "a damaged header is refused");
  flip(fds[0], 100);
  check(prefix_ok(fds, 1, 8999, s), "the part reads again once undamaged");
  if (ftruncate(fds[1], lseek(fds[1], 0, SEEK_END) - 16384))
    perror("ftruncate");
  check(!reads(fds, 2, 12001), "a short part is refused");
  unit_of[2] = 256;  // a build that lists another shape
  check(!reads(fds, 1, 100), "a part of another shape is refused");
  unit_of[2] = UNIT[2];

  // snapshots: one with logits, and a live state's end
  FakeSnap *x = (FakeSnap *)fake_alloc((int32_t[]){777}, 1, 1);
  for (int k = 0; k < ROLL; k++) x->roll[k] = byte_of(7, k, 1, 0);
  for (int k = 0; k < VOCAB; k++) x->logits[k] = (float)k * 0.5f;
  int snap = open_new("snap"), end = open_new("end");
  check(snapshot_file_write(&fake, (ModelSnapshot *)x, NULL, 0, snap) > 0,
        "a snapshot is written");
  FakeSnap *y = (FakeSnap *)snapshot_file_read(&fake, snap);
  check(y && y->pos == 777 && !memcmp(y->roll, x->roll, ROLL) && y->logits &&
            !memcmp(y->logits, x->logits, VOCAB * 4),
        "a snapshot reads back with its position, rows and logits");
  fake_free((ModelSnapshot *)y);
  check(snapshot_file_write(&fake, NULL, (ModelState *)s, s->pos, end) > 0,
        "a live state's end is written");
  y = (FakeSnap *)snapshot_file_read(&fake, end);
  check(y && y->pos == s->pos && !memcmp(y->roll, s->roll, ROLL) && !y->logits,
        "a live state's end reads back as a snapshot without logits");
  fake_free((ModelSnapshot *)y);
  flip(snap, 16384 + 4096);
  check(!snapshot_file_read(&fake, snap), "a damaged snapshot is refused");

  identity_checks();
  check(crc32c(0, "123456789", 9) == 0xe3069283,
        "CRC32C gives the standard check value of \"123456789\"");
  check(crc32c(crc32c(0, "1234", 4), "56789", 5) == 0xe3069283,
        "CRC32C continues across pieces");

  paged_checks();
  cow_shared_load_checks();

  // a second stop request ends a write
  shutdown_request(), shutdown_request();
  int last = open_new("rows-3");
  check(state_file_write(&fake, (ModelState *)s, 0, 9000, last) < 0,
        "a second stop request ends a write");
  close(last), close(snap), close(end);
  for (int i = 0; i < 3; i++) close(fds[i]);
  fake_free((ModelSnapshot *)x);
  fake_release(s);
  return failures != 0;
}
