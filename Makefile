# eremite: the engine library, the command-line program, the tests and the
# compilation database that clangd reads.
#
#   make                build/liberemite.a, build/eremite and every test program
#   make test           run the tests that need neither a GPU nor model files
#   make gpu-test       run the tests that need a GPU but no model files
#   make format         format the sources in place with clang-format
#   make format-check   fail if clang-format would change a source
#   make clean          remove build/ and compile_commands.json
#
#   THREADS=pthread     build the threads layer on POSIX threads instead of
#                       Grand Central Dispatch (src/platform/threads.h)
#   SANITIZE=1          build with AddressSanitizer and UndefinedBehaviorSanitizer
#
# Every .c, .m and .S file under src/ goes into the library, except src/cli/,
# which is the command-line program. Every tests/<group>/<name>.c becomes
# build/tests/<group>-<name>.

CC := clang
BUILD := build
OBJ := $(BUILD)/obj
CFLAGS := -std=c23 -O2 -Wall -Wextra -Isrc

# Libraries found through pkg-config.
PKGS := yyjson
CFLAGS += $(shell pkg-config --cflags $(PKGS))
LDLIBS := $(shell pkg-config --libs $(PKGS))

# Metal, Foundation and the Objective-C runtime, for the GPU layer.
LDLIBS += -framework Metal -framework Foundation -lobjc

# ICU, in which the tokenizer runs its regular expressions and normalization.
LDLIBS += -licucore

THREADS ?= gcd
THREADS_FLAG_gcd := -DEREMITE_THREADS_GCD
THREADS_FLAG_pthread := -DEREMITE_THREADS_PTHREAD
$(if $(THREADS_FLAG_$(THREADS)),,$(error THREADS is gcd or pthread))
CFLAGS += $(THREADS_FLAG_$(THREADS))

# Undefined behavior aborts the program instead of only being reported.
SANITIZE ?=
ifneq ($(SANITIZE),)
CFLAGS += -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
endif

# Every object and program is built for one configuration: when THREADS or
# SANITIZE differs from the last build's, that build's outputs are removed
# first.
CONFIG := threads=$(THREADS) sanitize=$(SANITIZE)
-include $(BUILD)/config.mk
ifneq ($(BUILT_CONFIG),$(CONFIG))
$(shell rm -rf $(OBJ) $(BUILD)/tests $(BUILD)/eremite $(BUILD)/liberemite.a; mkdir -p $(BUILD); \
  echo 'BUILT_CONFIG := $(CONFIG)' > $(BUILD)/config.mk)
endif

SOURCES := $(shell find src -name '*.c' -o -name '*.m' -o -name '*.S' 2>/dev/null)
LIB_SRC := $(filter-out src/cli/%,$(SOURCES))
LIB_OBJ := $(LIB_SRC:%=$(OBJ)/%.o)
LIB := $(if $(LIB_SRC),$(BUILD)/liberemite.a)
CLI := $(if $(wildcard src/cli/main.c),$(BUILD)/eremite)

TEST_SRC := $(wildcard tests/*/*.c)
TEST_GROUPS := $(sort $(patsubst tests/%/,%,$(dir $(TEST_SRC))))
TESTS := $(foreach t,$(TEST_SRC),$(BUILD)/tests/$(subst /,-,$(t:tests/%.c=%)))
group_tests = $(filter $(BUILD)/tests/$(1)-%,$(TESTS))
# Each test runs with a time limit, so that a hang fails the run: after
# TEST_SECONDS, the test and the processes it started end. 0 sets no limit.
TEST_SECONDS ?= 600
TIME_LIMIT := $(BUILD)/tests/time-limit

# A test links the whole library unless it sets TEST_LIBS to something
# smaller. Tests that include a library source to replace its calls rely on
# archive linking, which skips members whose symbols are already defined.
TEST_LIBS = $(LIB)
TEST_CFLAGS =

FORMAT_SRC = $(shell find src tests -name '*.c' -o -name '*.m' -o -name '*.h' 2>/dev/null)

.PHONY: all test gpu-test format format-check clean
all: $(LIB) $(CLI) $(TESTS) $(TIME_LIMIT) compile_commands.json

# Each compiled file writes its entry of the compilation database beside its
# output, with the command as written here.
database_entry = printf '{"directory": "%s", "file": "%s", "command": "%s"},\n' '$(CURDIR)' '$(1)' '$(2)' > $(3)

$(OBJ)/%.o: %
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@
	@$(call database_entry,$<,$(CC) $(CFLAGS) -c $< -o $@,$(@:.o=.json))

$(BUILD)/liberemite.a: $(LIB_OBJ)
	rm -f $@
	ar rcs $@ $^

$(BUILD)/eremite: src/cli/main.c $(LIB)
	@mkdir -p $(OBJ)/src/cli
	$(CC) $(CFLAGS) -MMD -MP -MF $(OBJ)/src/cli/main.d -o $@ $< $(LIB) $(LDLIBS)
	@$(call database_entry,$<,$(CC) $(CFLAGS) -c $<,$(OBJ)/src/cli/main.json)

define test_rule
$(BUILD)/tests/$(1)-%: tests/$(1)/%.c $$(LIB)
	@mkdir -p $$(@D)
	$$(CC) $$(CFLAGS) $$(TEST_CFLAGS) -Itests -MMD -MP -MF $$@.d -o $$@ $$< $$(TEST_LIBS) $$(LDLIBS)
	@$$(call database_entry,$$<,$$(CC) $$(CFLAGS) $$(TEST_CFLAGS) -Itests -c $$<,$$@.json)
endef
$(foreach g,$(TEST_GROUPS),$(eval $(call test_rule,$(g))))

# clangd's compilation database, from the entries beside each object and program.
compile_commands.json: $(LIB_OBJ) $(CLI) $(TESTS)
	@mkdir -p $(BUILD)
	@{ echo '['; find $(BUILD) -name '*.json' -exec cat {} + | sed '$$ s/,$$//'; echo ']'; } > $@

$(TIME_LIMIT): tests/time_limit.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $<

test: $(TIME_LIMIT) $(call group_tests,unit)
	@set -e; for t in $(call group_tests,unit); do echo "== $$t"; $(TIME_LIMIT) $(TEST_SECONDS) $$t; done

gpu-test: $(TIME_LIMIT) $(call group_tests,gpu)
	@set -e; for t in $(call group_tests,gpu); do echo "== $$t"; $(TIME_LIMIT) $(TEST_SECONDS) $$t; done

format:
	$(if $(FORMAT_SRC),clang-format -i $(FORMAT_SRC))

format-check:
	$(if $(FORMAT_SRC),clang-format --dry-run --Werror $(FORMAT_SRC))

clean:
	rm -rf $(BUILD) compile_commands.json

-include $(LIB_OBJ:.o=.d) $(OBJ)/src/cli/main.d $(TESTS:=.d)
