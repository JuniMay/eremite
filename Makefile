# eremite: the engine library, the command-line program, the tests and the
# compilation database that clangd reads.
#
#   make                build/liberemite.a, build/eremite and every test program
#   make test           run the tests that need neither a GPU nor model files
#   make gpu-test       run the tests that need a GPU but no model files
#   make format         format the C sources in place with clang-format
#   make format-check   fail if clang-format would change a C source
#   make clean          remove build/ and compile_commands.json
#
# Every .c and .S file under src/ goes into the library, except src/cli/,
# which is the command-line program. Every tests/<group>/<name>.c becomes
# build/tests/<group>-<name>.

CC := clang
BUILD := build
OBJ := $(BUILD)/obj
CFLAGS := -std=c23 -O2 -Wall -Wextra -Isrc

SOURCES := $(shell find src -name '*.c' -o -name '*.S' 2>/dev/null)
LIB_SRC := $(filter-out src/cli/%,$(SOURCES))
LIB_OBJ := $(LIB_SRC:%=$(OBJ)/%.o)
LIB := $(if $(LIB_SRC),$(BUILD)/liberemite.a)
CLI := $(if $(wildcard src/cli/main.c),$(BUILD)/eremite)

TEST_SRC := $(wildcard tests/*/*.c)
TEST_GROUPS := $(sort $(patsubst tests/%/,%,$(dir $(TEST_SRC))))
TESTS := $(foreach t,$(TEST_SRC),$(BUILD)/tests/$(subst /,-,$(t:tests/%.c=%)))
group_tests = $(filter $(BUILD)/tests/$(1)-%,$(TESTS))

# A test links the whole library unless it sets TEST_LIBS to something
# smaller. Tests that include a library source to replace its calls rely on
# archive linking, which skips members whose symbols are already defined.
TEST_LIBS = $(LIB)
TEST_CFLAGS =

FORMAT_SRC = $(shell find src tests -name '*.c' -o -name '*.h' 2>/dev/null)

.PHONY: all test gpu-test format format-check clean
all: $(LIB) $(CLI) $(TESTS) compile_commands.json

$(OBJ)/%.o: %
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -MJ $(@:.o=.json) -c $< -o $@

$(BUILD)/liberemite.a: $(LIB_OBJ)
	rm -f $@
	ar rcs $@ $^

$(BUILD)/eremite: src/cli/main.c $(LIB)
	@mkdir -p $(OBJ)/src/cli
	$(CC) $(CFLAGS) -MMD -MP -MF $(OBJ)/src/cli/main.d -MJ $(OBJ)/src/cli/main.json -o $@ $< $(LIB) $(LDLIBS)

define test_rule
$(BUILD)/tests/$(1)-%: tests/$(1)/%.c $$(LIB)
	@mkdir -p $$(@D)
	$$(CC) $$(CFLAGS) $$(TEST_CFLAGS) -Itests -MMD -MP -MF $$@.d -MJ $$@.json -o $$@ $$< $$(TEST_LIBS) $$(LDLIBS)
endef
$(foreach g,$(TEST_GROUPS),$(eval $(call test_rule,$(g))))

# clangd's compilation database, from the fragments that -MJ writes beside
# each object and program.
compile_commands.json: $(LIB_OBJ) $(CLI) $(TESTS)
	@mkdir -p $(BUILD)
	@{ echo '['; find $(BUILD) -name '*.json' -exec cat {} + | sed '$$ s/,$$//'; echo ']'; } > $@

test: $(call group_tests,unit)
	@set -e; for t in $^; do echo "== $$t"; $$t; done

gpu-test: $(call group_tests,gpu)
	@set -e; for t in $^; do echo "== $$t"; $$t; done

format:
	$(if $(FORMAT_SRC),clang-format -i $(FORMAT_SRC))

format-check:
	$(if $(FORMAT_SRC),clang-format --dry-run --Werror $(FORMAT_SRC))

clean:
	rm -rf $(BUILD) compile_commands.json

-include $(LIB_OBJ:.o=.d) $(OBJ)/src/cli/main.d $(TESTS:=.d)
