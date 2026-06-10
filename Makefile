MODULE_big = pg_vectorjoin
OBJS = src/pg_vectorjoin.o \
       src/vjoin_path.o \
       src/vjoin_plan.o \
       src/vjoin_exec_hash.o \
       src/vjoin_exec_nestloop.o \
       src/vjoin_hashtable.o \
       src/vjoin_simd.o \
       src/vjoin_exec_merge.o \
       src/vjoin_spill.o

EXTENSION = pg_vectorjoin
DATA = pg_vectorjoin--1.0.sql

PG_CPPFLAGS = -I$(srcdir)/include
PG_CFLAGS = -O2 -Wno-ignored-attributes

REGRESS = basic outer_join_test spill_test par_spill_test
REGRESS_OPTS = --inputdir=$(srcdir)/test

# Default to PostgreSQL 18 (Homebrew). Override with `make PG_CONFIG=...`
# to build against a different installation.
PG_CONFIG ?= /opt/homebrew/opt/postgresql@18/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)

# Fix stale sysroot on macOS when pg_config was built with an older SDK
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  _XCRUN_SYSROOT := $(shell xcrun --show-sdk-path 2>/dev/null)
  ifneq ($(_XCRUN_SYSROOT),)
    override PG_SYSROOT = $(_XCRUN_SYSROOT)
  endif
endif

include $(PGXS)
