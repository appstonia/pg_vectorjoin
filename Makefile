MODULE_big = pg_vectorjoin
OBJS = src/pg_vectorjoin.o \
       src/vjoin_path.o \
       src/vjoin_plan.o \
       src/vjoin_exec_hash.o \
       src/vjoin_exec_nestloop.o \
       src/vjoin_hashtable.o \
       src/vjoin_simd.o \
       src/vjoin_exec_merge.o

EXTENSION = pg_vectorjoin
DATA = pg_vectorjoin--1.0.sql

PG_CPPFLAGS = -I$(srcdir)/include
PG_CFLAGS = -O2

REGRESS = basic outer_join_test
REGRESS_OPTS = --inputdir=$(srcdir)/test

PG_CONFIG ?= pg_config
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
