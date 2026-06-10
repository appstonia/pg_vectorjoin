MODULE_big = pg_vectorjoin
ifdef VPATH
OBJS = $(VPATH)src/pg_vectorjoin.o \
       $(VPATH)src/vjoin_path.o \
       $(VPATH)src/vjoin_plan.o \
       $(VPATH)src/vjoin_exec_hash.o \
       $(VPATH)src/vjoin_exec_nestloop.o \
       $(VPATH)src/vjoin_hashtable.o \
       $(VPATH)src/vjoin_simd.o \
       $(VPATH)src/vjoin_exec_merge.o \
       $(VPATH)src/vjoin_spill.o
else
OBJS = src/pg_vectorjoin.o src/vjoin_path.o src/vjoin_plan.o \
       src/vjoin_exec_hash.o src/vjoin_exec_nestloop.o \
       src/vjoin_hashtable.o src/vjoin_simd.o \
       src/vjoin_exec_merge.o src/vjoin_spill.o
endif

EXTENSION = pg_vectorjoin
DATA = pg_vectorjoin--1.0.sql

PG_CPPFLAGS = -I$(srcdir)/include
PG_CFLAGS = -O2 -Wno-ignored-attributes

REGRESS = basic outer_join_test spill_test par_spill_test
REGRESS_OPTS = --inputdir=$(srcdir)/test

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)

include $(PGXS)
