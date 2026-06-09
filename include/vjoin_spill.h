#ifndef VJOIN_SPILL_H
#define VJOIN_SPILL_H

#include "postgres.h"
#include "storage/buffile.h"

/*
 * Per-batch spill file used by vectorized hash join when an inner relation
 * does not fit in work_mem.  Wraps a temporary BufFile and records a small
 * amount of metadata (tuple count, byte count) for EXPLAIN diagnostics.
 *
 * Disk record layout (per tuple):
 *
 *   uint32  hashval
 *   uint16  num_attrs
 *   uint8   isnull_bitmap[ceil(num_attrs / 8)]      -- bit i = isnull[i]
 *   for each attr where !isnull[i]:
 *       if attr_byval[i]:                           -- raw Datum (8 bytes)
 *           Datum    value
 *       else:                                       -- variable-length
 *           int32    length                         -- bytes to follow
 *           uint8    bytes[length]                  -- raw datum bytes
 *
 * For pass-by-reference attributes the caller is responsible for passing
 * already-detoasted datums (typical case after slot_getallattrs).  We
 * persist the on-disk varlena representation as-is (header + payload).
 */

typedef struct vjoin_spill_file
{
    BufFile    *bf;             /* NULL until first write */
    int64       tuple_count;
    int64       byte_count;
} vjoin_spill_file;

extern vjoin_spill_file *vjoin_spill_create(void);
extern void              vjoin_spill_close(vjoin_spill_file *sf);

/* Reset read position to beginning of file (no-op if not yet written). */
extern void              vjoin_spill_rewind(vjoin_spill_file *sf);

/*
 * Write one tuple to the spill file.
 *
 *   hashval     pre-computed 32-bit hash of the join keys
 *   num_attrs   number of attributes (must equal length of values/isnull)
 *   values      Datum[num_attrs] (raw)
 *   isnull      bool[num_attrs]
 *   attr_byval  bool[num_attrs]
 *   attr_typlen int16[num_attrs]   -- only consulted for byval attrs
 */
extern void vjoin_spill_write_tuple(vjoin_spill_file *sf,
                                    uint32 hashval,
                                    int num_attrs,
                                    const Datum *values,
                                    const bool *isnull,
                                    const bool *attr_byval,
                                    const int16 *attr_typlen);

/*
 * Read one tuple from the spill file into caller-provided buffers.
 * The caller must have already called vjoin_spill_rewind() before the
 * first read of a replay pass.
 *
 * Buffers values[] and isnull[] must be at least num_attrs wide (matching
 * the writer).  Variable-length bytes are returned via palloc'd buffers
 * in CurrentMemoryContext, ownership transferred to the caller.
 *
 * Returns true on a tuple, false on clean EOF.
 */
extern bool vjoin_spill_read_tuple(vjoin_spill_file *sf,
                                   uint32 *hashval,
                                   int num_attrs,
                                   Datum *values,
                                   bool *isnull,
                                   const bool *attr_byval,
                                   const int16 *attr_typlen);

#endif                          /* VJOIN_SPILL_H */
