/*-------------------------------------------------------------------------
 * vjoin_spill.c
 *      Per-batch BufFile-backed tuple spill for vectorized hash join.
 *
 * Disk format documented in vjoin_spill.h.  Tuples are written without any
 * framing other than the per-tuple header (hashval + num_attrs + isnull
 * bitmap).  Variable-length attributes are stored as a 4-byte length prefix
 * followed by the raw datum bytes; pass-by-value attributes are stored as
 * a fixed 8-byte Datum.
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "storage/buffile.h"
#include "utils/memutils.h"
#include "utils/palloc.h"
#include "varatt.h"

#include "vjoin_spill.h"

/* Stack-friendly threshold for the isnull bitmap.  Joins with > 1024
 * attributes will hit a palloc fallback (extremely unlikely in practice). */
#define VJOIN_SPILL_BITMAP_STACK_BYTES  128

vjoin_spill_file *
vjoin_spill_create(void)
{
    vjoin_spill_file *sf = palloc0(sizeof(vjoin_spill_file));

    /*
     * Remember the context the struct itself lives in.  The BufFile is
     * created lazily on first write, possibly while a short-lived context
     * (e.g. the per-batch context, which is reset between batches) is
     * current.  BufFileCreateTemp allocates in CurrentMemoryContext, so we
     * must switch back to this long-lived context to keep the BufFile alive
     * until vjoin_spill_close.
     */
    sf->owner = CurrentMemoryContext;

    /* Lazy: BufFile is only created on first write. */
    return sf;
}

void
vjoin_spill_close(vjoin_spill_file *sf)
{
    if (sf == NULL)
        return;
    if (sf->bf != NULL)
    {
        BufFileClose(sf->bf);
        sf->bf = NULL;
    }
    pfree(sf);
}

void
vjoin_spill_rewind(vjoin_spill_file *sf)
{
    if (sf == NULL || sf->bf == NULL)
        return;
    if (BufFileSeek(sf->bf, 0, 0, SEEK_SET) != 0)
        ereport(ERROR,
                (errcode_for_file_access(),
                 errmsg("could not rewind vjoin spill file")));
}

static inline void
vjoin_spill_ensure_bf(vjoin_spill_file *sf)
{
    if (sf->bf == NULL)
    {
        MemoryContext old = MemoryContextSwitchTo(sf->owner);

        sf->bf = BufFileCreateTemp(false);
        MemoryContextSwitchTo(old);
    }
}

static inline int
bitmap_bytes(int num_attrs)
{
    return (num_attrs + 7) / 8;
}

void
vjoin_spill_write_tuple(vjoin_spill_file *sf,
                        uint32 hashval,
                        int num_attrs,
                        const Datum *values,
                        const bool *isnull,
                        const bool *attr_byval,
                        const int16 *attr_typlen)
{
    uint16      n16;
    int         nbits;
    uint8       stack_bm[VJOIN_SPILL_BITMAP_STACK_BYTES];
    uint8      *bm;
    int         i;

    Assert(sf != NULL);
    Assert(num_attrs >= 0 && num_attrs <= UINT16_MAX);

    vjoin_spill_ensure_bf(sf);

    /* Build isnull bitmap. */
    nbits = bitmap_bytes(num_attrs);
    if (nbits <= VJOIN_SPILL_BITMAP_STACK_BYTES)
        bm = stack_bm;
    else
        bm = (uint8 *) palloc(nbits);
    memset(bm, 0, nbits);
    for (i = 0; i < num_attrs; i++)
    {
        if (isnull[i])
            bm[i >> 3] |= (uint8) (1u << (i & 7));
    }

    /* Header. */
    BufFileWrite(sf->bf, &hashval, sizeof(uint32));
    n16 = (uint16) num_attrs;
    BufFileWrite(sf->bf, &n16, sizeof(uint16));
    BufFileWrite(sf->bf, bm, nbits);

    sf->byte_count += sizeof(uint32) + sizeof(uint16) + nbits;

    /* Body. */
    for (i = 0; i < num_attrs; i++)
    {
        if (isnull[i])
            continue;

        if (attr_byval[i])
        {
            /* Always serialize as a full Datum to keep the format uniform
             * regardless of platform pointer width. */
            BufFileWrite(sf->bf, &values[i], sizeof(Datum));
            sf->byte_count += sizeof(Datum);
        }
        else
        {
            const void *src;
            int32       len;

            if (attr_typlen[i] > 0)
            {
                /* Fixed-length pass-by-reference (e.g. name).  Length
                 * known up-front from typlen. */
                len = (int32) attr_typlen[i];
                src = DatumGetPointer(values[i]);
            }
            else if (attr_typlen[i] == -1)
            {
                /* Variable-length (varlena).  Persist the raw on-disk
                 * representation; caller is responsible for any required
                 * detoasting before insertion. */
                struct varlena *v = (struct varlena *) DatumGetPointer(values[i]);

                len = (int32) VARSIZE_ANY(v);
                src = v;
            }
            else if (attr_typlen[i] == -2)
            {
                /* Cstring. */
                const char *s = DatumGetCString(values[i]);

                len = (int32) (strlen(s) + 1);
                src = s;
            }
            else
                elog(ERROR, "vjoin_spill: unsupported attr_typlen %d",
                     attr_typlen[i]);

            BufFileWrite(sf->bf, &len, sizeof(int32));
            BufFileWrite(sf->bf, src, (size_t) len);
            sf->byte_count += sizeof(int32) + len;
        }
    }

    if (bm != stack_bm)
        pfree(bm);
    sf->tuple_count++;
}

bool
vjoin_spill_read_tuple(vjoin_spill_file *sf,
                       uint32 *hashval,
                       int num_attrs,
                       Datum *values,
                       bool *isnull,
                       const bool *attr_byval,
                       const int16 *attr_typlen)
{
    size_t      got;
    uint16      n16;
    int         nbits;
    uint8       stack_bm[VJOIN_SPILL_BITMAP_STACK_BYTES];
    uint8      *bm;
    int         i;

    Assert(sf != NULL);

    if (sf->bf == NULL)
        return false;           /* never written → empty */

    /* Probe for EOF using the hashval header. */
    got = BufFileReadMaybeEOF(sf->bf, hashval, sizeof(uint32), true);
    if (got == 0)
        return false;
    if (got != sizeof(uint32))
        ereport(ERROR,
                (errcode_for_file_access(),
                 errmsg("short read from vjoin spill file (hashval)")));

    BufFileReadExact(sf->bf, &n16, sizeof(uint16));
    if ((int) n16 != num_attrs)
        ereport(ERROR,
                (errcode(ERRCODE_DATA_CORRUPTED),
                 errmsg("vjoin spill file attribute count mismatch: got %u, expected %d",
                        (unsigned) n16, num_attrs)));

    nbits = bitmap_bytes(num_attrs);
    if (nbits <= VJOIN_SPILL_BITMAP_STACK_BYTES)
        bm = stack_bm;
    else
        bm = (uint8 *) palloc(nbits);
    BufFileReadExact(sf->bf, bm, nbits);

    for (i = 0; i < num_attrs; i++)
    {
        bool        is_null = ((bm[i >> 3] >> (i & 7)) & 1u) != 0;

        isnull[i] = is_null;
        if (is_null)
        {
            values[i] = (Datum) 0;
            continue;
        }

        if (attr_byval[i])
        {
            BufFileReadExact(sf->bf, &values[i], sizeof(Datum));
        }
        else
        {
            int32       len;
            char       *buf;

            BufFileReadExact(sf->bf, &len, sizeof(int32));
            if (len < 0)
                ereport(ERROR,
                        (errcode(ERRCODE_DATA_CORRUPTED),
                         errmsg("vjoin spill file negative attr length")));
            buf = (char *) palloc((size_t) len);
            BufFileReadExact(sf->bf, buf, (size_t) len);
            values[i] = PointerGetDatum(buf);
        }
    }

    if (bm != stack_bm)
        pfree(bm);
    return true;
}
