#ifndef VJOIN_COMPAT_H
#define VJOIN_COMPAT_H

#include "postgres.h"

/* MarkGUCPrefixReserved was added in PG15 replacing EmitWarningsOnPlaceholders */
#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(prefix) EmitWarningsOnPlaceholders(prefix)
#endif

/* join_path_setup_hook was added in PG19 */
#if PG_VERSION_NUM >= 190000
#define VJOIN_HAS_SETUP_HOOK 1
#else
#define VJOIN_HAS_SETUP_HOOK 0
#endif

/* PGS_* strategy mask flags (PG19 only) */
#if PG_VERSION_NUM >= 190000
#include "nodes/pathnodes.h"
#define VJOIN_HAS_PGS_MASK 1
#else
#define VJOIN_HAS_PGS_MASK 0
#endif

/* CUSTOMPATH_SUPPORT_PROJECTION was added in PG17 */
#ifndef CUSTOMPATH_SUPPORT_PROJECTION
#define CUSTOMPATH_SUPPORT_PROJECTION 0x0004
#endif

/* custom_restrictinfo field in CustomPath was added in PG17 */
#if PG_VERSION_NUM >= 170000
#define VJOIN_HAS_CUSTOM_RESTRICTINFO 1
#else
#define VJOIN_HAS_CUSTOM_RESTRICTINFO 0
#endif

/* PG18 moved ExplainProperty* functions from commands/explain.h
 * to commands/explain_format.h */
#if PG_VERSION_NUM >= 180000
#include "commands/explain_format.h"
#else
#include "commands/explain.h"
#endif

/* PG18 added an extra "Size extra" parameter to heap_copy_minimal_tuple
 * and heap_form_minimal_tuple */
#if PG_VERSION_NUM >= 180000
#define vjoin_heap_copy_minimal_tuple(mt)  heap_copy_minimal_tuple((mt), 0)
#define vjoin_heap_form_minimal_tuple(desc, vals, nulls) \
    heap_form_minimal_tuple((desc), (vals), (nulls), 0)
#else
#define vjoin_heap_copy_minimal_tuple(mt)  heap_copy_minimal_tuple((mt))
#define vjoin_heap_form_minimal_tuple(desc, vals, nulls) \
    heap_form_minimal_tuple((desc), (vals), (nulls))
#endif

/* Value vs Integer API change: PG16+ uses makeInteger returning Integer*,
 * PG14-15 uses makeInteger returning Value* — but intVal() works across all */

#endif /* VJOIN_COMPAT_H */
