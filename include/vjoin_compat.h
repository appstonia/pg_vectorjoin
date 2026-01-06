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

/* custom_restrictinfo field in CustomPath was added in PG16 */
#if PG_VERSION_NUM >= 160000
#define VJOIN_HAS_CUSTOM_RESTRICTINFO 1
#else
#define VJOIN_HAS_CUSTOM_RESTRICTINFO 0
#endif

/* Value vs Integer API change: PG16+ uses makeInteger returning Integer*,
 * PG14-15 uses makeInteger returning Value* — but intVal() works across all */

#endif /* VJOIN_COMPAT_H */
