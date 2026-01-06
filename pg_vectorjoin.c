#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"

PG_MODULE_MAGIC;

void _PG_init(void);

/* GUC variables */
bool vjoin_enable = true;

void
_PG_init(void)
{
    DefineCustomBoolVariable("pg_vectorjoin.enable",
                             "Enable pg_vectorjoin extension.",
                             NULL,
                             &vjoin_enable,
                             true,
                             PGC_USERSET,
                             0, NULL, NULL, NULL);

#if PG_VERSION_NUM >= 150000
    MarkGUCPrefixReserved("pg_vectorjoin");
#else
    EmitWarningsOnPlaceholders("pg_vectorjoin");
#endif
}
