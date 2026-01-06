#include "postgres.h"
#include "optimizer/paths.h"
#include "vjoin_compat.h"
#include "pg_vectorjoin.h"

/*
 * vjoin_pathlist_hook — called by the planner for each join relation.
 * Currently a skeleton that just chains to the previous hook.
 */
void
vjoin_pathlist_hook(PlannerInfo *root,
                    RelOptInfo *joinrel,
                    RelOptInfo *outerrel,
                    RelOptInfo *innerrel,
                    JoinType jointype,
                    JoinPathExtraData *extra)
{
    /* Chain to any previous hook first */
    if (prev_join_pathlist_hook)
        prev_join_pathlist_hook(root, joinrel, outerrel, innerrel,
                                jointype, extra);

    /* Only inner joins for now */
    if (jointype != JOIN_INNER)
        return;

    /* Master kill switch */
    if (!vjoin_enable)
        return;

    /* TODO: clause analysis, path creation */
}
