"""Construct both stages from explicit model options."""
from .stage1 import PatchTF_RM_BM_DecoderUse32Tokens_ConcatFuse_Half_Skip6_No1x1 as Stage1
from .stage2 import PatchTF_RefineRM_BM_Stage2_Half_Skip6_No1x1 as Stage2


def build_pair(stage1_options, stage2_options):
    from options import validate_runtime
    cfg = validate_runtime({'stage1': stage1_options, 'stage2': stage2_options})
    return Stage1(**cfg['stage1']), Stage2(**cfg['stage2'])
