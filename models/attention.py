"""Boolean attention masks use True to block an attention edge."""
import torch


def install_attention_masks(model, blocked):
    model.block_cross_modal = bool(blocked)
    n = model.num_patches
    enc = query = None
    if blocked:
        enc = torch.zeros(2 * n, 2 * n, dtype=torch.bool)
        enc[:n, n:] = True
        enc[n:, :n] = True
        query = torch.zeros(n, 2 * n, dtype=torch.bool)
        query[:, n:] = True

    model.register_buffer('encoder_attention_mask', enc, persistent=False)
    model.register_buffer('query_memory_mask', query, persistent=False)


def mechanism_record(stage1, stage2, experiment, gradient_weight):

    blocked = experiment == 'A4'
    records = []
    for model in (stage1, stage2):
        if model is None:
            continue
        if model.single_scale_skip != (experiment == 'A5'):
            raise ValueError('Single-scale skip flag mismatch.')
        if model.block_cross_modal != blocked:
            raise ValueError('Cross-modal flag does not match experiment.')
        if blocked:
            n = model.num_patches
            expected = torch.zeros_like(model.encoder_attention_mask)
            expected[:n, n:] = expected[n:, :n] = True
            expected_q = torch.zeros_like(model.query_memory_mask)
            expected_q[:, n:] = True
            if not torch.equal(expected, model.encoder_attention_mask):
                raise ValueError('Encoder must block exactly the two off-diagonal blocks.')
            if not torch.equal(expected_q, model.query_memory_mask):
                raise ValueError('Decoder must block exactly BM memory.')
        elif model.encoder_attention_mask is not None or model.query_memory_mask is not None:
            raise ValueError('Unexpected mask in the full-attention model.')
        retained = ['rm_enc', 'bm_enc', 'encoder', 'decoder', 'rm_query',
                    'decode_fusion', 'global_decoder']
        retained += ['skip_fuse' + str(s) for s in (8, 16, 32, 64, 128, 256)]
        if not all(hasattr(model, name) for name in retained):
            raise ValueError('A required fusion/attention path is missing.')
        records.append({'parameters': sum(p.numel() for p in model.parameters()),
                        'block_cross_modal': blocked, 'retained_modules': retained,
                        'retained_skip_scales': [64] if model.single_scale_skip else [8,16,32,64,128,256],
                        'zeroed_fused_skip_scales': [8,16,32,128,256] if model.single_scale_skip else []})
    if stage2 is not None and stage2.direct_prediction != (experiment == 'A2'):
        raise ValueError('A2 must directly predict; other second stages must add coarse.')
    if experiment == 'A3' and gradient_weight != 0.0:
        raise ValueError('A3 gradient-loss weight must be exactly zero.')
    return {'experiment': experiment, 'gradient_weight': gradient_weight,
            'models': records, 'check_type': 'configuration_and_mask_values_only'}
