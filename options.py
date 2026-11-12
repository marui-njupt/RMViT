"""Explicit command-line arguments and model option validation."""
import argparse

MODEL_FIELDS = {'heads', 'enc_layers', 'dec_layers', 'dim_ff', 'dropout',
                'single_scale_skip', 'block_cross_modal'}


def validate_runtime(cfg):
    if not isinstance(cfg, dict):
        raise ValueError('Pass a dictionary containing stage1 and stage2 options.')
    for stage in ('stage1', 'stage2'):
        expected = MODEL_FIELDS | ({'direct_prediction'} if stage == 'stage2' else set())
        values = cfg.get(stage)
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f'{stage}: supply all model arguments explicitly.')
        for field, value in values.items():
            if field in {'single_scale_skip', 'block_cross_modal', 'direct_prediction'}:
                if type(value) is not bool:
                    raise ValueError(f'{stage}.{field}: boolean required.')
            elif field == 'dropout':
                if type(value) not in (float, int) or not 0 <= value < 1:
                    raise ValueError(f'{stage}.{field}: probability required.')
            elif type(value) is not int or value <= 0:
                raise ValueError(f'{stage}.{field}: positive integer required.')
    return cfg


def build_pair(stage1_options, stage2_options):
    from models.network import build_pair as build
    return build(stage1_options, stage2_options)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Check explicit model arguments without running a model.')
    for stage in ('stage1', 'stage2'):
        for name in ('heads', 'enc_layers', 'dec_layers', 'dim_ff'):
            parser.add_argument(f'--{stage}-{name.replace("_", "-")}', type=int, required=True)
        parser.add_argument(f'--{stage}-dropout', type=float, required=True)
        flags = ['single_scale_skip', 'block_cross_modal']
        if stage == 'stage2':
            flags.append('direct_prediction')
        for name in flags:
            parser.add_argument(f'--{stage}-{name.replace("_", "-")}',
                                action=argparse.BooleanOptionalAction, required=True)
    return parser.parse_args(argv)


def runtime_from_args(args):
    values = vars(args)
    return validate_runtime({stage: {key[len(stage)+1:]: value for key, value in values.items()
                                    if key.startswith(stage + '_')}
                             for stage in ('stage1', 'stage2')})


def parse_training_args(parser, argv=None):
    return parser.parse_args(argv)
