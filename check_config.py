"""Validate model arguments without starting an experiment."""
from options import runtime_from_args, parse_args


def main():
    args = parse_args()
    runtime_from_args(args)
    print('Model configuration fields are valid. No model or experiment was started.')


if __name__ == '__main__':
    main()
