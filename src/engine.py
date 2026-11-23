from arguments import parse_args
from data import get_node_dataloader
from objectives import *
from validation import validate_strict


def get_model_classes(model_impl):
    if model_impl != 'dualfusion':
        raise ValueError('Only the final teacher/student architecture is included')
    from teacher import MultiModalLymphNodeNet as Teacher
    from student import MultiModalLymphNodeNet as Student
    return Teacher, Student


def resolve_input_mode(stage, model_impl, mode):
    return ('full' if stage == 'teacher' else 'limited') if mode == 'auto' else mode


def build_loader(path, batch_size, is_train, distributed, stage, args):
    return get_node_dataloader(path, batch_size=batch_size, is_train=is_train, use_ddp=False,
        stage=stage, loader_workers=args.loader_workers, cache_workers=args.cache_workers,
        cache_rate=args.train_cache_rate if is_train else args.eval_cache_rate,
        persistent_workers=False, disable_random_aug=not is_train, balanced_sampler=False,
        physical_preprocessing=args.physical_preprocessing)
