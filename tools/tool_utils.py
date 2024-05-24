import typing as tp
import timeit

import torch


def time_func(
    key: str,
    func: tp.Callable[..., tp.Any],
    timers: tp.Dict[str, float],
    sync: bool = False,
    nvtx: bool = False,
):
    timers[key] = 0.0

    def wrapper(*args, **kwargs):
        if sync:
            torch.cuda.synchronize()
        if nvtx:
            torch.cuda.nvtx.range_push(key)
        start = timeit.default_timer()
        ret = func(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        if nvtx:
            torch.cuda.nvtx.range_pop()
        end = timeit.default_timer()
        timers[key] += end - start
        return ret

    return wrapper


def time_functions(
    names_models: tp.Sequence[
        tp.Tuple[tp.Union[str, tp.Tuple[str, ...]], torch.nn.Module]
    ],
    timers: tp.Dict[str, float],
    sync: bool = False,
    nvtx: bool = False,
):
    for fn_names, model in names_models:
        if isinstance(fn_names, str):
            fn_names = (fn_names,)
        for fn_name in fn_names:
            setattr(
                model,
                fn_name,
                time_func(
                    ".".join((model.__class__.__name__, fn_name)),
                    getattr(model, fn_name),
                    timers,
                    sync,
                    nvtx,
                ),
            )
