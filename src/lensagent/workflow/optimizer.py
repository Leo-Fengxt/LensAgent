"""Conversion between native MGE models and their Gaussian optimizer basis."""

import copy
from dataclasses import replace

from lensagent.modeling.parameters import pack_multi_gaussian_components


def optimizer_space(space):
    if not space.pso_proxy_lens_models:
        return space
    changes = {}
    for field in ("bounds_lens", "fixed_lens", "centers_lens", "sigmas_lens"):
        rows = []
        for profile, values in zip(space.model["lens_model_list"], getattr(space, field), strict=True):
            if profile != "MULTI_GAUSSIAN":
                rows.append(copy.deepcopy(values))
                continue
            for index in range(space.mge_components):
                row = {k: copy.deepcopy(v) for k, v in values.items() if k in {"center_x", "center_y"}}
                for name in ("amp", "sigma"):
                    if f"{name}_{index}" in values:
                        row[name] = copy.deepcopy(values[f"{name}_{index}"])
                    elif name in values:
                        row[name] = float(values[name][index])
                rows.append(row)
        changes[field] = tuple(rows)
    return replace(space, **changes,
                   model={**space.model, "lens_model_list": list(space.pso_proxy_lens_models)},
                   pso_proxy_lens_models=())


def optimizer_proposal(proposal, space):
    result = copy.deepcopy(proposal)
    if not space.pso_proxy_lens_models:
        return result
    packed = pack_multi_gaussian_components(proposal, space.model)
    rows = []
    for profile, values in zip(space.model["lens_model_list"], packed["kwargs_lens"], strict=True):
        if profile != "MULTI_GAUSSIAN":
            rows.append(copy.deepcopy(values))
        else:
            for amp, sigma in zip(values["amp"], values["sigma"], strict=True):
                rows.append({"amp": float(amp), "sigma": float(sigma),
                             "center_x": values["center_x"], "center_y": values["center_y"]})
    result["kwargs_lens"] = rows
    return result


def native_proposal(proposal, space):
    from lensagent.workflow.pso import _proposal_from_best_fit

    return pack_multi_gaussian_components(_proposal_from_best_fit(proposal, space), space.model)
