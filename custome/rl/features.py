
import os
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def extractor():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = os.environ["RL_FEATURE_MODEL"]
    tok = AutoTokenizer.from_pretrained(name)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float32,
    ).cpu().eval()

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    return tok, model


@lru_cache(maxsize=2048)
def embed(text):
    import torch

    tok, model = extractor()
    inp = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(os.environ["RL_MAX_FEATURE_TOKENS"]),
    )

    with torch.inference_mode():
        h = model(
            **inp,
            output_hidden_states=True,
        ).hidden_states[-1][0, -1]

    return h.float().cpu().numpy()


def distances(reference_codes, candidate):
    refs = np.stack([
        embed(t) for t in reference_codes
    ]).astype("float64")

    y = embed(candidate).astype("float64")
    center = refs.mean(0)
    centered = refs - center

    if len(refs) < 2:
        raise ValueError("Need >=2 reference examples")

    _, _, vh = np.linalg.svd(
        centered, full_matrices=False
    )

    result, ref_pts, new_pts = {}, {}, {}

    for n in (1, 2, 3):
        k = min(n, len(refs) - 1, vh.shape[0])

        r = centered @ vh[:k].T
        p = (y - center) @ vh[:k].T

        r = np.pad(r, ((0, 0), (0, n - k)))
        p = np.pad(p, (0, n - k))

        d = float(
            np.linalg.norm(r - p, axis=1).min()
        )

        pair = np.linalg.norm(
            r[:, None, :] - r[None, :, :],
            axis=-1,
        )
        scale = float(np.median(
            pair[np.triu_indices(len(refs), 1)]
        ))
        scale = max(scale, 1e-12)

        result[f"d{n}"] = 1.0 / (1.0 + d / scale)
        ref_pts[n] = r.tolist()
        new_pts[n] = p.tolist()

    return result, ref_pts, new_pts
