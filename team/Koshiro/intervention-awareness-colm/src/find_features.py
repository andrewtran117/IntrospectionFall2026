import json

import hydra
from tqdm import tqdm

from trials import post, result_dir


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    exp = cfg.find_features
    out = result_dir(cfg, "find_features")

    # Search the auto-interp explanations of the steering SAE for each concept word.
    features = {}
    for concept in tqdm(cfg.concepts):
        body = {"modelId": cfg.model.id, "layers": [cfg.model.source], "query": concept}
        hits = post("/explanation/search", body)["results"]
        candidates = []
        for h in hits:
            if h["neuron"]["maxActApprox"] > 0 and int(h["index"]) not in [
                c["index"] for c in candidates
            ]:  # skip dead latents and repeats
                candidates.append(
                    {
                        "index": int(h["index"]),
                        "description": h["description"],
                        "cosine": h["cosine_similarity"],
                        "max_act": h["neuron"]["maxActApprox"],
                    }
                )
        # Identical labels tie on cosine; prefer the latent that fires most strongly.
        candidates.sort(key=lambda c: (-round(c["cosine"], 4), -c["max_act"]))
        candidates = candidates[: exp.n_candidates]
        index = cfg.model.overrides[concept] if concept in cfg.model.overrides else candidates[0]["index"]
        features[concept] = {"index": index, "candidates": candidates}
    (out / "features.json").write_text(json.dumps(features, indent=2))

    print("| concept | feature | description | other candidates |")
    print("|---|---|---|---|")
    for concept, f in features.items():
        desc = {c["index"]: c["description"] for c in f["candidates"]}
        others = "; ".join(f"{c['index']} {c['description']}" for c in f["candidates"] if c["index"] != f["index"])
        print(f"| {concept} | {f['index']} | {desc[f['index']]} | {others} |")


if __name__ == "__main__":
    main()
