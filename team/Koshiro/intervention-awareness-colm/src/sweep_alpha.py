import json
import shutil

import hydra
import pandas as pd

from trials import collect, parse, result_dir


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    exp = cfg.sweep_alpha
    out = result_dir(cfg, "sweep_alpha")
    if not (out / "features.json").exists():  # a resumed run keeps its inputs even if `latest` has moved
        shutil.copy(exp.features, out / "features.json")
    features = json.loads((out / "features.json").read_text())
    collect(cfg, "sweep", exp.task, list(exp.alphas), exp.n_trials, features, out / "trials.jsonl")

    # Pick the alpha whose injection trials are most often reported as injections (the paper's criterion).
    df = pd.read_json(out / "trials.jsonl", lines=True)
    df = df[df.condition == "injection"]
    df["claim"] = [parse(r, o, "loose") for r, o in zip(df.response, df.order)]
    rates = pd.crosstab([df.alpha, df.concept], df.claim, normalize="index")
    rates = rates.reindex(columns=["injection", "invalid"], fill_value=0) * 100
    table = rates.groupby("alpha").mean().join(rates.groupby("alpha").std(ddof=0), lsuffix="_mean", rsuffix="_std")
    table.to_csv(out / "sweep.csv")
    alpha = table.injection_mean.idxmax().item()
    (out / "alpha.json").write_text(json.dumps({"alpha": alpha}))
    print(table.round(1).to_markdown())
    print(f"best alpha: {alpha}")


if __name__ == "__main__":
    main()
