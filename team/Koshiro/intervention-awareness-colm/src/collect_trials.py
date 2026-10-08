import json
import shutil

import hydra

from trials import collect, result_dir


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    exp = cfg.collect_trials
    out = result_dir(cfg, "collect_trials")
    for src, name in [(exp.features, "features.json"), (exp.alpha, "alpha.json")]:
        if not (out / name).exists():  # a resumed run keeps its inputs even if `latest` has moved
            shutil.copy(src, out / name)
    features = json.loads((out / "features.json").read_text())
    alpha = json.loads((out / "alpha.json").read_text())["alpha"]
    print(f"alpha: {alpha}")
    collect(cfg, "main", exp.task, [alpha], exp.n_trials, features, out / "trials.jsonl")


if __name__ == "__main__":
    main()
