import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from trials import parse, result_dir

CONDITIONS = ["control", "injection", "gaslight"]
CLAIMS = {"2way": ["control", "injection", "invalid"], "3way": ["control", "injection", "gaslight", "invalid"]}
LABELS = {"control": "Control", "injection": "Hidden Intervention", "gaslight": "Gaslight", "invalid": "Invalid"}


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    exp = cfg.analyze_trials
    out = result_dir(cfg, "analyze_trials")
    claims = CLAIMS[exp.task]
    df = pd.concat([pd.read_json(path, lines=True) for path in exp.trials], ignore_index=True)
    # Injection trials come from the main run only; unsteered trials from the alpha sweep are pooled in.
    df = df[(df.task == exp.task) & ((df.stage == "main") | (df.condition != "injection"))]
    print(df.groupby(["condition", "stage"]).size().to_markdown())

    matrices = {}
    for mode in ["strict", "loose", "loose_multi"]:
        df["claim"] = [parse(r, o, mode) for r, o in zip(df.response, df.order)]
        # Percentages per concept, then mean and population std over concepts, as in the paper.
        rates = pd.crosstab([df.condition, df.concept], df.claim, normalize="index")
        rates = rates.reindex(columns=claims, fill_value=0) * 100
        rates.to_csv(out / f"rates_{mode}.csv")
        mean = rates.groupby("condition").mean().reindex(CONDITIONS)
        std = rates.groupby("condition").std(ddof=0).reindex(CONDITIONS)
        mean.join(std, lsuffix="_mean", rsuffix="_std").to_csv(out / f"matrix_{mode}.csv")
        matrices[mode] = mean, std

        annot = mean.round(1).astype(str) + "\n±" + std.round(1).astype(str)
        ax = sns.heatmap(mean.rename(index=LABELS, columns=LABELS), annot=annot.values, fmt="", vmin=0, vmax=100)
        ax.set(xlabel="Model claim", ylabel="Actual condition")
        plt.savefig(out / f"heatmap_{mode}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # The paper's figures use loose parsing.
    mean, std = matrices["loose"]
    paper = pd.DataFrame(OmegaConf.to_container(cfg.model.paper[exp.task]), index=claims).T.reindex(CONDITIONS)
    comparison = pd.DataFrame({"ours": mean.stack(), "ours_std": std.stack(), "paper": paper.stack()})
    comparison["diff"] = comparison.ours - comparison.paper
    comparison.to_csv(out / "comparison.csv")
    print(comparison.round(1).to_markdown())


if __name__ == "__main__":
    main()
