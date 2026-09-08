import matplotlib
import matplotlib.pyplot as plt

_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "svg", "ps", "cairo", "template"}


class FitnessPlotter:
    """Live-updating (and always-saved) plot of best/mean efficiency per generation.

    Call update() once per CMA-ES generation. Call reset_run() when starting a new
    CMA-ES run (e.g. a new trial/config in a batch) to archive the finished curve
    as a faded background line and start tracking a fresh one.
    """

    def __init__(self, path="fitness_curve.png", title="CMA-ES Optimization Progress"):
        self.path = path
        self.title = title
        self.gens = []
        self.best_eff = []
        self.mean_eff = []
        self.past_curves = []
        self.interactive = matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS
        if self.interactive:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))

    def reset_run(self, title=None):
        if self.gens:
            self.past_curves.append((list(self.gens), list(self.best_eff)))
        self.gens, self.best_eff, self.mean_eff = [], [], []
        if title is not None:
            self.title = title

    def update(self, gen, fitness_values):
        # fitness_values are -eff (CMA-ES minimizes), so negate back to efficiency
        efficiencies = [-f for f in fitness_values]
        self.gens.append(gen)
        self.best_eff.append(max(efficiencies))
        self.mean_eff.append(sum(efficiencies) / len(efficiencies))

        self.ax.clear()
        for gens, best in self.past_curves[-10:]:
            self.ax.plot(gens, best, color="gray", alpha=0.25, linewidth=1)
        self.ax.plot(self.gens, self.best_eff, marker="o", label="best efficiency")
        self.ax.plot(self.gens, self.mean_eff, marker=".", linestyle="--", label="mean efficiency")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Efficiency (higher = better)")
        self.ax.set_title(self.title)
        self.ax.legend(loc="lower right")
        self.ax.grid(True, linestyle=":", alpha=0.6)

        self.fig.savefig(self.path, dpi=150)
        if self.interactive:
            try:
                self.fig.canvas.draw()
                plt.pause(0.001)
            except Exception:
                self.interactive = False

    def close(self):
        plt.close(self.fig)
