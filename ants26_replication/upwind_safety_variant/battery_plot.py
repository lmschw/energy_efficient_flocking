"""Plot per-agent battery level over time, to check whether a controller is actually
balancing/optimizing battery usage across the swarm rather than draining some agents
much faster than others."""
import numpy as np
import matplotlib.pyplot as plt


def plot_battery_levels(battery_log, dt, out_path, title="Battery Level per Agent"):
    """battery_log: (n_steps, n_agents) array -- the "battery" entry of the telemetry
    dict returned by simulation_free_global_mod_2_LJ(..., record_battery=True)."""
    n_steps, n_agents = battery_log.shape
    time = np.arange(n_steps) * dt

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis", max(n_agents, 2))
    for a in range(n_agents):
        ax.plot(time, battery_log[:, a], color=cmap(a), linewidth=1, alpha=0.85)

    batt_min = battery_log.min(axis=1)
    batt_max = battery_log.max(axis=1)
    ax.fill_between(time, batt_min, batt_max, color="gray", alpha=0.15, label="min-max range")
    ax.plot(time, battery_log.mean(axis=1), color="black", linewidth=2.5, linestyle="--", label="mean")

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Battery level")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    final_spread = batt_max[-1] - batt_min[-1]
    print(f"Saved {out_path} -- final battery spread across agents: {final_spread:.2f} "
          f"(min={batt_min[-1]:.2f}, max={batt_max[-1]:.2f}, mean={battery_log[-1].mean():.2f})")
    return out_path
