from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path('docs/report/assets')
OUT.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, title, body, fc):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                           linewidth=1.8, edgecolor='black', facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.35, title, fontsize=14, fontweight='bold', va='top', ha='left')
    ax.text(x + 0.18, y + h - 1.02, body, fontsize=12, va='top', ha='left', linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None, lx=None, ly=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.6, color='black'))
    if label:
        ax.text(lx if lx is not None else (x1+x2)/2, ly if ly is not None else (y1+y2)/2,
                label, fontsize=11, ha='center', va='center', bbox=dict(fc='white', ec='none', alpha=0.9))


# Figure 1: Architecture
fig, ax = plt.subplots(figsize=(15, 8), dpi=220)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

add_box(ax, 0.8, 5.1, 3.8, 2.7, 'Next.js Frontend', 'Upload, prompt input,\nlive event feed,\nartifact playback', '#efe9dd')
add_box(ax, 0.8, 1.5, 3.8, 2.9, 'WebSocket Bridge API', '/upload-video, /ws/foley,\nartifact URLs, event relay', '#efe9dd')
add_box(ax, 5.4, 3.0, 3.8, 2.5, 'FoleyOrchestrator', 'Event loop, retries,\nuncertainty controls,\ntrace persistence', '#e6e6f2')
add_box(ax, 5.4, 6.0, 3.8, 2.5, 'Planner / Controller', 'Event planning,\nprompt rewrite,\niteration policy', '#e6e6f2')

add_box(ax, 10.2, 6.0, 4.2, 2.5, 'Perception Endpoint', 'Video keyframes +\nVLM timeline extraction', '#e7f0e9')
add_box(ax, 10.2, 3.0, 4.2, 2.5, 'Execution Endpoint', 'Audio generation\n(AudioGen)', '#e7f0e9')
add_box(ax, 10.2, 0.2, 4.2, 2.5, 'Verification Endpoint', 'Dual CLAP scoring\nand agreement gap', '#e7f0e9')

arrow(ax, 2.7, 5.05, 2.7, 4.45, 'HTTP/WS request path', 4.6, 4.8)
arrow(ax, 4.62, 2.95, 5.35, 3.45, 'run request', 4.95, 3.95)
arrow(ax, 5.35, 3.6, 4.62, 2.35, 'typed event stream', 5.05, 2.7)
arrow(ax, 2.7, 4.45, 2.7, 5.05, 'renderable events', 1.2, 4.75)

arrow(ax, 7.3, 5.52, 7.3, 5.95)
arrow(ax, 9.22, 4.3, 10.15, 7.1)
arrow(ax, 9.22, 4.2, 10.15, 4.2)
arrow(ax, 9.22, 3.6, 10.15, 2.0)

fig.tight_layout()
fig.savefig(OUT / 'architecture_diagram.png', bbox_inches='tight')
plt.close(fig)

# Figure 2: Agent loop
fig, ax = plt.subplots(figsize=(9, 11), dpi=220)
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis('off')

add_box(ax, 3.25, 11.5, 3.5, 1.0, 'Start Event', '', '#e7f2f8')
add_box(ax, 3.25, 9.7, 3.5, 1.6, 'Generate Audio\nCandidate', '', '#e7f2f8')
add_box(ax, 3.25, 7.7, 3.5, 1.6, 'Dual CLAP\nVerification', '', '#e7f2f8')
add_box(ax, 3.25, 5.7, 3.5, 1.6, 'Cross-Modal\nAgreement Check', '', '#e7f2f8')

# diamond
diamond = plt.Polygon([[5.0,5.0],[2.8,3.9],[5.0,2.8],[7.2,3.9]], closed=True,
                      edgecolor='black', facecolor='#f2edd6', linewidth=1.8)
ax.add_patch(diamond)
ax.text(5.0, 3.9, 'Controller\nDecision', ha='center', va='center', fontsize=14, fontweight='bold')

add_box(ax, 0.7, 1.0, 2.8, 1.5, 'Accept\nCandidate', '', '#e7f2f8')
add_box(ax, 3.6, 1.0, 2.8, 1.7, 'Rewrite Prompt\nand Retry', '', '#e7f2f8')
add_box(ax, 6.5, 1.0, 2.8, 1.7, 'Stop and Use\nBest So Far', '', '#e7f2f8')

arrow(ax, 5.0, 11.45, 5.0, 11.3)
arrow(ax, 5.0, 9.65, 5.0, 9.3)
arrow(ax, 5.0, 7.65, 5.0, 7.3)
arrow(ax, 5.0, 5.65, 5.0, 5.05)

arrow(ax, 4.2, 3.3, 2.1, 2.55, 'ACCEPT', 3.0, 2.95)
arrow(ax, 5.0, 2.75, 5.0, 2.72, 'RETRY_REWRITE /\nRETRY_BEST', 6.25, 2.9)
arrow(ax, 5.8, 3.3, 7.9, 2.7, 'STOP_BEST', 7.1, 3.0)

ax.annotate('', xy=(6.7, 10.45), xytext=(6.4, 2.7),
            arrowprops=dict(arrowstyle='->', lw=1.6, connectionstyle='arc3,rad=0.25'))
ax.text(7.65, 6.3, 'next attempt', fontsize=11)

fig.tight_layout()
fig.savefig(OUT / 'agent_loop_diagram.png', bbox_inches='tight')
plt.close(fig)

print('generated:', OUT / 'architecture_diagram.png')
print('generated:', OUT / 'agent_loop_diagram.png')
