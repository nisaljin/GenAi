from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path('docs/report/assets')
OUT.mkdir(parents=True, exist_ok=True)


def load_font(size=24, bold=False):
    candidates = [
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Supplemental/Helvetica Neue.ttc',
        '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def rounded_box(draw, box, fill):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline='black', width=3)


def draw_text_block(draw, box, title, body, title_font, body_font):
    x1, y1, x2, y2 = box
    draw.multiline_text((x1 + 26, y1 + 18), title, font=title_font, fill='black', spacing=6)
    if body:
        draw.multiline_text((x1 + 26, y1 + 86), body, font=body_font, fill='black', spacing=7)


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def arrow(draw, p1, p2, width=3):
    draw.line([p1, p2], fill='black', width=width)
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head_len = 18
    head_w = 8
    tip = (x2, y2)
    left = (x2 - int(head_len * ux) + int(head_w * px), y2 - int(head_len * uy) + int(head_w * py))
    right = (x2 - int(head_len * ux) - int(head_w * px), y2 - int(head_len * uy) - int(head_w * py))
    draw.polygon([tip, left, right], fill='black')


def edge_label(draw, xy, text, font):
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    pad = 6
    draw.rectangle((x - pad, y - pad, x + (r - l) + pad, y + (b - t) + pad), fill='white')
    draw.text((x, y), text, font=font, fill='black')


# -----------------------------
# Architecture diagram
# -----------------------------
W, H = 2800, 1500
img = Image.new('RGB', (W, H), '#f4f4f4')
d = ImageDraw.Draw(img)

f_title = load_font(72, True)
f_node_title = load_font(56, True)
f_node_body = load_font(42, False)
f_edge = load_font(34, False)

d.text((80, 44), 'Agentic Multimodal System Architecture', font=f_title, fill='black')

frontend = (80, 230, 760, 620)
ws = (80, 700, 760, 1190)
orch = (1040, 620, 1760, 1140)
planner = (1040, 230, 1760, 520)
per = (2120, 230, 2740, 520)
exe = (2120, 660, 2740, 900)
ver = (2120, 980, 2740, 1390)

for box, col in [
    (frontend, '#eee7db'),
    (ws, '#eee7db'),
    (orch, '#e7e7f3'),
    (planner, '#e7e7f3'),
    (per, '#e7efe7'),
    (exe, '#e7efe7'),
    (ver, '#e7efe7'),
]:
    rounded_box(d, box, col)

draw_text_block(d, frontend, 'Next.js Frontend', 'Upload, prompt input,\nlive event feed,\nartifact playback', f_node_title, f_node_body)
draw_text_block(d, ws, 'WebSocket Bridge API', '/upload-video, /ws/foley,\nartifact URLs, event relay', f_node_title, f_node_body)
draw_text_block(d, orch, 'FoleyOrchestrator', 'Event loop, retries,\nuncertainty controls,\ntrace persistence', f_node_title, f_node_body)
draw_text_block(d, planner, 'Planner / Controller', 'Event planning, prompt rewrite,\niteration policy', f_node_title, f_node_body)
draw_text_block(d, per, 'Perception Endpoint', 'Video keyframes +\nVLM timeline extraction', f_node_title, f_node_body)
draw_text_block(d, exe, 'Execution Endpoint', 'Audio generation\n(AudioGen)', f_node_title, f_node_body)
draw_text_block(d, ver, 'Verification Endpoint', 'Dual CLAP scoring\nand agreement gap', f_node_title, f_node_body)

# Frontend <-> WS vertical
fx, fy = center(frontend)
wx, wy = center(ws)
arrow(d, (fx, frontend[3]), (wx, ws[1]))
arrow(d, (wx, ws[1]), (fx, frontend[3]))
edge_label(d, (570, 640), 'HTTP/WS request path', f_edge)
edge_label(d, (22, 690), 'renderable events', f_edge)

# WS <-> Orchestrator
arrow(d, (ws[2], wy - 50), (orch[0], center(orch)[1] - 110))
arrow(d, (orch[0], center(orch)[1] - 5), (ws[2], wy + 70))
edge_label(d, (780, 820), 'run request', f_edge)
edge_label(d, (780, 940), 'typed event stream', f_edge)

# Orchestrator -> Planner + endpoints
ox, oy = center(orch)
px, py = center(planner)
arrow(d, (ox, orch[1]), (px, planner[3]))
arrow(d, (orch[2], oy - 130), (per[0], center(per)[1]))
arrow(d, (orch[2], oy), (exe[0], center(exe)[1]))
arrow(d, (orch[2], oy + 120), (ver[0], center(ver)[1]))

img.save(OUT / 'architecture_diagram.png')


# -----------------------------
# Agent loop diagram
# -----------------------------
W2, H2 = 2400, 3200
img2 = Image.new('RGB', (W2, H2), '#f4f4f4')
d2 = ImageDraw.Draw(img2)

d2.text((80, 48), 'Per-Event Agentic Decision Loop', font=f_title, fill='black')

f_node_title2 = load_font(56, True)
f_node_body2 = load_font(32, False)
f_edge2 = load_font(34, False)

start = (760, 220, 1640, 480)
gen = (760, 600, 1640, 980)
verif = (760, 1100, 1640, 1480)
xmod = (760, 1600, 1640, 1980)
accept = (140, 2620, 820, 3050)
rewrite = (860, 2620, 1540, 3180)
stop = (1580, 2620, 2260, 3050)
callout = (1680, 1080, 2340, 1560)

for box in [start, gen, verif, xmod, accept, rewrite, stop]:
    rounded_box(d2, box, '#e6f0f6')
rounded_box(d2, callout, '#eef6ea')

# decision diamond
diamond = [(1200, 2080), (840, 2280), (1200, 2480), (1560, 2280)]
d2.polygon(diamond, fill='#f2edd6', outline='black', width=3)

draw_text_block(
    d2, start, 'Start Event',
    'Input: planned event and\nretry budget.',
    f_node_title2, f_node_body2
)
draw_text_block(
    d2, gen, 'Generate Audio\nCandidate',
    'Action: synthesize WAV from\ncurrent prompt and duration.',
    f_node_title2, f_node_body2
)
draw_text_block(
    d2, verif, 'Dual CLAP\nVerification',
    'Output: score_primary,\nscore_secondary, score_gap.',
    f_node_title2, f_node_body2
)
draw_text_block(
    d2, xmod, 'Cross-Modal\nAgreement Check',
    'Compares prompt tokens with\nscene-derived expected cues.',
    f_node_title2, f_node_body2
)
draw_text_block(
    d2, accept, 'Accept\nCandidate',
    'Commit selected audio\nfor this event.',
    f_node_title2, f_node_body2
)
draw_text_block(
    d2, rewrite, 'Rewrite Prompt\nand Retry',
    'Use controller-suggested\nor refined next prompt.',
    f_node_title2, f_node_body2
)
draw_text_block(
    d2, stop, 'Stop and Use\nBest So Far',
    'Terminate retries and keep\nhighest-quality candidate.',
    f_node_title2, f_node_body2
)
d2.multiline_text((1030, 2195), 'Controller\nDecision', font=f_node_title2, fill='black', spacing=8)
draw_text_block(
    d2, callout, 'Uncertainty Gate',
    'If verifier disagreement or\ncross-modal mismatch is high,\nACCEPT can be blocked and\nforced to retry/stop-best.',
    load_font(48, True), f_node_body2
)

# vertical main flow
sx, _ = center(start)
gx, _ = center(gen)
vx, _ = center(verif)
xx, _ = center(xmod)
arrow(d2, (sx, start[3]), (gx, gen[1]))
arrow(d2, (gx, gen[3]), (vx, verif[1]))
arrow(d2, (vx, verif[3]), (xx, xmod[1]))
arrow(d2, (xx, xmod[3]), (1200, 2080))

# branches from diamond
arrow(d2, (980, 2360), (820, 2710))
arrow(d2, (1200, 2480), (1200, 2620))
arrow(d2, (1420, 2360), (1580, 2710))

edge_label(d2, (640, 2520), 'ACCEPT', f_edge2)
edge_label(d2, (1265, 2560), 'RETRY_REWRITE / RETRY_BEST', f_edge2)
edge_label(d2, (1500, 2520), 'STOP_BEST', f_edge2)

# retry loop (rewrite -> back to generate), routed on far right
p1 = (1540, 2900)
p2 = (1880, 2900)
p3 = (1880, 790)
p4 = (1640, 790)
d2.line([p1, p2, p3, p4], fill='black', width=3)
arrow(d2, p3, p4)
edge_label(d2, (1910, 1800), 'next attempt', f_edge2)

img2.save(OUT / 'agent_loop_diagram.png')

print('generated', OUT / 'architecture_diagram.png')
print('generated', OUT / 'agent_loop_diagram.png')
