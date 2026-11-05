"""Native (tkinter) live viewer for a run: learned strategy vs optimal and how it evolved, plus the Count and
Bankroll tabs when the run directory holds count.json or bankroll.json.

Usage: python -m bjrl.ui.app --run runs/main   (or --run runs/count)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

POLL_MS = 1500
ACT = ["S", "H", "D", "P"]
ACT_NAME = ["Stand", "Hit", "Double", "Split"]
KINDS = [
    ("hard", "Hard totals", "hard_rows", lambda r: str(r)),
    ("soft", "Soft totals", "soft_rows", lambda r: f"A,{r - 11}"),
    ("pair", "Pairs", "pair_rows", lambda r: "A,A" if r == 11 else f"{r},{r}"),
]
SPEEDS = {"slow": 4, "normal": 12, "fast": 30, "very fast": 80}

BG, PANEL, INK, MUTED, LINE, GRID = "#f6f7f9", "#ffffff", "#1b1f27", "#5d6675", "#d9dde4", "#e8ebf0"
ACT_FILL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
ACT_INK = ["#ffffff", INK, "#ffffff", INK]
MISS, ACCENT, SERIES2, TRAINING = "#d00000", "#0072B2", "#7a4fd0", "#009E73"


def fmt_count(n):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.3g}{suf}"
    return str(round(n))


def fmt_time(s):
    s = round(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {s % 3600 // 60}m"


ROWS = {"hard_rows": list(range(4, 22)), "soft_rows": list(range(12, 22)), "pair_rows": list(range(2, 12)),
        "dealer_cols": list(range(2, 12))}
RESOLVED, TIE, BUDGET = 1, 2, 3
STATUS_NAME = ["open", "resolved", "tie", "budget"]
METHOD_NAME = {"natural": "learned from natural play", "explore": "learned from exploring starts"}


def dealer_label(c):
    return "A" if c == 11 else str(c)


class RunFiles:
    """Incremental reader of <run>/meta.json and <run>/history.jsonl."""

    def __init__(self, run: Path):
        self.run = run
        self.meta = None
        self.snaps = []
        self._meta_mtime = None
        self._offset = 0

    def poll(self) -> bool:
        changed = False
        meta_path = self.run / "meta.json"
        try:
            mtime = meta_path.stat().st_mtime_ns
            if mtime != self._meta_mtime:
                self.meta = json.loads(meta_path.read_text())
                self._meta_mtime = mtime
                changed = True
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # absent or caught mid-write: try again next poll
        try:
            with (self.run / "history.jsonl").open("rb") as f:
                if os.fstat(f.fileno()).st_size < self._offset:
                    self.snaps, self._offset, changed = [], 0, True
                f.seek(self._offset)
                chunk = f.read()
        except FileNotFoundError:
            return changed
        # Only consume through the last newline so a torn final line is reread whole next time.
        end = chunk.rfind(b"\n")
        if end < 0:
            return changed
        for line in chunk[:end].split(b"\n"):
            if line.strip():
                try:
                    self.snaps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self._offset += end + 1
        return True


class StrategyCanvas(tk.Canvas):
    """Hard/Soft/Pairs charts of view(): {"policy", "reference", "status" or None, "cur", "ref", "match", "mark"}."""

    def __init__(self, master, view):
        super().__init__(master, bg=PANEL, highlightthickness=0, height=1)
        self.view = view
        self.cells = {}
        self.tables = []
        self.bind("<Configure>", lambda e: self.layout())
        self.bind("<Motion>", self.on_hover)
        self.bind("<Leave>", lambda e: self.delete("tip"))

    def layout(self):
        self.delete("all")
        self.cells, self.tables = {}, []
        W, H = self.winfo_width(), self.winfo_height()
        gap, label_w, spacing, title_h, head_h = 2, 40, 28, 22, 18
        by_w = (W - 2 * spacing - 3 * label_w - 16) / 30 - gap
        by_h = (H - title_h - head_h - 8) / len(ROWS["hard_rows"]) - gap
        c = max(10, min(by_w, by_h, 40))
        step = c + gap
        total_w = 3 * (label_w + 10 * step) + 2 * spacing
        x = max(8, (W - total_w) / 2)
        small = ("TkDefaultFont", max(9, int(c * 0.42)))
        cell_font = ("TkDefaultFont", max(9, int(c * 0.55)), "bold")
        for key, title, rows_key, label in KINDS:
            rows = ROWS[rows_key]
            x0, y0 = x + label_w, 4 + title_h + head_h
            self.create_text(x, 4, text=title, anchor="nw", fill=INK, font=("TkDefaultFont", 13, "bold"))
            self.create_text(x0 + 10 * step, 6, text="", anchor="ne", fill=MUTED, tags=f"match-{key}")
            self.create_text(x0 - 6, y0 - head_h / 2, text="vs", anchor="e", fill=MUTED, font=small)
            for ci, col in enumerate(ROWS["dealer_cols"]):
                self.create_text(x0 + ci * step + c / 2, y0 - head_h / 2, text=dealer_label(col), fill=MUTED, font=small)
            grid = []
            for ri, r in enumerate(rows):
                cy = y0 + ri * step
                self.create_text(x0 - 6, cy + c / 2, text=label(r), anchor="e", fill=MUTED, font=small)
                row = []
                for ci in range(10):
                    cx = x0 + ci * step
                    rect = self.create_rectangle(cx, cy, cx + c, cy + c, width=0, fill=GRID)
                    txt = self.create_text(cx + c / 2, cy + c / 2, text="", font=cell_font)
                    t, rw = c * 0.3, max(1.5, min(3, c * 0.12))
                    ring = self.create_rectangle(cx + rw / 2, cy + rw / 2, cx + c - rw / 2, cy + c - rw / 2,
                                                 outline=MISS, width=rw, state="hidden")
                    tri = self.create_polygon(cx + c - t, cy, cx + c, cy, cx + c, cy + t, fill=MISS, state="hidden")
                    d = max(4, c * 0.22)
                    dot = self.create_oval(cx + 2, cy + c - 2 - d, cx + 2 + d, cy + c - 2, fill=PANEL, outline=INK,
                                           width=1.5, state="hidden")
                    row.append((rect, txt, ring, tri, dot))
                grid.append(row)
            self.cells[key] = grid
            self.tables.append((key, title, rows_key, label, x0, y0, step, c, len(rows)))
            x += label_w + 10 * step + spacing
        self.render()

    def render(self):
        v = self.view()
        if not self.cells or not v:
            return
        for key, *_ in KINDS:
            st = v["status"][key] if v["status"] else None
            for ri, row in enumerate(self.cells[key]):
                for ci, (rect, txt, ring, tri, dot) in enumerate(row):
                    cur, best = v["policy"][key][ri][ci], v["reference"][key][ri][ci]
                    if cur < 0 or best < 0:
                        self.itemconfigure(rect, fill=GRID)
                        self.itemconfigure(txt, text="")
                        for item in (ring, tri, dot):
                            self.itemconfigure(item, state="hidden")
                        continue
                    wrong = "normal" if v["mark"] and cur != best else "hidden"
                    self.itemconfigure(rect, fill=ACT_FILL[cur])
                    self.itemconfigure(txt, text=ACT[cur], fill=ACT_INK[cur])
                    self.itemconfigure(ring, state=wrong)
                    self.itemconfigure(tri, state=wrong)
                    s = st[ri][ci] if st else RESOLVED
                    self.itemconfigure(dot, state="hidden" if s in (RESOLVED, -1) else "normal",
                                       fill=INK if s == BUDGET else PANEL)
            m = (v.get("match") or {}).get(key)
            self.itemconfigure(f"match-{key}", text="" if m is None else m if isinstance(m, str) else f"{m * 100:.1f}% match")

    def on_hover(self, e):
        self.delete("tip")
        v = self.view()
        if not v:
            return
        for key, title, rows_key, label, x0, y0, step, c, nrows in self.tables:
            ci, ri = int((e.x - x0) // step), int((e.y - y0) // step)
            if 0 <= ci < 10 and 0 <= ri < nrows:
                cur, best = v["policy"][key][ri][ci], v["reference"][key][ri][ci]
                if cur < 0 or best < 0:
                    return
                hand = label(ROWS[rows_key][ri])
                up = dealer_label(ROWS["dealer_cols"][ci])
                text = f"{title.split()[0]} {hand} vs {up}: {v['cur']} {ACT_NAME[cur]}, {v['ref']} {ACT_NAME[best]}"
                if cur != best:
                    text += " (differs)"
                st = v["status"][key][ri][ci] if v["status"] else None
                if st is not None and st < 0:
                    text += "; never a first decision in natural play (needs three cards), starting action kept"
                elif st is not None:
                    text += f"; {STATUS_NAME[st]}"
                    if "n" in v:
                        text += f", {fmt_count(v['n'][key][ri][ci])} samples"
                    if v.get("freq"):
                        text += f", {v['freq'][key][ri][ci] * 100:.4f}% of hands"
                self.show_tip(e.x, e.y, text)
                return

    def show_tip(self, x, y, text):
        show_tip(self, x, y, text)


def show_tip(canvas, x, y, text):
    t = canvas.create_text(0, 0, text=text, anchor="nw", fill=BG, tags="tip", justify="left")
    x1, y1, x2, y2 = canvas.bbox(t)
    w, h = x2 - x1 + 16, y2 - y1 + 10
    tx = min(x + 14, canvas.winfo_width() - w - 4)
    ty = y + 16 if y + 16 + h < canvas.winfo_height() else y - h - 8
    canvas.coords(t, tx + 8, ty + 5)
    canvas.tag_lower(canvas.create_rectangle(tx, ty, tx + w, ty + h, fill=INK, outline="", tags="tip"), t)


class LineChart(tk.Canvas):
    L, R, T, B = 58, 16, 54, 36

    def __init__(self, master, app, title, key, series, y_range, ticks, fmt, ref=None):
        super().__init__(master, bg=PANEL, highlightthickness=0, cursor="crosshair", height=1)
        self.app, self.title, self.key = app, title, key
        self.series, self.y_range, self.ticks, self.fmt, self.ref = series, y_range, ticks, fmt, ref
        self.scale = None
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Button-1>", self.on_click)

    def draw(self):
        self.delete("all")
        W, H = self.winfo_width(), self.winfo_height()
        L, R, T, B = self.L, self.R, self.T, self.B
        self.create_text(12, 8, text=self.title, anchor="nw", fill=INK, font=("TkDefaultFont", 13, "bold"))
        kx = 12
        for _, _, _, color, name in self.series:
            self.create_line(kx, 32, kx + 14, 32, fill=color, width=3)
            t = self.create_text(kx + 20, 32, text=name, anchor="w", fill=MUTED)
            kx = self.bbox(t)[2] + 16
        snaps = self.app.files.snaps
        if not snaps:
            self.scale = None
            return
        xs = [math.log10(max(1, s["hands"])) for s in snaps]
        x0, x1 = math.floor(xs[0]), math.ceil(xs[-1])
        x1 = max(x1, x0 + 1)
        y0, y1 = self.y_range(snaps)
        px = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
        py = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
        self.scale = (xs, x0, x1, px, py)
        for d in range(x0, x1 + 1):
            self.create_line(px(d), T, px(d), H - B, fill=GRID)
            self.create_text(px(d), H - B + 12, text=fmt_count(10 ** d), fill=MUTED)
        self.create_text((L + W - R) / 2, H - 6, text="hands played (log scale)", anchor="s", fill=MUTED)
        for v, lab in self.ticks(y0, y1):
            self.create_line(L, py(v), W - R, py(v), fill=GRID)
            self.create_text(L - 6, py(v), text=lab, anchor="e", fill=MUTED)
        self.create_line(L, H - B, W - R, H - B, fill=LINE)
        if self.ref is not None:
            self.create_line(L, py(self.ref), W - R, py(self.ref), fill=MUTED, dash=(4, 3))
        for get, f, lo, color, _ in self.series:
            run = []
            for x, s in zip(xs, snaps):
                v = get(s)
                if v > lo:
                    run += [px(x), py(f(v))]
                    continue
                if len(run) >= 4:
                    self.create_line(*run, fill=color, width=2, joinstyle="round")
                run = []
            if len(run) >= 4:
                self.create_line(*run, fill=color, width=2, joinstyle="round")
        self.draw_marker()

    def draw_marker(self):
        self.delete("marker")
        if not self.scale:
            return
        xs, _, _, px, py = self.scale
        W, H = self.winfo_width(), self.winfo_height()
        idx = self.app.idx
        s, x = self.app.files.snaps[idx], px(xs[idx])
        self.create_line(x, self.T, x, H - self.B, fill=INK, width=1.2, tags="marker")
        for get, f, lo, color, _ in self.series:
            v = get(s)
            if v > lo:
                y = py(f(v))
                self.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline=PANEL, width=1.5, tags="marker")
        right = x < W / 2
        # Put the label in the vertical half away from the data at the marker so it never covers the line.
        ys = [py(f(get(s))) for get, f, lo, _, _ in self.series if get(s) > lo]
        top = not ys or min(ys) > (self.T + H - self.B) / 2
        y = self.T + 4 if top else H - self.B - 4
        anchor = ("n" if top else "s") + ("w" if right else "e")
        t = self.create_text(x + (8 if right else -8), y, text=self.fmt(s), anchor=anchor,
                             fill=INK, font=("TkDefaultFont", 12, "bold"), tags="marker")
        bx = self.bbox(t)
        self.tag_lower(self.create_rectangle(bx[0] - 2, bx[1], bx[2] + 2, bx[3], fill=PANEL, outline="", tags="marker"), t)

    def on_click(self, e):
        if not self.scale:
            return
        xs, x0, x1, _, _ = self.scale
        xv = x0 + (e.x - self.L) / (self.winfo_width() - self.L - self.R) * (x1 - x0)
        self.app.select(min(range(len(xs)), key=lambda i: abs(xs[i] - xv)))


def match_range(snaps):
    lo = min(min(s["match"]["overall"], s["match"]["weighted"]) for s in snaps) * 100
    return max(0, math.floor(lo / 10) * 10), 100


def match_ticks(a, b):
    step = 20 if b - a > 60 else 10
    return [(v, f"{v}%") for v in range(math.ceil(a / step) * step, int(b) + 1, step)]


def regret_range(snaps):
    vs = [s["regret"] for s in snaps if s["regret"] > 0]
    if not vs:
        return -6, 0
    lo, hi = math.floor(math.log10(min(vs))), math.ceil(math.log10(max(vs)))
    return lo, max(hi, lo + 1)


class App:
    def __init__(self, root, run: Path, snap=None):
        self.root, self.files = root, RunFiles(run)
        self.idx, self.playing, self.play_job, self.pinned = 0, False, None, snap
        self.follow = tk.BooleanVar(value=snap is None)
        self.mark = tk.BooleanVar(value=True)
        self.speed = tk.StringVar(value="normal")
        root.title(f"Blackjack RL strategy: {run}")
        root.configure(bg=BG)
        root.geometry("1400x960")
        root.minsize(1040, 740)
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        self.train_tab = tab = tk.Frame(self.tabs, bg=BG)
        self.tabs.add(tab, text="Training")
        self.extra = {}

        head = tk.Frame(tab, bg=BG)
        head.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(head, text="Blackjack RL: learned strategy vs optimal", bg=BG, fg=INK,
                 font=("TkDefaultFont", 17, "bold")).pack(side="left", padx=(0, 16))
        self.status = tk.Label(head, text="waiting", bg=BG, fg=MUTED, font=("TkDefaultFont", 12, "bold"),
                               padx=10, highlightthickness=1, highlightbackground=LINE)
        self.status.pack(side="left", padx=(0, 20))
        self.stats = {}
        for key, caption in (("hands", "HANDS PLAYED"), ("rounds", "ROUNDS"), ("elapsed", "ELAPSED"),
                             ("rate", "HANDS / SEC"), ("eps", "EPSILON"), ("opt", "OPTIMAL EV / HAND")):
            box = tk.Frame(head, bg=BG)
            box.pack(side="left", padx=(0, 24))
            self.stats[key] = tk.Label(box, text="-", bg=BG, fg=INK, font=("TkDefaultFont", 15, "bold"))
            self.stats[key].pack(anchor="w")
            tk.Label(box, text=caption, bg=BG, fg=MUTED, font=("TkDefaultFont", 10)).pack(anchor="w")

        self.empty = tk.Label(tab, text="Waiting for first snapshot. Polling every 1.5 seconds.",
                              bg=PANEL, fg=MUTED, font=("TkDefaultFont", 14), pady=60)
        self.body = tk.Frame(tab, bg=BG)
        self.body.rowconfigure(0, weight=2, minsize=340)
        self.body.rowconfigure(3, weight=1, minsize=190)
        self.body.columnconfigure(0, weight=1)
        self.body.columnconfigure(1, weight=1)

        self.strategy = StrategyCanvas(self.body, self.training_view)
        self.strategy.grid(row=0, column=0, columnspan=2, sticky="nsew")
        legend = tk.Frame(self.body, bg=PANEL)
        legend.grid(row=1, column=0, columnspan=2, sticky="ew")
        for i, name in enumerate(ACT_NAME):
            tk.Label(legend, text=ACT[i], bg=ACT_FILL[i], fg=ACT_INK[i], width=2,
                     font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=(12 if i == 0 else 4, 2), pady=6)
            tk.Label(legend, text=name, bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 10))
        ring_swatch(legend).pack(side="left", padx=(4, 4))
        tk.Label(legend, text="differs from optimal", bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 14))
        tk.Label(legend, text="Rows: your hand. Columns: dealer upcard. Grey: cannot occur.",
                 bg=PANEL, fg=MUTED).pack(side="left")

        controls = tk.Frame(self.body, bg=PANEL)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.play_btn = tk.Button(controls, text="Play", width=6, command=self.toggle_play, highlightbackground=PANEL)
        self.play_btn.pack(side="left", padx=(12, 6), pady=8)
        speed = tk.OptionMenu(controls, self.speed, *SPEEDS, command=lambda _: self.playing and self.start_play())
        speed.configure(bg=PANEL, fg=INK, activeforeground=INK, highlightthickness=0, width=8)
        speed.pack(side="left", padx=(0, 10))
        self.scrub = tk.Scale(controls, from_=0, to=0, orient="horizontal", showvalue=False, command=self.on_scrub,
                              bg=PANEL, highlightthickness=0, troughcolor=GRID, sliderlength=18, width=14, bd=0)
        # Pack squeezes the last-packed widget first, so the slider goes last and absorbs a narrow window.
        self.info = tk.Label(controls, text="", bg=PANEL, fg=MUTED, width=40, anchor="w")
        self.info.pack(side="right", padx=(0, 12))
        tk.Checkbutton(controls, text="mark differences", variable=self.mark, command=self.strategy.render,
                       bg=PANEL, fg=INK).pack(side="right", padx=(0, 10))
        tk.Checkbutton(controls, text="follow live", variable=self.follow, command=self.on_follow,
                       bg=PANEL, fg=INK).pack(side="right", padx=(0, 8))
        self.scrub.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.charts = [
            LineChart(self.body, self, "Cells matching optimal strategy", "match",
                      [(lambda s: s["match"]["overall"], lambda v: v * 100, -1, ACCENT, "overall"),
                       (lambda s: s["match"]["weighted"], lambda v: v * 100, -1, SERIES2,
                        "weighted by how often each cell occurs")],
                      match_range, match_ticks,
                      lambda s: f"{s['match']['overall'] * 100:.1f}% / {s['match']['weighted'] * 100:.1f}% weighted",
                      ref=100),
            LineChart(self.body, self, "Regret per hand vs optimal (log scale)", "regret",
                      [(lambda s: s["regret"], math.log10, 0, MISS, "expected profit lost per hand, in bets")],
                      regret_range,
                      lambda a, b: [(v, f"1e{v}") for v in range(math.ceil(a), int(b) + 1)],
                      lambda s: f"{s['regret']:.2e}" if s["regret"] > 0 else "0"),
        ]
        self.charts[0].grid(row=3, column=0, sticky="nsew", padx=(0, 6))
        self.charts[1].grid(row=3, column=1, sticky="nsew", padx=(6, 0))
        self.notes = tk.Label(self.body, text="", bg=BG, fg=MUTED, anchor="w", justify="left")
        self.notes.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.notes.bind("<Configure>", lambda e: self.notes.configure(wraplength=e.width))

        self.shown = None
        self.poll()

    def training_view(self):
        files = self.files
        if not files.meta or not files.snaps:
            return None
        snap = files.snaps[self.idx]
        return {"policy": snap["policy"], "reference": files.meta["optimal"], "status": None, "cur": "learned",
                "ref": "optimal", "match": snap.get("match"), "mark": self.mark.get()}

    def poll_extra(self):
        """Add the Count and Bankroll tabs once their files exist, and reload them when the file or one of its
        optional companions (passed to load by keyword) changes."""
        for name, cls, title, companions in (("count.json", CountTab, "Count", {}),
                                             ("bankroll.json", BankrollTab, "Bankroll", {"sweep": "bankroll_sweep.json"})):
            files = [name, *companions.values()]
            mtime = tuple(p.stat().st_mtime_ns if p.exists() else None for p in (self.files.run / f for f in files))
            if mtime[0] is None:
                continue
            tab = self.extra.get(name)
            if tab and tab.mtime == mtime:
                continue
            try:
                data = json.loads((self.files.run / name).read_text())
                extra = {k: json.loads((self.files.run / f).read_text()) if t else None
                         for (k, f), t in zip(companions.items(), mtime[1:])}
            except json.JSONDecodeError:
                continue
            if tab is None:
                tab = self.extra[name] = cls(self.tabs, self)
                self.tabs.add(tab, text=title)
            tab.mtime = mtime
            tab.load(data, **extra)
        if self.extra and self.files.meta is None and str(self.train_tab) in self.tabs.tabs():
            self.tabs.hide(self.train_tab)
        elif self.files.meta is not None:
            self.tabs.add(self.train_tab)

    def poll(self):
        self.poll_extra()
        files = self.files
        had_meta = files.meta is not None
        changed = files.poll()
        have = files.meta is not None and bool(files.snaps)
        if have != self.shown:
            self.shown = have
            if have:
                self.empty.pack_forget()
                self.body.pack(fill="both", expand=True, padx=16, pady=(0, 12))
            else:
                self.body.pack_forget()
                self.empty.pack(fill="x", padx=16, pady=16)
        if changed:
            if have:
                last = len(files.snaps) - 1
                if self.pinned is not None:
                    self.idx, self.pinned = min(self.pinned, last), None
                elif self.follow.get() and not self.playing:
                    self.idx = last
                self.idx = min(self.idx, last)
                self.scrub.configure(to=last)
                if not had_meta:
                    self.strategy.layout()
                for c in self.charts:
                    c.draw()
                self.render()
            self.render_status()
        self.root.after(POLL_MS, self.poll)

    def render_status(self):
        snaps, meta = self.files.snaps, self.files.meta
        done = bool(meta and meta.get("done"))
        if not snaps:
            self.status.configure(text="waiting", fg=MUTED, highlightbackground=LINE)
            return
        self.status.configure(text="done" if done else "training", fg=ACCENT if done else TRAINING,
                              highlightbackground=ACCENT if done else TRAINING)
        last = snaps[-1]
        rate = fmt_count(last["hands"] / last["elapsed_s"]) if last["elapsed_s"] > 0 else "-"
        values = {"hands": f"{last['hands']:,}", "rounds": f"{last['round']:,}", "elapsed": fmt_time(last["elapsed_s"]),
                  "rate": rate, "eps": f"{last['eps']:.3f}"}
        if meta and meta.get("optimal_ev") is not None:
            values["opt"] = f"{meta['optimal_ev'] * 100:.2f}%"
        for k, v in values.items():
            self.stats[k].configure(text=v)

    def render(self):
        snaps, meta = self.files.snaps, self.files.meta
        s = snaps[self.idx]
        self.strategy.render()
        for c in self.charts:
            c.draw_marker()
        self.scrub.set(self.idx)
        wrong = total = 0
        for key, *_ in KINDS:
            for cur_row, opt_row in zip(s["policy"][key], meta["optimal"][key]):
                for cur, best in zip(cur_row, opt_row):
                    if cur >= 0 and best >= 0:
                        total += 1
                        wrong += cur != best
        self.info.configure(text=f"snapshot {self.idx + 1}/{len(snaps)}: {s['hands']:,} hands, "
                                 f"{wrong}/{total} cells differ")
        self.notes.configure(text=(
            f"Selected snapshot ({s.get('phase', 'main')}, round {s['round']:,}): match {s['match']['overall'] * 100:.1f}% overall, "
            f"{s['match']['weighted'] * 100:.1f}% weighted by frequency; regret {s['regret']:.2e} bets/hand; "
            f"largest EV loss in a wrong cell {s['max_cell_gap']:.4f}; epsilon {s['eps']:.3f}. "
            "Wrong cells are often near ties, so regret is the headline number."))

    def select(self, i):
        self.follow.set(False)
        self.pause()
        self.idx = i
        self.render()

    def on_scrub(self, value):
        # Scale also fires this after render() sets it, so only a real move counts as user scrubbing.
        i = int(float(value))
        if self.files.snaps and i != self.idx:
            self.select(i)

    def on_follow(self):
        if self.follow.get() and self.files.snaps:
            self.pause()
            self.idx = len(self.files.snaps) - 1
            self.render()

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.files.snaps:
            if self.idx >= len(self.files.snaps) - 1:
                self.idx = 0
            self.follow.set(False)
            self.start_play()

    def start_play(self):
        self.playing = True
        self.play_btn.configure(text="Pause")
        if self.play_job:
            self.root.after_cancel(self.play_job)
        self.play_job = self.root.after(0, self.step)

    def step(self):
        if self.idx >= len(self.files.snaps) - 1:
            self.pause()
            return
        self.idx += 1
        self.render()
        self.play_job = self.root.after(int(1000 / SPEEDS[self.speed.get()]), self.step)

    def pause(self):
        self.playing = False
        self.play_btn.configure(text="Play")
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None


# Strategy identity colors: the reference palette's categorical slots in fixed order. Only the first three validate
# all-pairs; the fourth is used in faceted panels and cards that also carry the strategy's name, never color alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TC_MIN, TC_MAX = -6, 6
TCS = list(range(TC_MIN, TC_MAX + 1))
MONO = ("Menlo", 11)


def ring_swatch(master):
    c = tk.Canvas(master, width=18, height=18, bg=PANEL, highlightthickness=0)
    c.create_rectangle(2, 2, 17, 17, fill=GRID, outline=MISS, width=2.5)
    c.create_polygon(12, 1, 18, 1, 18, 7, fill=MISS)
    return c


def tint(color, a):
    """color at opacity a over the white panel (tkinter has no alpha)."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 + (c - 255) * a):02x}" for c in rgb)


def tc_text(t):
    return f"<={t:+d}" if t == TC_MIN else f">={t:+d}" if t == TC_MAX else f"{t:+d}"


def nice_ticks(lo, hi, n=5):
    span = max(hi - lo, 1e-12)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def fmt_money(v, cents=False, sign=False):
    s = "-" if v < 0 else ("+" if sign and v > 0 else "")
    v = abs(v)
    if cents:
        return f"{s}${v:,.2f}"
    return f"{s}${v / 1000:,.0f}K" if v >= 10000 else f"{s}${v:,.0f}"


def wrapped_label(master):
    lab = tk.Label(master, text="", bg=BG, fg=MUTED, anchor="w", justify="left")
    lab.pack(fill="x", padx=16, pady=(0, 8))
    lab.bind("<Configure>", lambda e: lab.configure(wraplength=e.width))
    return lab


def text_panel(master, title):
    frame = tk.Frame(master, bg=PANEL)
    tk.Label(frame, text=title, bg=PANEL, fg=INK, font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=12, pady=(8, 4))
    body = tk.Frame(frame, bg=PANEL)
    body.pack(fill="both", expand=True, padx=(12, 4), pady=(0, 8))
    text = tk.Text(body, bg=PANEL, fg=INK, font=MONO, relief="flat", highlightthickness=0, wrap="none", height=1,
                   cursor="arrow")
    bar = tk.Scrollbar(body, command=text.yview)
    text.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    text.tag_configure("muted", foreground=MUTED)
    text.tag_configure("head", foreground=MUTED, font=(MONO[0], MONO[1], "bold"))
    text.tag_configure("active", foreground=INK, font=(MONO[0], MONO[1], "bold"))
    text.tag_configure("miss", foreground=MISS, font=(MONO[0], MONO[1], "bold"))
    return frame, text


def set_text(text, lines):
    """lines: list of [(chunk, tag), ...]."""
    y = text.yview()[0]
    text.configure(state="normal")
    text.delete("1.0", "end")
    for line in lines:
        for chunk, tag in line:
            text.insert("end", chunk, tag)
        text.insert("end", "\n")
    text.configure(state="disabled")
    text.yview_moveto(y)


class EdgeChart(tk.Canvas):
    """Flat-bet edge per start-of-hand true count, count play vs basic play, with +-1 SE whiskers."""
    L, R, T, B = 64, 16, 58, 40

    def __init__(self, master, tab):
        super().__init__(master, bg=PANEL, highlightthickness=0, height=1, cursor="crosshair")
        self.tab = tab
        self.scale = None
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Motion>", self.on_hover)
        self.bind("<Leave>", lambda e: self.delete("tip"))
        self.bind("<Button-1>", self.on_click)

    def series(self):
        e = self.tab.data["edge"]
        return [("basic play", SERIES[0], e["basic"]), ("count-indexed play", SERIES[2], e["count"])]

    def draw(self):
        self.delete("all")
        if not self.tab.data:
            return
        W, H = self.winfo_width(), self.winfo_height()
        L, R, T, B = self.L, self.R, self.T, self.B
        self.create_text(12, 8, text="Edge per hand by true count (flat bet)", anchor="nw", fill=INK,
                         font=("TkDefaultFont", 13, "bold"))
        kx = 12
        for name, color, _ in self.series():
            self.create_line(kx, 36, kx + 14, 36, fill=color, width=3)
            t = self.create_text(kx + 20, 36, text=name, anchor="w", fill=MUTED)
            kx = self.bbox(t)[2] + 16
        vals = [v * 100 + s * 100 * k for _, _, e in self.series() for v, s in zip(e["edge"], e["se"]) for k in (-1, 1)]
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = (hi - lo) * 0.06
        lo, hi = lo - pad, hi + pad
        px = lambda i: L + (i + 0.5) / len(TCS) * (W - L - R)
        py = lambda v: T + (1 - (v - lo) / (hi - lo)) * (H - T - B)
        self.scale = (px, py)
        sel = self.tab.bucket()
        half = (W - L - R) / len(TCS) / 2
        self.create_rectangle(px(sel) - half, T, px(sel) + half, H - B, fill=BG, outline="")
        for v in nice_ticks(lo, hi):
            self.create_line(L, py(v), W - R, py(v), fill=GRID)
            self.create_text(L - 6, py(v), text=f"{v:+.1f}%" if v else "0%", anchor="e", fill=MUTED)
        self.create_line(L, py(0), W - R, py(0), fill=LINE)
        for i, t in enumerate(TCS):
            self.create_text(px(i), H - B + 12, text=tc_text(t), fill=INK if i == sel else MUTED)
        self.create_text((L + W - R) / 2, H - 6, text="true count at the start of the hand", anchor="s", fill=MUTED)
        for _, color, e in self.series():
            pts = []
            for i, (v, s) in enumerate(zip(e["edge"], e["se"])):
                x, y = px(i), py(v * 100)
                self.create_line(x, py((v - s) * 100), x, py((v + s) * 100), fill=color, width=1)
                pts += [x, y]
            self.create_line(*pts, fill=color, width=2, joinstyle="round", capstyle="round")
            for i in range(0, len(pts), 2):
                self.create_oval(pts[i] - 4, pts[i + 1] - 4, pts[i] + 4, pts[i + 1] + 4, fill=color, outline=PANEL, width=2)

    def nearest(self, x):
        W = self.winfo_width()
        return min(len(TCS) - 1, max(0, int((x - self.L) / (W - self.L - self.R) * len(TCS))))

    def on_hover(self, e):
        self.delete("tip")
        if not self.scale:
            return
        i = self.nearest(e.x)
        lines = [f"TC {tc_text(TCS[i])}: {self.tab.data['edge']['count']['freq'][i] * 100:.2f}% of hands"]
        for name, _, ed in self.series():
            lines.append(f"{name}: {ed['edge'][i] * 100:+.3f}% +- {ed['se'][i] * 100:.3f}%")
        lines.append(f"variance per hand {self.tab.data['edge']['count']['var'][i]:.3f}")
        show_tip(self, e.x, e.y, "\n".join(lines))

    def on_click(self, e):
        if self.scale:
            self.tab.select(self.nearest(e.x))


class CountTab(tk.Frame):
    """How the strategy changes with the Hi-Lo true count: one chart per TC bucket, index plays, edge per count."""

    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app, self.data, self.mtime = app, None, None
        self.tc = tk.IntVar(value=0)
        self.playing, self.job = False, None
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(head, text="Card counting: strategy by true count", bg=BG, fg=INK,
                 font=("TkDefaultFont", 17, "bold")).pack(side="left", padx=(0, 12))
        self.method = tk.Label(head, text="", bg=PANEL, fg=ACCENT, font=("TkDefaultFont", 12, "bold"), padx=8, pady=2)
        self.method.pack(side="left", padx=(0, 16))
        self.summary = wrapped_label(self)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        body.rowconfigure(0, weight=3, minsize=330)
        body.rowconfigure(3, weight=2, minsize=230)
        for c in range(3):
            body.columnconfigure(c, weight=1, uniform="bottom")
        self.strategy = StrategyCanvas(body, self.view)
        self.strategy.grid(row=0, column=0, columnspan=3, sticky="nsew")
        legend = tk.Frame(body, bg=PANEL)
        legend.grid(row=1, column=0, columnspan=3, sticky="ew")
        for i, name in enumerate(ACT_NAME):
            tk.Label(legend, text=ACT[i], bg=ACT_FILL[i], fg=ACT_INK[i], width=2,
                     font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=(12 if i == 0 else 4, 2), pady=6)
            tk.Label(legend, text=name, bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 10))
        ring_swatch(legend).pack(side="left", padx=(4, 4))
        tk.Label(legend, text="differs from basic strategy", bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 14))
        tk.Label(legend, text="○", bg=PANEL, fg=INK, font=("TkDefaultFont", 13)).pack(side="left")
        tk.Label(legend, text="tie (within 0.001): earlier action kept", bg=PANEL, fg=MUTED).pack(side="left",
                                                                                                  padx=(2, 14))
        tk.Label(legend, text="●", bg=PANEL, fg=INK, font=("TkDefaultFont", 13)).pack(side="left")
        tk.Label(legend, text="sample budget ran out: basic play", bg=PANEL, fg=MUTED).pack(side="left", padx=(2, 0))

        controls = tk.Frame(body, bg=PANEL)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        self.play_btn = tk.Button(controls, text="Sweep", width=6, command=self.toggle, highlightbackground=PANEL)
        self.play_btn.pack(side="left", padx=(12, 10), pady=8)
        tk.Label(controls, text="true count  -6", bg=PANEL, fg=MUTED).pack(side="left")
        self.info = tk.Label(controls, text="", bg=PANEL, fg=INK, width=58, anchor="w", font=("TkDefaultFont", 12, "bold"))
        self.info.pack(side="right", padx=(0, 12))
        self.scale = tk.Scale(controls, from_=TC_MIN, to=TC_MAX, orient="horizontal", variable=self.tc, resolution=1,
                              showvalue=True, command=lambda _: self.render(), bg=PANEL, highlightthickness=0,
                              troughcolor=GRID, sliderlength=22, width=14, bd=0, tickinterval=1, length=420)
        tk.Label(controls, text="+6", bg=PANEL, fg=MUTED).pack(side="right", padx=(0, 16))
        self.scale.pack(side="left", fill="x", expand=True, padx=(8, 4))

        idx, self.index_text = text_panel(body, "Index plays (resolved deviations)")
        idx.grid(row=3, column=0, sticky="nsew", padx=(0, 6))
        ref, self.ref_text = text_panel(body, "Illustrious 18 check (Hi-Lo, multi-deck S17)")
        ref.grid(row=3, column=1, sticky="nsew", padx=6)
        self.edge = EdgeChart(body, self)
        self.edge.grid(row=3, column=2, sticky="nsew", padx=(6, 0))

    def bucket(self):
        return self.tc.get() - TC_MIN

    def view(self):
        if not self.data:
            return None
        b = self.bucket()
        return {"policy": self.data["policy"][b], "reference": self.data["basic"], "status": self.data["status"][b],
                "n": self.data["n"][b], "freq": self.data["freq"][b] if "freq" in self.data else None,
                "cur": f"TC {tc_text(TCS[b])}", "ref": "basic", "match": None, "mark": True}

    def load(self, data):
        self.data = data
        sc = data["status_counts"]
        method = data.get("method", "explore")
        self.method.configure(text=METHOD_NAME[method])
        pairs = (f"{sc['resolved']:,} of {sum(sc.values()):,} (count, cell) pairs resolved, {sc['tie']:,} tie, "
                 f"{sc['budget']:,} out of budget")
        if method == "natural":
            share = sum(e["freq"] for e in data["budget_cells"]) * 100
            text = (f"{data['hands']:,} natural hands with {data['samples']:,} paired first decisions in "
                    f"{fmt_time(data['elapsed_s'])}; {pairs} at {fmt_count(data['budget_hands'])} hands, "
                    f"{share:.2f}% of all hands. ")
        else:
            text = (f"{data['samples']:,} exploring-start samples in {fmt_time(data['elapsed_s'])}; {pairs} at "
                    f"{fmt_count(data['max_per_cell'])} samples per pair. ")
        self.summary.configure(text=text + "Decisions use the true count at the start of the hand.")
        self.strategy.layout()
        self.render()

    def render(self):
        if not self.data:
            return
        d, b = self.data, self.bucket()
        tc = TCS[b]
        self.strategy.render()
        self.edge.draw()
        changed = sum(1 for k in ("hard", "soft", "pair") for r1, r0, rs in
                      zip(d["policy"][b][k], d["basic"][k], d["status"][b][k]) for a, a0, s in zip(r1, r0, rs)
                      if a >= 0 and a0 >= 0 and a != a0 and s == RESOLVED)
        e = d["edge"]["count"]
        self.info.configure(text=f"TC {tc_text(tc)}: {changed} cells deviate, {e['freq'][b] * 100:.1f}% of hands, "
                                 f"edge {e['edge'][b] * 100:+.2f}%")
        lines = []
        for x in d["index"]:
            on = tc >= x["tc"] if x["side"] == "above" else tc <= x["tc"]
            note = f"  (unresolved {', '.join(f'{u:+d}' for u in x['unresolved'])})" if x["unresolved"] else ""
            lines.append([(("▶ " if on else "  ") + x["text"], "active" if on else "muted"), (note, "muted")])
        if not lines:
            lines = [[("no resolved deviation from basic strategy", "muted")]]
        set_text(self.index_text, lines)
        ref = [[(f"{'play':<10}{'switch':>8}{'ref':>5}{'found':>7}{'band':>12}  ok", "head")]]
        for r in d["reference"]:
            m = r["measured"]
            found = "never" if m > TC_MAX else f"{m:+d}"
            band = f"[{r['band'][0]:+d},{'never' if r['band'][1] > TC_MAX else format(r['band'][1], '+d')}]"
            ref.append([(f"{r['play']:<10}{r['low'] + '>' + r['high']:>8}{r['reference']:>+5d}{found:>7}{band:>12}  ", None),
                        ("yes" if r["ok"] else "NO", None if r["ok"] else "miss")])
        n_ok = sum(r["ok"] for r in d["reference"])
        ref.append([(f"{n_ok}/{len(d['reference'])} within +-2 of the published index", "muted")])
        set_text(self.ref_text, ref)

    def select(self, i):
        self.pause()
        self.tc.set(TCS[i])
        self.render()

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.playing = True
            self.play_btn.configure(text="Pause")
            if self.tc.get() >= TC_MAX:
                self.tc.set(TC_MIN)
            self.step()

    def step(self):
        self.render()
        if self.tc.get() >= TC_MAX:
            self.pause()
            return
        self.job = self.after(900, self.advance)

    def advance(self):
        self.tc.set(self.tc.get() + 1)
        self.step()

    def pause(self):
        self.playing = False
        self.play_btn.configure(text="Sweep")
        if self.job:
            self.after_cancel(self.job)
            self.job = None


class FanChart(tk.Canvas):
    """Small multiples, one panel per strategy on a shared y-scale: 5-95% and 25-75% bankroll bands and the median."""
    L, R, T, B, GAP = 70, 16, 64, 42, 18

    def __init__(self, master, tab):
        super().__init__(master, bg=PANEL, highlightthickness=0, height=1, cursor="crosshair")
        self.tab = tab
        self.geom = None
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<Motion>", self.on_hover)
        self.bind("<Leave>", lambda e: (self.delete("tip"), self.delete("cross")))

    def draw(self):
        self.delete("all")
        data = self.tab.data
        if not data:
            return
        W, H = self.winfo_width(), self.winfo_height()
        L, R, T, B, G = self.L, self.R, self.T, self.B, self.GAP
        strats = data["strategies"]
        self.create_text(12, 8, text=f"Bankroll over time, {data['players']:,} players each starting at "
                                     f"{fmt_money(data['bankroll'])}", anchor="nw", fill=INK, font=("TkDefaultFont", 13, "bold"))
        kx = 12
        for sw, name in ((tint(MUTED, 0.18), "5 to 95% of players"), (tint(MUTED, 0.42), "25 to 75%"), (MUTED, "median")):
            if name == "median":
                self.create_line(kx, 36, kx + 16, 36, fill=sw, width=2)
            else:
                self.create_rectangle(kx, 30, kx + 16, 42, fill=sw, outline="")
            t = self.create_text(kx + 22, 36, text=name, anchor="w", fill=MUTED)
            kx = self.bbox(t)[2] + 16
        logx = self.tab.logx.get()
        hours = strats[0]["hours"]
        x0 = math.log10(hours[0]) if logx else 0
        x1 = math.log10(hours[-1]) if logx else hours[-1]
        lo = min(min(s["percentiles"]["5"]) for s in strats)
        hi = max(max(s["percentiles"]["95"]) for s in strats)
        lo, hi = min(lo, 0), hi + (hi - lo) * 0.04
        pw = (W - L - R - G * (len(strats) - 1)) / len(strats)
        py = lambda v: T + (1 - (v - lo) / (hi - lo)) * (H - T - B)
        ticks = nice_ticks(lo, hi)
        panels = []
        for k, s in enumerate(strats):
            left = L + k * (pw + G)
            px = (lambda left: lambda h: left + ((math.log10(h) if logx else h) - x0) / (x1 - x0) * pw)(left)
            panels.append((left, px))
            color = SERIES[k % len(SERIES)]
            self.create_rectangle(left, T - 14, left + 14, T - 4, fill=color, outline="")
            self.create_text(left + 20, T - 9, text=s["label"], anchor="w", fill=INK, font=("TkDefaultFont", 12, "bold"))
            for v in ticks:
                self.create_line(left, py(v), left + pw, py(v), fill=GRID)
                if k == 0:
                    self.create_text(L - 6, py(v), text=fmt_money(v), anchor="e", fill=MUTED)
            for h in (nice_hours_log(hours[0], hours[-1]) if logx else nice_ticks(0, hours[-1], 4)):
                if hours[0] <= h <= hours[-1] or (not logx and h == 0):
                    x = px(max(h, hours[0]) if logx else h)
                    self.create_line(x, T, x, H - B, fill=GRID)
                    anchor = "nw" if x - left < 20 else "ne" if left + pw - x < 20 else "n"
                    self.create_text(x, H - B + 5, text=f"{h:,.0f}h", fill=MUTED, anchor=anchor)
            self.create_line(left, H - B, left + pw, H - B, fill=LINE)
            start = data["bankroll"]
            self.create_line(left, py(start), left + pw, py(start), fill=MUTED)
            p = s["percentiles"]
            hs = [0.0] + hours if not logx else hours
            def path(key, hs=hs, p=p):
                vals = ([start] if not logx else []) + p[key]
                return [(px(h) if logx or h > 0 else left, py(v)) for h, v in zip(hs, vals)]
            for a, b_, alpha in (("5", "95", 0.14), ("25", "75", 0.32)):
                pts = path(a) + path(b_)[::-1]
                self.create_polygon(*[c for xy in pts for c in xy], fill=tint(color, alpha), outline="")
            med = path("50")
            self.create_line(*[c for xy in med for c in xy], fill=color, width=2, joinstyle="round", capstyle="round")
            ex, ey = med[-1]
            self.create_oval(ex - 4, ey - 4, ex + 4, ey + 4, fill=color, outline=PANEL, width=2)
            lab = self.create_text(ex - 8, ey - 10, text=f"median {fmt_money(p['50'][-1])}", anchor="se", fill=INK,
                                   font=("TkDefaultFont", 11, "bold"))
            bx = self.bbox(lab)
            self.tag_lower(self.create_rectangle(bx[0] - 2, bx[1], bx[2] + 2, bx[3], fill=PANEL, outline=""), lab)
        self.create_text(L, H - 6, text="hours played at 100 hands per hour" + (" (log scale)" if logx else "") +
                         "; grey line: starting bankroll", anchor="sw", fill=MUTED)
        self.geom = (panels, pw, py)

    def on_hover(self, e):
        self.delete("tip")
        self.delete("cross")
        if not self.geom:
            return
        panels, pw, py = self.geom
        strats = self.tab.data["strategies"]
        for k, (left, px) in enumerate(panels):
            if left <= e.x <= left + pw:
                hours = strats[k]["hours"]
                i = min(range(len(hours)), key=lambda j: abs(px(hours[j]) - e.x))
                H = self.winfo_height()
                for kk, (l2, px2) in enumerate(panels):
                    x = px2(hours[i])
                    self.create_line(x, self.T, x, H - self.B, fill=INK, tags="cross")
                    y = py(strats[kk]["percentiles"]["50"][i])
                    self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=SERIES[kk % len(SERIES)], outline=PANEL, width=2,
                                     tags="cross")
                s = strats[k]
                p = s["percentiles"]
                lines = [f"{s['label']} after {hours[i]:,.0f} hours ({s['checkpoints'][i]:,} hands)"]
                for key, name in (("95", "95th pct"), ("75", "75th pct"), ("50", "median"), ("25", "25th pct"), ("5", "5th pct")):
                    lines.append(f"{name:<9} {fmt_money(p[key][i]):>9}")
                lines.append(f"{'mean':<9} {fmt_money(s['mean'][i]):>9}")
                lines.append(f"ruined so far {s['ruin_curve'][i] * 100:.1f}%")
                show_tip(self, e.x, e.y, "\n".join(lines))
                return


def nice_hours_log(a, b):
    return [h for h in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if a <= h <= b]


class BankrollTab(tk.Frame):
    """Bankroll simulation: percentile bands per strategy and a summary card per strategy."""

    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app, self.data, self.mtime = app, None, None
        self.logx = tk.BooleanVar(value=False)
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(head, text="Bankroll: betting and playing by the count", bg=BG, fg=INK,
                 font=("TkDefaultFont", 17, "bold")).pack(side="left", padx=(0, 16))
        tk.Checkbutton(head, text="log time axis", variable=self.logx, command=lambda: self.chart.draw(),
                       bg=BG, fg=INK).pack(side="right")
        self.settings = wrapped_label(self)
        self.cards = tk.Frame(self, bg=BG)
        self.cards.pack(fill="x", padx=16, pady=(0, 10))
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(side="bottom", fill="x", padx=16, pady=(8, 12))
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=2)
        self.sweep_frame, self.sweep_text = text_panel(bottom, "Sweep")
        self.sweep_title = self.sweep_frame.winfo_children()[0]
        self.note = tk.Label(bottom, text="", bg=BG, fg=MUTED, anchor="nw", justify="left")
        self.note.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        self.note.bind("<Configure>", lambda e: self.note.configure(wraplength=e.width))
        self.chart = FanChart(self, self)
        self.chart.pack(fill="both", expand=True, padx=16)

    def load(self, data, sweep=None):
        self.data = data
        sit = data.get("sit_out")
        self.settings.configure(text=(
            f"{data['kelly']:g} Kelly from the edge-per-count table, ${data['table_min']:,.0f} to ${data['table_max']:,.0f} "
            f"table, $5 chips, bet at most 1/4 of bankroll; the sit-out strategy skips rounds below TC "
            f"{'(none)' if sit is None else format(sit, '+d')} but keeps counting them, and they still take time; "
            f"{data['hands']:,} hands per player; simulated in {fmt_time(data['elapsed_s'])}"))
        for w in self.cards.winfo_children():
            w.destroy()
        last = len(data["strategies"]) - 1
        for k, s in enumerate(data["strategies"]):
            card = tk.Frame(self.cards, bg=PANEL, highlightthickness=0)
            card.pack(side="left", fill="both", expand=True, padx=(0 if k == 0 else 6, 0 if k == last else 6))
            top = tk.Frame(card, bg=PANEL)
            top.pack(fill="x", padx=12, pady=(10, 4))
            tk.Frame(top, bg=SERIES[k % len(SERIES)], width=14, height=14).pack(side="left", padx=(0, 8))
            tk.Label(top, text=s["label"], bg=PANEL, fg=INK, font=("TkDefaultFont", 12, "bold")).pack(side="left")
            grid = tk.Frame(card, bg=PANEL)
            grid.pack(fill="x", padx=12, pady=(0, 10))
            played = s.get("frac_played", s.get("hands_bet", 0) / max(s["hands_played"], 1))
            stats = [(fmt_money(s["median_final"]), "MEDIAN FINAL"), (f"{s['ruin'] * 100:.1f}%", "P(RUIN)"),
                     (fmt_money(s["win_per_hour"], cents=True, sign=True), "PER HOUR"),
                     (f"{s['pct_action']:+.2f}%", "OF ACTION"), (fmt_money(s["avg_bet"], cents=True), "AVG BET"),
                     (f"{played * 100:.0f}%", "ROUNDS PLAYED")]
            for i, (v, cap) in enumerate(stats):
                r, c = 2 * (i // 3), i % 3
                tk.Label(grid, text=v, bg=PANEL, fg=INK, font=("TkDefaultFont", 14, "bold")).grid(row=r, column=c, sticky="w", padx=(0, 14))
                tk.Label(grid, text=cap, bg=PANEL, fg=MUTED, font=("TkDefaultFont", 9)).grid(row=r + 1, column=c, sticky="w", padx=(0, 14), pady=(0, 4))
        cc = [f"{s['label']}: measured {s['cross_check']['measured_pct_action']:+.3f}% of action vs "
              f"{s['cross_check']['predicted_pct_action']:+.3f}% predicted (z {s['cross_check']['z']:+.1f})"
              for s in data["strategies"]]
        self.note.configure(text="Win-rate cross-check against the edge table.\n" + "\n".join(cc) + "\n\nRuin means the "
                                 "bankroll fell below the table minimum; a ruined player stops playing and keeps that bankroll.")
        self.load_sweep(sweep)
        self.chart.draw()

    def load_sweep(self, sweep):
        if not sweep:
            self.sweep_frame.grid_forget()
            return
        self.sweep_frame.grid(row=0, column=0, sticky="nsew")
        rows = sweep["rows"]
        h = rows[0]["hours"]
        self.sweep_title.configure(text=f"Sweep: count-indexed play by Kelly fraction and sit-out threshold "
                                        f"({sweep['players']:,} players, bankroll at {h:,.0f} h)")
        sit = lambda t: "off" if t is None else f"{t:+d}"
        head = f"{'Kelly':>5}  {'sit out':>7}  {'per hour':>9}  {'median':>8}  {'p5':>8}  {'P(ruin)':>7}  {'avg bet':>8}  {'played':>6}"
        lines = [[(head, "head")]]
        cur = (self.data["kelly"], self.data.get("sit_out"))
        for r in rows:
            tag = "active" if (r["kelly"], r["sit_out"]) == cur else None
            lines.append([(f"{r['kelly']:>5.2f}  {sit(r['sit_out']):>7}  {fmt_money(r['win_per_hour'], cents=True, sign=True):>9}  "
                           f"{'$' + format(r['median'], ',.0f'):>8}  {'$' + format(r['p5'], ',.0f'):>8}  {r['ruin'] * 100:>6.1f}%  "
                           f"{fmt_money(r['avg_bet'], cents=True):>8}  {r['frac_played'] * 100:>5.0f}%", tag)])
        self.sweep_text.configure(height=len(lines))
        set_text(self.sweep_text, lines)


def screenshot(root, path, delay_ms):
    def capture():
        root.update()
        x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
        r = subprocess.run(["screencapture", "-x", "-R", f"{x},{y},{w},{h}", path])
        root.destroy()
        sys.exit(r.returncode)

    root.lift()
    root.attributes("-topmost", True)
    root.after(delay_ms, capture)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--snap", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--screenshot", help=argparse.SUPPRESS)
    ap.add_argument("--screenshot-delay", type=int, default=3000, help=argparse.SUPPRESS)
    ap.add_argument("--tab", choices=("training", "count", "bankroll"), help="tab to open on")
    ap.add_argument("--tc", type=int, help="true count to show on the Count tab")
    a = ap.parse_args()
    root = tk.Tk()
    app = App(root, Path(a.run), snap=a.snap)
    if a.tab:
        tab = app.train_tab if a.tab == "training" else app.extra.get(f"{a.tab}.json")
        if tab is not None:
            app.tabs.select(tab)
    if a.tc is not None and "count.json" in app.extra:
        app.extra["count.json"].tc.set(a.tc)
        app.extra["count.json"].render()
    if a.screenshot:
        screenshot(root, a.screenshot, a.screenshot_delay)
    root.mainloop()


if __name__ == "__main__":
    main()
