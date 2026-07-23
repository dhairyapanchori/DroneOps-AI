"""
DroneOps AI — RL-Focused Enterprise Training Dashboard
======================================================
PySide6 + pyqtgraph implementation designed for RL engineers.

Features:
- Telemetry hooking for True Episode Replay (without modifying core code)
- Fading trajectory trails
- RL analytics: Reward Distribution, Success Trend
- Compact metrics for maximum chart space
"""

import sys
import os
import time
import datetime
import threading
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QFrame, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QPushButton, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QFont, QColor, QPainter, QPen
import pyqtgraph as pg

import utils.config as cfg

# ═══════════════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════════════

BG      = "#0b0f19"
PANEL   = "#111827"
CARD    = "#1a2332"
BORDER  = "#1e293b"
GREEN   = "#10b981"
BLUE    = "#3b82f6"
YELLOW  = "#f59e0b"
ORANGE  = "#f97316"
RED     = "#ef4444"
PURPLE  = "#8b5cf6"
CYAN    = "#06b6d4"
TEXT    = "#f1f5f9"
TEXT2   = "#94a3b8"
TEXT3   = "#64748b"

pg.setConfigOptions(antialias=True, background=PANEL, foreground=TEXT2)

DRONE_COLORS = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"
]

def _qss() -> str:
    return f"""
    QMainWindow {{ background: {BG}; }}
    QWidget {{ color: {TEXT}; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 11px; }}
    QFrame[role="panel"] {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
    }}
    QFrame[role="kpi"] {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    }}
    QFrame[role="card"] {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;
    }}
    QLabel {{ background: transparent; border: none; }}
    QPushButton[role="tab"], QPushButton[role="toggle"] {{
        background: transparent; border: none; border-bottom: 2px solid transparent;
        color: {TEXT3}; font-size: 11px; font-weight: 500; padding: 6px 14px;
    }}
    QPushButton[role="tab"]:hover, QPushButton[role="toggle"]:hover {{ color: {TEXT}; }}
    QPushButton[role="tab"][active="true"] {{
        color: {GREEN}; border-bottom: 2px solid {GREEN};
    }}
    QPushButton[role="toggle"][active="true"] {{
        color: {BLUE}; border-bottom: 2px solid {BLUE};
    }}
    QPushButton[role="status"] {{
        background: #064e3b; color: {GREEN}; border: 1px solid #065f46;
        border-radius: 4px; font-size: 11px; font-weight: 600; padding: 5px 14px;
    }}
    QTableWidget {{
        background: transparent; border: none; gridline-color: {BORDER}; font-size: 10px;
    }}
    QTableWidget::item {{ padding: 2px 4px; border-bottom: 1px solid {BORDER}; }}
    QHeaderView::section {{
        background: {CARD}; color: {TEXT3}; font-size: 9px; font-weight: 600;
        border: none; border-bottom: 1px solid {BORDER}; padding: 4px;
    }}
    QTextEdit {{
        background: transparent; border: none; color: {TEXT2};
        font-family: 'Consolas', 'Cascadia Code', monospace; font-size: 10px;
    }}
    QScrollBar:vertical {{ background: {PANEL}; width: 6px; border: none; }}
    QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; min-height: 20px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    """

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _panel(title: str = "") -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setProperty("role", "panel")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(6)
    if title:
        t = QLabel(title)
        t.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {TEXT}; letter-spacing: 0.5px;")
        lay.addWidget(t)
    return f, lay

def _label(text: str, size: int = 11, color: str = TEXT, bold: bool = False) -> QLabel:
    w = "700" if bold else "400"
    lbl = QLabel(str(text))
    lbl.setStyleSheet(f"font-size: {size}px; font-weight: {w}; color: {color};")
    return lbl

def _sparkline(color: str, height: int = 30) -> tuple[pg.PlotWidget, pg.PlotDataItem]:
    pw = pg.PlotWidget()
    pw.setFixedHeight(height)
    pw.setBackground("transparent")
    pw.hideAxis("bottom")
    pw.hideAxis("left")
    pw.getViewBox().setMouseEnabled(x=False, y=False)
    pw.getViewBox().setMenuEnabled(False)
    pw.setAntialiasing(True)
    curve = pw.plot(pen=pg.mkPen(color, width=1.5))
    return pw, curve


# ═══════════════════════════════════════════════════════════════════════
# Main Dashboard
# ═══════════════════════════════════════════════════════════════════════

class TrainingDashboard:
    def __init__(self, trainer):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.trainer = trainer
        self.start_time = time.time()
        self._last_ep_seen = 0

        # ── RL Histories ──────────────────────────────────────────────
        self.h_episodes          = []
        self.h_rewards           = []
        self.h_avg_rewards       = []
        self.h_best_line         = []
        self.h_critic_loss       = []
        self.h_actor_loss        = []
        self.h_alpha             = []
        self.h_alpha_loss        = []
        self.h_coordination      = []
        self.h_success_rate      = []
        self.h_buffer            = []

        # ── Telemetry Hooks for Episode Replay ────────────────────────
        self._live_telemetry = []
        self._prev_telemetry = []
        self._telemetry_lock = threading.Lock()
        self._playback_frame = 0
        self._replay_mode = "Live"  # "Live" or "Previous"
        self._hook_env()

        self._active_tab = "Reward"

        # ── Build UI ──────────────────────────────────────────────────
        self.win = QMainWindow()
        self.win.setWindowTitle("DroneOps AI — RL-Focused Training Dashboard")
        self.win.resize(1680, 960)
        self.win.setStyleSheet(_qss())

        root = QWidget()
        self.win.setCentralWidget(root)
        ml = QVBoxLayout(root)
        ml.setContentsMargins(10, 6, 10, 6)
        ml.setSpacing(8)

        # 1. Header
        ml.addWidget(self._build_header(), 0)

        # 2. KPIs (Simplified)
        ml.addWidget(self._build_kpi_row(), 0)

        # 3. Main Stage (Chart + Episode Replay)
        stage_w = QWidget()
        stage_l = QHBoxLayout(stage_w)
        stage_l.setContentsMargins(0, 0, 0, 0)
        stage_l.setSpacing(8)
        stage_l.addWidget(self._build_main_chart(), 55)
        stage_l.addWidget(self._build_episode_replay(), 45)
        ml.addWidget(stage_w, 6)

        # 4. Analytics & Metrics
        bot_w = QWidget()
        bot_l = QHBoxLayout(bot_w)
        bot_l.setContentsMargins(0, 0, 0, 0)
        bot_l.setSpacing(8)
        bot_l.addWidget(self._build_reward_dist(), 20)
        bot_l.addWidget(self._build_success_trend(), 20)
        bot_l.addWidget(self._build_compact_metrics(), 25)
        bot_l.addWidget(self._build_log_and_hyper(), 35)
        ml.addWidget(bot_w, 3)

        # ── Timers ────────────────────────────────────────────────────
        # Fast timer for smooth map playback/rendering (30 fps)
        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(self._render_map_frame)
        self._anim_timer.start(33)

        # Slow timer for heavy chart/metric updates (2 fps)
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self.update_step)
        self._poll_timer.start(500)

    # ══════════════════════════════════════════════════════════════════
    # Telemetry Hooking
    # ══════════════════════════════════════════════════════════════════

    def _hook_env(self):
        """Monkey-patch env to capture full trajectory without changing backend."""
        env = self.trainer.env
        orig_reset = env.reset
        orig_step = env.step

        def hooked_reset(*args, **kwargs):
            ret = orig_reset(*args, **kwargs)
            with self._telemetry_lock:
                if self._live_telemetry:
                    self._prev_telemetry = self._live_telemetry
                self._live_telemetry = []
                # Capture initial state
                self._capture_frame(env)
            return ret

        def hooked_step(*args, **kwargs):
            ret = orig_step(*args, **kwargs)
            with self._telemetry_lock:
                self._capture_frame(env)
            return ret

        env.reset = hooked_reset
        env.step = hooked_step

    def _capture_frame(self, env):
        frame = {
            "drones": [(d.pos.copy(), d.alive) for d in env.drones],
            "targets": env.targets.copy(),
            "obstacles": env.obstacles.copy(),
            "reached": set(env.targets_reached)
        }
        self._live_telemetry.append(frame)

    # ══════════════════════════════════════════════════════════════════
    # Section Builders
    # ══════════════════════════════════════════════════════════════════

    def _build_header(self) -> QWidget:
        bar = QFrame()
        bar.setProperty("role", "panel")
        bar.setFixedHeight(52)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(16, 4, 16, 4)
        hl.setSpacing(20)

        title = QLabel("DroneOps AI")
        title.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {GREEN};")
        hl.addWidget(title)
        hl.addWidget(_label("|", 14, TEXT3))
        hl.addWidget(_label("RL-FOCUSED TRAINING DASHBOARD", 12, TEXT, True))
        hl.addStretch()

        for attr in ("Episode", "Step", "Elapsed Time", "ETA"):
            col = QVBoxLayout()
            col.setSpacing(0)
            col.addWidget(_label(attr, 8, TEXT3, True), 0, Qt.AlignCenter)
            val = _label("—", 13, TEXT, True)
            col.addWidget(val, 0, Qt.AlignCenter)
            setattr(self, f"_hdr_{attr.replace(' ', '_').lower()}", val)
            w = QWidget()
            w.setLayout(col)
            w.setFixedWidth(110)
            hl.addWidget(w)

        self._status_badge = QPushButton("● Running")
        self._status_badge.setProperty("role", "status")
        hl.addWidget(self._status_badge)
        return bar

    def _build_kpi_row(self) -> QWidget:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        self._kpi_cards = {}
        defs = [
            ("avg_reward",   "AVERAGE REWARD",    GREEN),
            ("best_reward",  "BEST REWARD",       YELLOW),
            ("success_rate", "SUCCESS RATE (20)", CYAN),
            ("coord_score",  "COORD SCORE",       PURPLE),
            ("replay_buf",   "REPLAY BUFFER",     BLUE),
        ]
        for key, title, color in defs:
            card = QFrame()
            card.setProperty("role", "kpi")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 10, 14, 8)
            cl.setSpacing(2)

            top = QHBoxLayout()
            top.addWidget(_label(title, 9, TEXT3, True))
            top.addStretch()
            cl.addLayout(top)

            val = _label("—", 24, TEXT, True)
            cl.addWidget(val)

            spark_w, spark_c = _sparkline(color, 24)
            cl.addWidget(spark_w)
            
            sub = _label("", 8, TEXT3)
            cl.addWidget(sub)

            rl.addWidget(card)
            self._kpi_cards[key] = {"val": val, "sub": sub, "curve": spark_c}

        return row

    def _build_main_chart(self) -> QFrame:
        f, lay = _panel("TRAINING PERFORMANCE & CONVERGENCE")
        
        tabs_w = QWidget()
        tabs_l = QHBoxLayout(tabs_w)
        tabs_l.setContentsMargins(0, 0, 0, 0)
        tabs_l.setSpacing(0)
        self._chart_tabs = {}
        for name in ("Reward", "Actor & Critic Loss", "Entropy (Alpha)"):
            btn = QPushButton(name)
            btn.setProperty("role", "tab")
            btn.setProperty("active", name == "Reward")
            btn.clicked.connect(lambda checked, n=name: self._switch_chart_tab(n))
            tabs_l.addWidget(btn)
            self._chart_tabs[name] = btn
        tabs_l.addStretch()
        lay.addWidget(tabs_w)

        self._chart = pg.PlotWidget()
        self._chart.setBackground(PANEL)
        self._chart.showGrid(x=True, y=True, alpha=0.15)
        self._chart.setLabel("bottom", "Episode", color=TEXT3, **{"font-size": "9px"})
        self._chart.getAxis("bottom").setPen(pg.mkPen(BORDER))
        self._chart.getAxis("left").setPen(pg.mkPen(BORDER))
        self._chart.getAxis("bottom").setTextPen(pg.mkPen(TEXT3))
        self._chart.getAxis("left").setTextPen(pg.mkPen(TEXT3))
        self._chart.addLegend(offset=(10, 10))

        # We keep several lines ready and hide/show based on tab
        self._lines = {
            "r_raw": self._chart.plot(pen=pg.mkPen(BLUE, width=1, alpha=150), name="Raw Reward"),
            "r_avg": self._chart.plot(pen=pg.mkPen(GREEN, width=2.5), name="Avg Reward (50)"),
            "c_loss": self._chart.plot(pen=pg.mkPen(CYAN, width=2), name="Critic Loss"),
            "a_loss": self._chart.plot(pen=pg.mkPen(RED, width=2), name="Actor Loss"),
            "alpha": self._chart.plot(pen=pg.mkPen(YELLOW, width=2), name="Alpha"),
        }
        self._switch_chart_tab("Reward") # initialize visibility
        lay.addWidget(self._chart)
        return f

    def _build_episode_replay(self) -> QFrame:
        f, lay = _panel("EPISODE REPLAY")

        # Top toggles
        top = QHBoxLayout()
        self._tog_live = QPushButton("Live Episode")
        self._tog_live.setProperty("role", "toggle")
        self._tog_live.setProperty("active", True)
        self._tog_prev = QPushButton("Previous Episode")
        self._tog_prev.setProperty("role", "toggle")
        
        def _set_mode(m):
            self._replay_mode = m
            self._playback_frame = 0
            self._tog_live.setProperty("active", m == "Live")
            self._tog_prev.setProperty("active", m == "Previous")
            self._tog_live.style().unpolish(self._tog_live); self._tog_live.style().polish(self._tog_live)
            self._tog_prev.style().unpolish(self._tog_prev); self._tog_prev.style().polish(self._tog_prev)
            
        self._tog_live.clicked.connect(lambda: _set_mode("Live"))
        self._tog_prev.clicked.connect(lambda: _set_mode("Previous"))
        
        top.addWidget(self._tog_live)
        top.addWidget(self._tog_prev)
        top.addStretch()
        self._frame_lbl = _label("Frame: 0", 10, TEXT3)
        top.addWidget(self._frame_lbl)
        lay.addLayout(top)

        # Map
        self._map = pg.PlotWidget()
        self._map.setBackground(BG)
        self._map.setAspectLocked(True)
        self._map.setXRange(-13, 13)
        self._map.setYRange(-13, 13)
        self._map.hideAxis("bottom")
        self._map.hideAxis("left")
        self._map.getViewBox().setMouseEnabled(x=False, y=False)
        self._map.getViewBox().setMenuEnabled(False)

        # Trails
        self._trails = []
        for i in range(6):
            c = pg.mkPen(DRONE_COLORS[i], width=2, style=Qt.DotLine)
            curve = self._map.plot(pen=c)
            self._trails.append(curve)

        self._map_drones = pg.ScatterPlotItem(size=14, pen=pg.mkPen(None))
        self._map_targets = pg.ScatterPlotItem(size=16, pen=pg.mkPen(None), symbol="star")
        self._map_obstacles = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None), symbol="x")
        self._map_base = pg.ScatterPlotItem(size=18, pen=pg.mkPen(None), symbol="t")
        
        self._map.addItem(self._map_targets)
        self._map.addItem(self._map_obstacles)
        self._map.addItem(self._map_base)
        self._map.addItem(self._map_drones)
        lay.addWidget(self._map)
        return f

    def _build_reward_dist(self) -> QFrame:
        f, lay = _panel("REWARD DISTRIBUTION (LAST 100)")
        self._hist = pg.PlotWidget()
        self._hist.setBackground(PANEL)
        self._hist.hideAxis("left")
        self._hist.getAxis("bottom").setPen(pg.mkPen(BORDER))
        self._hist.getAxis("bottom").setTextPen(pg.mkPen(TEXT3))
        self._hist.getViewBox().setMouseEnabled(x=False, y=False)
        self._hist_bg = pg.BarGraphItem(x=[], height=[], width=1, brush=BLUE)
        self._hist.addItem(self._hist_bg)
        lay.addWidget(self._hist)
        return f

    def _build_success_trend(self) -> QFrame:
        f, lay = _panel("COORDINATION & SUCCESS")
        self._trend = pg.PlotWidget()
        self._trend.setBackground(PANEL)
        self._trend.setYRange(0, 100)
        self._trend.getAxis("bottom").setPen(pg.mkPen(BORDER))
        self._trend.getAxis("left").setPen(pg.mkPen(BORDER))
        self._trend.getAxis("bottom").setTextPen(pg.mkPen(TEXT3))
        self._trend.getAxis("left").setTextPen(pg.mkPen(TEXT3))
        self._trend.addLegend(offset=(5, 5))
        self._line_succ = self._trend.plot(pen=pg.mkPen(CYAN, width=2), name="Success %")
        self._line_coord = self._trend.plot(pen=pg.mkPen(PURPLE, width=2), name="Coord x100")
        lay.addWidget(self._trend)
        return f

    def _build_compact_metrics(self) -> QFrame:
        f, lay = _panel("AI METRICS")
        grid = QGridLayout()
        grid.setSpacing(8)
        self._mcards = {}
        idx = 0
        for name, key, col in [
            ("Actor Loss", "a_loss", RED),
            ("Critic Loss", "c_loss", CYAN),
            ("Alpha", "alpha", YELLOW),
            ("Alpha Loss", "alpha_loss", ORANGE),
        ]:
            card = QFrame()
            card.setProperty("role", "card")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(8, 6, 8, 4)
            cl.setSpacing(2)
            cl.addWidget(_label(name, 9, TEXT3, True))
            val = _label("—", 14, TEXT, True)
            cl.addWidget(val)
            sw, sc = _sparkline(col, 20)
            cl.addWidget(sw)
            grid.addWidget(card, idx // 2, idx % 2)
            self._mcards[key] = (val, sc)
            idx += 1
        lay.addLayout(grid)
        return f

    def _build_log_and_hyper(self) -> QFrame:
        f = QFrame(); f.setProperty("role", "panel")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        
        # Hyperparameters (super compact)
        top = QHBoxLayout()
        top.addWidget(_label("HYPERPARAMS:", 9, TEXT3, True))
        self._hyp_lbls = []
        for p in ["LR_ACTOR", "GAMMA", "BATCH", "UPDATE_EVERY"]:
            v = getattr(cfg, p, getattr(self.trainer, p, "N/A"))
            top.addWidget(_label(f"{p}:", 9, TEXT2))
            top.addWidget(_label(f"{v}", 9, TEXT, True))
        top.addStretch()
        lay.addLayout(top)
        
        lay.addWidget(_label("LIVE LOG:", 9, TEXT3, True))
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        lay.addWidget(self._log_text)
        self._log_count = 0
        return f

    # ══════════════════════════════════════════════════════════════════
    # Logic & Updating
    # ══════════════════════════════════════════════════════════════════

    def _switch_chart_tab(self, name: str):
        self._active_tab = name
        for k, btn in self._chart_tabs.items():
            btn.setProperty("active", k == name)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        
        for k, line in self._lines.items():
            line.setVisible(False)
            
        if name == "Reward":
            self._lines["r_raw"].setVisible(True)
            self._lines["r_avg"].setVisible(True)
            self._chart.setLabel("left", "Reward")
        elif name == "Actor & Critic Loss":
            self._lines["c_loss"].setVisible(True)
            self._lines["a_loss"].setVisible(True)
            self._chart.setLabel("left", "Loss")
        elif name == "Entropy (Alpha)":
            self._lines["alpha"].setVisible(True)
            self._chart.setLabel("left", "Alpha")

    def _render_map_frame(self):
        """High-frequency playback of episode trajectory (30fps)."""
        with self._telemetry_lock:
            buf = self._live_telemetry if self._replay_mode == "Live" else self._prev_telemetry
            if not buf:
                return
                
            # If Live, always show the latest frame to reduce latency
            # If Previous, play back linearly
            if self._replay_mode == "Live":
                self._playback_frame = len(buf) - 1
            else:
                self._playback_frame = (self._playback_frame + 1) % len(buf)
                
            frame = buf[self._playback_frame]
            self._frame_lbl.setText(f"Frame: {self._playback_frame}/{len(buf)-1}")
            
            # Trails
            if len(buf) > 1:
                # Get history up to current frame for drawing trails
                hist = buf[:self._playback_frame+1]
                for i in range(6):
                    if i < len(frame["drones"]):
                        pts = [h["drones"][i][0] for h in hist]
                        x = [p[0] for p in pts]
                        y = [p[1] for p in pts]
                        self._trails[i].setData(x, y)
            else:
                for t in self._trails:
                    t.setData([], [])
                    
            # Drones
            d_pos = []
            d_brush = []
            for i, (pos, alive) in enumerate(frame["drones"]):
                d_pos.append(pos)
                col = DRONE_COLORS[i] if alive else "#475569" # dim if dead
                d_brush.append(pg.mkBrush(col))
                
            if d_pos:
                self._map_drones.setData(
                    pos=np.array(d_pos), brush=d_brush,
                    pen=pg.mkPen(BORDER, width=2), size=14, symbol="o"
                )
                
            # Env features
            tgts = frame.get("targets", [])
            reached = frame.get("reached", set())
            if len(tgts) > 0:
                t_brush = [pg.mkBrush(YELLOW if i in reached else GREEN) for i in range(len(tgts))]
                self._map_targets.setData(pos=np.array(tgts), brush=t_brush, size=16, symbol="star")
            
            obs = frame.get("obstacles", [])
            if len(obs) > 0:
                self._map_obstacles.setData(pos=np.array(obs), brush=pg.mkBrush(RED), pen=pg.mkPen(RED, width=1.5), size=12, symbol="x")
                
            self._map_base.setData(pos=np.array([[0,0]]), brush=pg.mkBrush(TEXT3), size=18, symbol="t")

    def update_step(self):
        """Low-frequency (2fps) update for charts and stats."""
        trainer = self.trainer
        ep = getattr(trainer, "current_ep", 0)
        metrics = getattr(trainer, "metrics", None)

        if metrics and hasattr(metrics, "rewards") and ep > self._last_ep_seen and len(metrics.rewards) > 0:
            self._last_ep_seen = ep
            self.h_episodes.append(ep)
            self.h_rewards.append(metrics.rewards[-1])
            self.h_avg_rewards.append(float(np.mean(self.h_rewards[-50:])))
            self.h_best_line.append(getattr(trainer, "best_ep_reward", 0))
            
            self.h_critic_loss.append(getattr(trainer, "last_c_loss", 0))
            self.h_actor_loss.append(getattr(trainer, "last_a_loss", 0))
            self.h_alpha.append(getattr(trainer, "last_alpha", 0))
            self.h_alpha_loss.append(getattr(trainer, "last_alpha_loss", 0))
            
            coord = metrics.coordination[-1] if hasattr(metrics, "coordination") and metrics.coordination else 0
            self.h_coordination.append(coord * 100) # scale to 100 for plot
            
            recent = self.h_rewards[-20:]
            succ_rate = sum(1 for r in recent if r > 10) / max(1, len(recent)) * 100
            self.h_success_rate.append(succ_rate)
            self.h_buffer.append(len(trainer.buf) if hasattr(trainer, "buf") else 0)

        # Update Header
        max_ep = getattr(cfg, "MAX_EPISODES", 500)
        total_steps = getattr(trainer, "total_steps", 0)
        elapsed = time.time() - self.start_time
        eta = (elapsed / max(1, ep)) * (max_ep - ep) if ep > 0 else 0
        
        self._hdr_episode.setText(f"{ep} / {max_ep}")
        self._hdr_step.setText(f"{total_steps:,}")
        self._hdr_elapsed_time.setText(str(datetime.timedelta(seconds=int(elapsed))))
        self._hdr_eta.setText(str(datetime.timedelta(seconds=int(eta))))
        
        is_training = getattr(trainer, "is_training", True)
        if not is_training:
            self._status_badge.setText("● Completed")
            self._status_badge.setStyleSheet("background: #1e1b4b; color: #8b5cf6;")

        # Update KPIs
        if self.h_rewards:
            self._kpi_cards["avg_reward"]["val"].setText(f"{self.h_avg_rewards[-1]:.1f}")
            self._kpi_cards["best_reward"]["val"].setText(f"{self.h_best_line[-1]:.1f}")
            self._kpi_cards["success_rate"]["val"].setText(f"{self.h_success_rate[-1]:.0f}%")
            self._kpi_cards["coord_score"]["val"].setText(f"{self.h_coordination[-1]/100:.2f}")
            self._kpi_cards["replay_buf"]["val"].setText(f"{self.h_buffer[-1]:,}")
            
            for key, hist in [
                ("avg_reward", self.h_avg_rewards), ("best_reward", self.h_best_line),
                ("success_rate", self.h_success_rate), ("coord_score", self.h_coordination),
                ("replay_buf", self.h_buffer)
            ]:
                if len(hist) > 1:
                    self._kpi_cards[key]["curve"].setData(hist[-60:])

        # Update Main Chart
        if self.h_rewards:
            eps = self.h_episodes
            self._lines["r_raw"].setData(eps, self.h_rewards)
            self._lines["r_avg"].setData(eps, self.h_avg_rewards)
            self._lines["c_loss"].setData(eps, self.h_critic_loss)
            self._lines["a_loss"].setData(eps, self.h_actor_loss)
            self._lines["alpha"].setData(eps, self.h_alpha)

        # Update Distribution (Hist)
        if len(self.h_rewards) > 0:
            recent_r = self.h_rewards[-100:]
            y, x = np.histogram(recent_r, bins=20)
            self._hist_bg.setOpts(x=x[:-1], height=y, width=(x[1]-x[0])*0.8)
            
        # Update Trend
        if self.h_success_rate:
            eps = self.h_episodes
            self._line_succ.setData(eps, self.h_success_rate)
            self._line_coord.setData(eps, self.h_coordination)

        # Update Compact Metrics
        self._mcards["a_loss"][0].setText(f"{getattr(trainer, 'last_a_loss', 0):.4f}")
        self._mcards["c_loss"][0].setText(f"{getattr(trainer, 'last_c_loss', 0):.4f}")
        self._mcards["alpha"][0].setText(f"{getattr(trainer, 'last_alpha', 0):.4f}")
        self._mcards["alpha_loss"][0].setText(f"{getattr(trainer, 'last_alpha_loss', 0):.4f}")
        if len(self.h_actor_loss) > 1:
            self._mcards["a_loss"][1].setData(self.h_actor_loss[-50:])
            self._mcards["c_loss"][1].setData(self.h_critic_loss[-50:])
            self._mcards["alpha"][1].setData(self.h_alpha[-50:])
            self._mcards["alpha_loss"][1].setData(self.h_alpha_loss[-50:])

        # Update Log
        events = getattr(trainer, "log_events", [])
        if len(events) > self._log_count:
            for line in events[self._log_count:]:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                col = GREEN if "Saved" in line else (YELLOW if "best" in line.lower() else (RED if "Fail" in line else BLUE))
                self._log_text.append(f'<span style="color:{col}">●</span> <span style="color:{TEXT3}">{ts}</span> <span style="color:{TEXT2}">{line}</span>')
            self._log_count = len(events)
            sb = self._log_text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def run(self):
        self.win.show()
        self.app.exec()
