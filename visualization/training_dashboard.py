"""
DroneOps AI — Enterprise Training Dashboard
=============================================
PySide6 + pyqtgraph implementation for real-time training monitoring.

Architecture:
    main.py creates TrainingDashboard(trainer) and calls dashboard.run().
    A QTimer polls trainer attributes every 500ms to update all widgets.
    Training runs in a daemon thread — the dashboard never writes to trainer.
"""

import sys
import os
import time
import datetime
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QFrame, QSizePolicy, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QPushButton, QProgressBar, QSpacerItem, QAbstractItemView,
)
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QPainterPath
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


def _qss() -> str:
    """Global Qt stylesheet."""
    return f"""
    QMainWindow {{ background: {BG}; }}
    QWidget {{ color: {TEXT}; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 11px; }}
    QFrame[role="panel"] {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
    }}
    QFrame[role="card"] {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    }}
    QFrame[role="kpi"] {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    }}
    QLabel {{ background: transparent; border: none; }}
    QPushButton[role="tab"] {{
        background: transparent; border: none; border-bottom: 2px solid transparent;
        color: {TEXT3}; font-size: 11px; font-weight: 500; padding: 6px 14px;
    }}
    QPushButton[role="tab"]:hover {{ color: {TEXT}; }}
    QPushButton[role="tab"][active="true"] {{
        color: {GREEN}; border-bottom: 2px solid {GREEN};
    }}
    QPushButton[role="status"] {{
        background: #064e3b; color: {GREEN}; border: 1px solid #065f46;
        border-radius: 4px; font-size: 11px; font-weight: 600; padding: 5px 14px;
    }}
    QProgressBar {{
        background: {BORDER}; border: none; border-radius: 3px; max-height: 6px;
    }}
    QProgressBar::chunk {{ background: {BLUE}; border-radius: 3px; }}
    QTableWidget {{
        background: {PANEL}; border: none; gridline-color: {BORDER}; font-size: 10px;
    }}
    QTableWidget::item {{ padding: 4px 8px; border-bottom: 1px solid {BORDER}; }}
    QHeaderView::section {{
        background: {CARD}; color: {TEXT3}; font-size: 9px; font-weight: 600;
        border: none; border-bottom: 1px solid {BORDER}; padding: 5px 8px;
    }}
    QTextEdit {{
        background: transparent; border: none; color: {TEXT2};
        font-family: 'Consolas', 'Cascadia Code', monospace; font-size: 10px;
    }}
    QScrollBar:vertical {{
        background: {PANEL}; width: 6px; border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER}; border-radius: 3px; min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    """


# ═══════════════════════════════════════════════════════════════════════
# Circular progress widget (for Replay Buffer card)
# ═══════════════════════════════════════════════════════════════════════

class CircularProgress(QWidget):
    """Small circular progress ring with percentage text in the center."""

    def __init__(self, parent=None, size=44, color=BLUE):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._pct = 0
        self._color = color

    def set_value(self, pct: float):
        self._pct = max(0, min(100, int(pct)))
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(4, 4, self.width() - 8, self.height() - 8)
        pen_bg = QPen(QColor(BORDER), 3)
        p.setPen(pen_bg)
        p.drawArc(r, 0, 360 * 16)
        pen_fg = QPen(QColor(self._color), 3)
        p.setPen(pen_fg)
        span = int(-self._pct / 100 * 360 * 16)
        p.drawArc(r, 90 * 16, span)
        p.setPen(QPen(QColor(TEXT)))
        f = p.font()
        f.setPixelSize(10)
        f.setBold(True)
        p.setFont(f)
        p.drawText(r, Qt.AlignCenter, f"{self._pct}%")
        p.end()


# ═══════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════

def _panel(title: str = "") -> tuple[QFrame, QVBoxLayout]:
    """Create a styled panel frame with layout and optional title."""
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
    lbl = QLabel(text)
    lbl.setStyleSheet(f"font-size: {size}px; font-weight: {w}; color: {color};")
    return lbl


def _sparkline(color: str, height: int = 30) -> tuple[pg.PlotWidget, pg.PlotDataItem]:
    """Create a tiny, borderless sparkline plot."""
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


def _make_metric_row(icon_char: str, icon_color: str, name: str) -> tuple[QFrame, QLabel]:
    """Build one metric row: colored icon + name + value label (returned)."""
    row = QFrame()
    row.setStyleSheet(f"background: {CARD}; border-radius: 6px; border: 1px solid {BORDER};")
    rl = QHBoxLayout(row)
    rl.setContentsMargins(10, 8, 10, 8)
    rl.setSpacing(10)
    # icon circle
    icon = QLabel(icon_char)
    icon.setFixedSize(28, 28)
    icon.setAlignment(Qt.AlignCenter)
    icon.setStyleSheet(
        f"background: {icon_color}22; color: {icon_color}; font-size: 14px;"
        f"font-weight: 700; border-radius: 14px; border: none;"
    )
    rl.addWidget(icon)
    rl.addWidget(_label(name, 11, TEXT2))
    rl.addStretch()
    val = _label("—", 14, TEXT, True)
    rl.addWidget(val)
    return row, val


# ═══════════════════════════════════════════════════════════════════════
# Training Dashboard
# ═══════════════════════════════════════════════════════════════════════

class TrainingDashboard:
    """Enterprise-grade training dashboard — PySide6 + pyqtgraph."""

    def __init__(self, trainer):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.trainer = trainer
        self.start_time = time.time()
        self._last_ep_seen = 0

        # ── Historical data ───────────────────────────────────────────
        self.h_episodes          = []
        self.h_rewards           = []
        self.h_avg_rewards       = []
        self.h_best_line         = []
        self.h_critic_loss       = []
        self.h_actor_loss        = []
        self.h_alpha             = []
        self.h_alpha_loss        = []
        self.h_coordination      = []
        self.h_mission_success   = []
        self.h_speed             = []
        self.h_buffer            = []

        self._active_tab = "Reward"

        # ── Build UI ──────────────────────────────────────────────────
        self.win = QMainWindow()
        self.win.setWindowTitle("DroneOps AI — Enterprise Training Dashboard")
        self.win.resize(1680, 960)
        self.win.setStyleSheet(_qss())

        root = QWidget()
        self.win.setCentralWidget(root)
        ml = QVBoxLayout(root)
        ml.setContentsMargins(10, 6, 10, 6)
        ml.setSpacing(6)

        ml.addWidget(self._build_header(), 0)
        ml.addWidget(self._build_kpi_row(), 0)

        # ── Chart + Model Metrics row ─────────────────────────────────
        mid_w = QWidget()
        mid_l = QHBoxLayout(mid_w)
        mid_l.setContentsMargins(0, 0, 0, 0)
        mid_l.setSpacing(8)
        mid_l.addWidget(self._build_chart_area(), 7)
        mid_l.addWidget(self._build_model_metrics(), 3)
        ml.addWidget(mid_w, 5)

        # ── Map + Checkpoints + Hyperparams row ───────────────────────
        info_w = QWidget()
        info_l = QHBoxLayout(info_w)
        info_l.setContentsMargins(0, 0, 0, 0)
        info_l.setSpacing(8)
        info_l.addWidget(self._build_swarm_map(), 35)
        info_l.addWidget(self._build_checkpoints(), 30)
        info_l.addWidget(self._build_hyperparams(), 35)
        ml.addWidget(info_w, 4)

        # ── Log + System + Mission row ────────────────────────────────
        bot_w = QWidget()
        bot_l = QHBoxLayout(bot_w)
        bot_l.setContentsMargins(0, 0, 0, 0)
        bot_l.setSpacing(8)
        bot_l.addWidget(self._build_log(), 50)
        bot_l.addWidget(self._build_system_overview(), 20)
        bot_l.addWidget(self._build_current_mission(), 30)
        ml.addWidget(bot_w, 3)

        # ── Timer ─────────────────────────────────────────────────────
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_step)
        self._timer.start(500)

    # ══════════════════════════════════════════════════════════════════
    # Section builders
    # ══════════════════════════════════════════════════════════════════

    def _build_header(self) -> QWidget:
        bar = QFrame()
        bar.setProperty("role", "panel")
        bar.setFixedHeight(52)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(16, 4, 16, 4)
        hl.setSpacing(20)

        # Title
        title = QLabel("DroneOps AI")
        title.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {GREEN};")
        hl.addWidget(title)
        sep = _label("|", 14, TEXT3)
        hl.addWidget(sep)
        hl.addWidget(_label("ENTERPRISE TRAINING DASHBOARD", 12, TEXT, True))
        sub = _label("SAC + GNN + Transformer Multi-Agent Reinforcement Learning", 9, TEXT3)
        hl.addWidget(sub)
        hl.addStretch()

        # Stats cluster
        for attr in ("Episode", "Step", "Elapsed Time", "ETA"):
            col = QVBoxLayout()
            col.setSpacing(0)
            lbl = _label(attr, 8, TEXT3)
            lbl.setAlignment(Qt.AlignCenter)
            col.addWidget(lbl)
            val = _label("—", 13, TEXT, True)
            val.setAlignment(Qt.AlignCenter)
            col.addWidget(val)
            setattr(self, f"_hdr_{attr.replace(' ', '_').lower()}", val)
            w = QWidget()
            w.setLayout(col)
            w.setFixedWidth(110)
            hl.addWidget(w)

        # Status badge
        self._status_badge = QPushButton("● Running")
        self._status_badge.setProperty("role", "status")
        hl.addWidget(self._status_badge)
        return bar

    # ── KPI cards row ─────────────────────────────────────────────────

    def _build_kpi_row(self) -> QWidget:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        self._kpi_cards = {}
        defs = [
            ("avg_reward",   "AVERAGE REWARD",      GREEN,  "▲"),
            ("best_reward",  "BEST REWARD",          YELLOW, "★"),
            ("replay_buf",   "REPLAY BUFFER",        BLUE,   "◉"),
            ("train_speed",  "TRAINING SPEED",       YELLOW, "⚡"),
            ("mission_succ", "MISSION SUCCESS",      GREEN,  "✓"),
            ("coord_score",  "COORDINATION SCORE",   PURPLE, "◈"),
        ]
        for key, title, color, icon_char in defs:
            card = QFrame()
            card.setProperty("role", "kpi")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 10, 14, 8)
            cl.setSpacing(2)

            # top row: icon + title
            top = QHBoxLayout()
            top.setSpacing(8)
            icon = QLabel(icon_char)
            icon.setFixedSize(26, 26)
            icon.setAlignment(Qt.AlignCenter)
            icon.setStyleSheet(
                f"background: {color}22; color: {color}; font-size: 13px;"
                f"font-weight: 700; border-radius: 13px; border: none;"
            )
            top.addWidget(icon)
            top.addWidget(_label(title, 9, TEXT3, True))
            top.addStretch()
            cl.addLayout(top)

            # value
            val = _label("—", 22, TEXT, True)
            cl.addWidget(val)

            # sparkline
            spark_w, spark_c = _sparkline(color, 28)
            cl.addWidget(spark_w)

            # subtitle
            sub = _label("", 8, TEXT3)
            cl.addWidget(sub)

            rl.addWidget(card)
            self._kpi_cards[key] = {"val": val, "sub": sub, "curve": spark_c}

            # circular progress for replay buffer
            if key == "replay_buf":
                self._buf_ring = CircularProgress(size=40, color=BLUE)
                top.addWidget(self._buf_ring)

        return row

    # ── Chart area ────────────────────────────────────────────────────

    def _build_chart_area(self) -> QFrame:
        f, lay = _panel("TRAINING PERFORMANCE")

        # Tab bar
        tabs_w = QWidget()
        tabs_l = QHBoxLayout(tabs_w)
        tabs_l.setContentsMargins(0, 0, 0, 0)
        tabs_l.setSpacing(0)
        self._chart_tabs = {}
        for name in ("Reward", "Loss", "Alpha", "Mission Success", "Coordination Score"):
            btn = QPushButton(name)
            btn.setProperty("role", "tab")
            btn.setProperty("active", name == "Reward")
            btn.clicked.connect(lambda checked, n=name: self._switch_chart_tab(n))
            tabs_l.addWidget(btn)
            self._chart_tabs[name] = btn
        tabs_l.addStretch()
        lay.addWidget(tabs_w)

        # Chart
        self._chart = pg.PlotWidget()
        self._chart.setBackground(PANEL)
        self._chart.showGrid(x=True, y=True, alpha=0.15)
        self._chart.setLabel("bottom", "Episode", color=TEXT3, **{"font-size": "9px"})
        self._chart.setLabel("left", "Reward", color=TEXT3, **{"font-size": "9px"})
        self._chart.getAxis("bottom").setPen(pg.mkPen(BORDER))
        self._chart.getAxis("left").setPen(pg.mkPen(BORDER))
        self._chart.getAxis("bottom").setTextPen(pg.mkPen(TEXT3))
        self._chart.getAxis("left").setTextPen(pg.mkPen(TEXT3))
        legend = self._chart.addLegend(offset=(60, 10))
        legend.setLabelTextColor(TEXT2)
        legend.setLabelTextSize("9px")

        self._chart_lines = {
            "reward":  self._chart.plot(pen=pg.mkPen(BLUE,  width=1.2), name="Episode Reward"),
            "avg":     self._chart.plot(pen=pg.mkPen(PURPLE, width=2),  name="Average Reward (50)"),
            "best":    self._chart.plot(pen=pg.mkPen(GREEN, width=1, style=Qt.DashLine), name="Best Reward"),
        }
        lay.addWidget(self._chart)
        return f

    # ── Model metrics sidebar ─────────────────────────────────────────

    def _build_model_metrics(self) -> QFrame:
        f, lay = _panel("AI MODEL METRICS")
        self._metric_vals = {}
        for icon_c, ic, name, key in [
            ("A", RED,    "Actor Loss",    "actor_loss"),
            ("C", BLUE,   "Critic Loss",   "critic_loss"),
            ("α", GREEN,  "Alpha",         "alpha"),
            ("α", ORANGE, "Alpha Loss",    "alpha_loss"),
            ("H", CYAN,   "Entropy",       "entropy"),
            ("λ", PURPLE, "Learning Rate", "lr"),
        ]:
            row_w, val_lbl = _make_metric_row(icon_c, ic, name)
            lay.addWidget(row_w)
            self._metric_vals[key] = val_lbl
        lay.addStretch()
        return f

    # ── Swarm map ─────────────────────────────────────────────────────

    def _build_swarm_map(self) -> QFrame:
        f, lay = _panel("LIVE SWARM VISUALIZATION")
        self._map = pg.PlotWidget()
        self._map.setBackground(BG)
        self._map.setAspectLocked(True)
        self._map.setXRange(-13, 13)
        self._map.setYRange(-13, 13)
        self._map.hideAxis("bottom")
        self._map.hideAxis("left")
        self._map.getViewBox().setMouseEnabled(x=False, y=False)
        self._map.getViewBox().setMenuEnabled(False)

        self._map_drones    = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None))
        self._map_targets   = pg.ScatterPlotItem(size=14, pen=pg.mkPen(None), symbol="star")
        self._map_obstacles = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), symbol="x")
        self._map_base      = pg.ScatterPlotItem(size=16, pen=pg.mkPen(None), symbol="t")
        self._map.addItem(self._map_drones)
        self._map.addItem(self._map_targets)
        self._map.addItem(self._map_obstacles)
        self._map.addItem(self._map_base)

        # Drone labels
        self._drone_texts = []
        for i in range(6):
            t = pg.TextItem(f"D{i}", color=TEXT2, anchor=(0.5, 1.5))
            t.setFont(QFont("Segoe UI", 7, QFont.Bold))
            self._map.addItem(t)
            self._drone_texts.append(t)

        # Legend
        leg = QHBoxLayout()
        leg.setSpacing(12)
        for sym, col, name in [("●", BLUE, "Drone"), ("★", GREEN, "Target"),
                                ("✕", RED, "Obstacle"), ("▲", TEXT3, "Base")]:
            leg.addWidget(_label(f"{sym} {name}", 8, col))
        leg.addStretch()
        lay.addLayout(leg)
        lay.addWidget(self._map)
        return f

    # ── Checkpoints ───────────────────────────────────────────────────

    def _build_checkpoints(self) -> QFrame:
        f, lay = _panel("CHECKPOINT STATUS")
        grid = QGridLayout()
        grid.setSpacing(6)

        self._ckpt_labels = {}
        models = [
            ("Actor",       "actor_trained.pth",  0, 0),
            ("Critic",      "critic_trained.pth", 0, 1),
            ("GNN",         "gnn_trained.pth",    0, 2),
            ("Transformer", "trans_trained.pth",  1, 0),
            ("Meta Adapter","meta_trained.pth",   1, 1),
        ]
        for name, fname, r, c in models:
            card = QFrame()
            card.setStyleSheet(
                f"background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(10, 8, 10, 8)
            cl.setSpacing(4)

            # header row with check icon
            hr = QHBoxLayout()
            check = QLabel("✓")
            check.setFixedSize(20, 20)
            check.setAlignment(Qt.AlignCenter)
            check.setStyleSheet(
                f"background: {GREEN}22; color: {GREEN}; font-size: 12px;"
                f"font-weight: 700; border-radius: 10px; border: none;"
            )
            hr.addWidget(check)
            hr.addWidget(_label(name, 11, TEXT, True))
            hr.addStretch()
            cl.addLayout(hr)

            status_lbl = _label("Pending", 9, TEXT3)
            size_lbl = _label("", 8, TEXT3)
            cl.addWidget(status_lbl)
            cl.addWidget(size_lbl)
            grid.addWidget(card, r, c)
            self._ckpt_labels[fname] = (check, status_lbl, size_lbl)

        lay.addLayout(grid)
        lay.addStretch()
        return f

    # ── Hyperparameters table ─────────────────────────────────────────

    def _build_hyperparams(self) -> QFrame:
        f, lay = _panel("HYPERPARAMETERS")

        import sys as _sys
        trainer_mod = _sys.modules.get("training.trainer")

        params = [
            ("Learning Rate (Actor)",  getattr(cfg, "LR_ACTOR",  "N/A"),  "Actor optimizer learning rate"),
            ("Learning Rate (Critic)", getattr(cfg, "LR_CRITIC", "N/A"),  "Critic optimizer learning rate"),
            ("Discount Factor (γ)",    getattr(cfg, "GAMMA",     "N/A"),  "Future reward discount"),
            ("Soft Update Tau (τ)",    getattr(cfg, "TAU",       "N/A"),  "Target network update rate"),
            ("Batch Size",             getattr(cfg, "BATCH",     "N/A"),  "Training batch size"),
            ("Replay Capacity",        f"{getattr(cfg, 'BUFFER_SIZE', 0):,}", "Experience buffer capacity"),
            ("Alpha (Entropy Coef)",   "dynamic",                         "Entropy regularization"),
            ("Update Frequency",       getattr(cfg, "UPDATE_EVERY",
                                         getattr(trainer_mod, "UPDATE_EVERY", "N/A")),
                                                                          "Update every N steps"),
            ("Updates Per Step",       getattr(cfg, "UPDATES_PER_STEP",
                                         getattr(trainer_mod, "UPDATES_PER_STEP", "N/A")),
                                                                          "Gradient updates per step"),
        ]

        tbl = QTableWidget(len(params), 3)
        tbl.setHorizontalHeaderLabels(["Parameter", "Value", "Description"])
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setSelectionMode(QAbstractItemView.NoSelection)
        tbl.setShowGrid(False)
        tbl.setAlternatingRowColors(False)

        for i, (name, val, desc) in enumerate(params):
            tbl.setItem(i, 0, QTableWidgetItem(name))
            v_item = QTableWidgetItem(str(val))
            v_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tbl.setItem(i, 1, v_item)
            d_item = QTableWidgetItem(desc)
            d_item.setForeground(QColor(TEXT3))
            tbl.setItem(i, 2, d_item)
            tbl.setRowHeight(i, 26)

        # Store reference to alpha row for live update
        self._hyper_alpha_item = tbl.item(6, 1)
        lay.addWidget(tbl)
        return f

    # ── Training log ──────────────────────────────────────────────────

    def _build_log(self) -> QFrame:
        f, lay = _panel("LIVE TRAINING LOG")
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lay.addWidget(self._log_text)
        self._log_count = 0
        return f

    # ── System overview ───────────────────────────────────────────────

    def _build_system_overview(self) -> QFrame:
        f, lay = _panel("SYSTEM OVERVIEW")
        self._sys_bars = {}
        for name, color in [("CPU Usage", BLUE), ("Memory Usage", PURPLE),
                             ("Buffer Usage", GREEN)]:
            rl = QHBoxLayout()
            rl.setSpacing(6)
            rl.addWidget(_label(name, 9, TEXT2))
            rl.addStretch()
            val = _label("—", 9, TEXT, True)
            rl.addWidget(val)
            lay.addLayout(rl)
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar {{ background: {BORDER}; border: none; border-radius: 3px; max-height: 6px; }}"
                f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
            )
            lay.addWidget(bar)
            self._sys_bars[name] = (val, bar)
        lay.addStretch()
        return f

    # ── Current mission ───────────────────────────────────────────────

    def _build_current_mission(self) -> QFrame:
        f, lay = _panel("CURRENT MISSION")
        self._mission_labels = {}
        for key, label_text in [("phase", "Phase"), ("progress", "Progress"),
                                 ("targets", "Targets Found"), ("time", "Time Remaining")]:
            rl = QHBoxLayout()
            rl.addWidget(_label(label_text, 10, TEXT2))
            rl.addStretch()
            val = _label("—", 11, TEXT, True)
            rl.addWidget(val)
            lay.addLayout(rl)
            self._mission_labels[key] = val
            if key == "progress":
                self._mission_bar = QProgressBar()
                self._mission_bar.setMaximum(100)
                self._mission_bar.setTextVisible(False)
                self._mission_bar.setStyleSheet(
                    f"QProgressBar {{ background: {BORDER}; max-height: 6px; border-radius: 3px; }}"
                    f"QProgressBar::chunk {{ background: {GREEN}; border-radius: 3px; }}"
                )
                lay.addWidget(self._mission_bar)
        lay.addStretch()
        return f

    # ══════════════════════════════════════════════════════════════════
    # Chart tab switching
    # ══════════════════════════════════════════════════════════════════

    def _switch_chart_tab(self, name: str):
        self._active_tab = name
        for k, btn in self._chart_tabs.items():
            btn.setProperty("active", k == name)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._redraw_chart()

    def _redraw_chart(self):
        chart = self._chart
        for line in self._chart_lines.values():
            line.clear()

        chart.legend.clear()

        if self._active_tab == "Reward" and self.h_rewards:
            eps = self.h_episodes
            self._chart_lines["reward"].setData(eps, self.h_rewards)
            self._chart_lines["avg"].setData(eps, self.h_avg_rewards)
            if self.h_best_line:
                self._chart_lines["best"].setData(eps, self.h_best_line)
            chart.setLabel("left", "Reward")

        elif self._active_tab == "Loss" and self.h_critic_loss:
            eps = self.h_episodes[:len(self.h_critic_loss)]
            self._chart_lines["reward"].setData(eps, self.h_critic_loss)
            self._chart_lines["reward"].setPen(pg.mkPen(BLUE, width=1.2))
            self._chart_lines["avg"].setData(eps, self.h_actor_loss)
            self._chart_lines["avg"].setPen(pg.mkPen(PURPLE, width=1.2))
            chart.setLabel("left", "Loss")

        elif self._active_tab == "Alpha" and self.h_alpha:
            eps = self.h_episodes[:len(self.h_alpha)]
            self._chart_lines["reward"].setData(eps, self.h_alpha)
            self._chart_lines["reward"].setPen(pg.mkPen(GREEN, width=1.5))
            self._chart_lines["avg"].setData(eps, self.h_alpha_loss)
            self._chart_lines["avg"].setPen(pg.mkPen(YELLOW, width=1.2))
            chart.setLabel("left", "Alpha")

        elif self._active_tab == "Mission Success" and self.h_mission_success:
            eps = self.h_episodes[:len(self.h_mission_success)]
            self._chart_lines["reward"].setData(eps, self.h_mission_success)
            self._chart_lines["reward"].setPen(pg.mkPen(GREEN, width=1.5))
            chart.setLabel("left", "Success %")

        elif self._active_tab == "Coordination Score" and self.h_coordination:
            eps = self.h_episodes[:len(self.h_coordination)]
            self._chart_lines["reward"].setData(eps, self.h_coordination)
            self._chart_lines["reward"].setPen(pg.mkPen(PURPLE, width=1.5))
            chart.setLabel("left", "Coordination")

    # ══════════════════════════════════════════════════════════════════
    # Main polling update
    # ══════════════════════════════════════════════════════════════════

    def update_step(self):
        trainer = self.trainer
        ep = getattr(trainer, "current_ep", 0)
        metrics = getattr(trainer, "metrics", None)
        env = getattr(trainer, "env", None)
        planner = getattr(trainer, "planner", None)
        is_training = getattr(trainer, "is_training", True)

        # ── Collect new episode data ──────────────────────────────────
        if metrics and hasattr(metrics, "rewards") and ep > self._last_ep_seen and len(metrics.rewards) > 0:
            self._last_ep_seen = ep
            self.h_episodes.append(ep)
            self.h_rewards.append(metrics.rewards[-1])

            # moving average
            window = self.h_rewards[-50:]
            self.h_avg_rewards.append(float(np.mean(window)))

            best = getattr(trainer, "best_ep_reward", 0)
            if best == -float("inf"):
                best = 0
            self.h_best_line.append(best)

            # Losses — metrics stores them too, but trainer.last_* is always latest
            self.h_critic_loss.append(getattr(trainer, "last_c_loss", 0))
            self.h_actor_loss.append(getattr(trainer, "last_a_loss", 0))
            self.h_alpha.append(getattr(trainer, "last_alpha", 0))
            self.h_alpha_loss.append(getattr(trainer, "last_alpha_loss", 0))

            # coordination from metrics
            if hasattr(metrics, "coordination") and len(metrics.coordination) > 0:
                self.h_coordination.append(metrics.coordination[-1])
            else:
                self.h_coordination.append(0)

            # mission success (reward > 10 heuristic)
            recent = self.h_rewards[-20:] if len(self.h_rewards) >= 20 else self.h_rewards
            succ_rate = sum(1 for r in recent if r > 10) / max(1, len(recent)) * 100
            self.h_mission_success.append(succ_rate)

            buf_sz = len(trainer.buf) if hasattr(trainer, "buf") else 0
            self.h_buffer.append(buf_sz)

            elapsed = time.time() - self.start_time
            tps = getattr(trainer, "total_steps", 0) / max(1, elapsed)
            self.h_speed.append(tps)

        # ── Update header ─────────────────────────────────────────────
        max_ep = getattr(cfg, "MAX_EPISODES", 500)
        total_steps = getattr(trainer, "total_steps", 0)
        elapsed = time.time() - self.start_time
        eta = (elapsed / max(1, ep)) * (max_ep - ep) if ep > 0 else 0

        self._hdr_episode.setText(f"{ep} / {max_ep}")
        self._hdr_step.setText(f"{total_steps:,}")
        self._hdr_elapsed_time.setText(str(datetime.timedelta(seconds=int(elapsed))))
        self._hdr_eta.setText(str(datetime.timedelta(seconds=int(eta))))

        if is_training:
            self._status_badge.setText("● Running")
            self._status_badge.setStyleSheet(
                f"background: #064e3b; color: {GREEN}; border: 1px solid #065f46;"
                f"border-radius: 4px; font-size: 11px; font-weight: 600; padding: 5px 14px;"
            )
        else:
            self._status_badge.setText("● Completed")
            self._status_badge.setStyleSheet(
                f"background: #1e1b4b; color: {PURPLE}; border: 1px solid #312e81;"
                f"border-radius: 4px; font-size: 11px; font-weight: 600; padding: 5px 14px;"
            )

        # ── Update KPI cards ──────────────────────────────────────────
        if self.h_rewards:
            avg_r = self.h_avg_rewards[-1] if self.h_avg_rewards else 0
            best_r = self.h_best_line[-1] if self.h_best_line else 0
            buf_sz = self.h_buffer[-1] if self.h_buffer else 0
            speed = self.h_speed[-1] if self.h_speed else 0
            succ = self.h_mission_success[-1] if self.h_mission_success else 0
            coord = self.h_coordination[-1] if self.h_coordination else 0
            buf_cap = getattr(cfg, "BUFFER_SIZE", 200000)

            self._kpi_cards["avg_reward"]["val"].setText(f"+{avg_r:.1f}" if avg_r >= 0 else f"{avg_r:.1f}")
            self._kpi_cards["best_reward"]["val"].setText(f"+{best_r:.1f}" if best_r >= 0 else f"{best_r:.1f}")
            self._kpi_cards["replay_buf"]["val"].setText(f"{buf_sz:,}")
            self._kpi_cards["train_speed"]["val"].setText(f"{speed:.1f} TPS")
            self._kpi_cards["mission_succ"]["val"].setText(f"{succ:.1f}%")
            self._kpi_cards["coord_score"]["val"].setText(f"{coord:.2f}")

            # Subtitles
            if len(self.h_avg_rewards) > 20:
                old = np.mean(self.h_avg_rewards[-40:-20]) if len(self.h_avg_rewards) > 40 else self.h_avg_rewards[0]
                change = ((avg_r - old) / max(0.01, abs(old))) * 100
                arrow = "▲" if change >= 0 else "▼"
                self._kpi_cards["avg_reward"]["sub"].setText(f"{arrow} {abs(change):.1f}% vs last 20 eps")

            best_ep = self.h_rewards.index(max(self.h_rewards)) + 1 if self.h_rewards else 0
            self._kpi_cards["best_reward"]["sub"].setText(f"Episode {best_ep}")
            self._kpi_cards["replay_buf"]["sub"].setText(f"/ {buf_cap:,} Capacity")
            self._buf_ring.set_value(buf_sz / max(1, buf_cap) * 100)

            ep_per_min = ep / max(1, elapsed / 60)
            self._kpi_cards["train_speed"]["sub"].setText(f"{ep_per_min:.1f} episodes/min")

            # Sparklines
            for key, hist in [
                ("avg_reward", self.h_avg_rewards), ("best_reward", self.h_best_line),
                ("replay_buf", self.h_buffer), ("train_speed", self.h_speed),
                ("mission_succ", self.h_mission_success), ("coord_score", self.h_coordination),
            ]:
                if len(hist) > 1:
                    self._kpi_cards[key]["curve"].setData(hist[-60:])

        # ── Update chart ──────────────────────────────────────────────
        if self._active_tab == "Reward":
            # Reset pens to defaults for reward tab
            self._chart_lines["reward"].setPen(pg.mkPen(BLUE, width=1.2))
            self._chart_lines["avg"].setPen(pg.mkPen(PURPLE, width=2))
        self._redraw_chart()

        # ── Update model metrics ──────────────────────────────────────
        self._metric_vals["actor_loss"].setText(f"{getattr(trainer, 'last_a_loss', 0):.4f}")
        self._metric_vals["critic_loss"].setText(f"{getattr(trainer, 'last_c_loss', 0):.4f}")
        self._metric_vals["alpha"].setText(f"{getattr(trainer, 'last_alpha', 0):.4f}")
        self._metric_vals["alpha_loss"].setText(f"{getattr(trainer, 'last_alpha_loss', 0):.4f}")
        # Entropy ≈ -log_alpha for display
        alpha = getattr(trainer, "last_alpha", 0.2)
        target_ent = float(getattr(cfg, "TARGET_ENTROPY", -4.0))
        self._metric_vals["entropy"].setText(f"{-target_ent:.2f}")
        self._metric_vals["lr"].setText(f"{getattr(cfg, 'LR_ACTOR', 'N/A')}")

        # ── Update alpha in hyperparams table ─────────────────────────
        if self._hyper_alpha_item:
            self._hyper_alpha_item.setText(f"{alpha:.4f}")

        # ── Update swarm map ──────────────────────────────────────────
        if env:
            # Drones
            alive_pos, alive_colors, dead_pos = [], [], []
            for d in getattr(env, "drones", []):
                pos = getattr(d, "pos", [0, 0])
                if getattr(d, "alive", True):
                    alive_pos.append(pos)
                    e = getattr(d, "energy", 1.0)
                    if e > 0.5:
                        alive_colors.append(pg.mkBrush(BLUE))
                    elif e > 0.2:
                        alive_colors.append(pg.mkBrush(YELLOW))
                    else:
                        alive_colors.append(pg.mkBrush(RED))
                else:
                    dead_pos.append(pos)

            if alive_pos:
                ap = np.array(alive_pos)
                self._map_drones.setData(
                    pos=ap, brush=alive_colors,
                    pen=pg.mkPen(BORDER, width=1), size=12, symbol="o"
                )
            else:
                self._map_drones.clear()

            # Drone labels
            drones = getattr(env, "drones", [])
            for i, t in enumerate(self._drone_texts):
                if i < len(drones):
                    d = drones[i]
                    p = getattr(d, "pos", [0, 0])
                    e = int(getattr(d, "energy", 1.0) * 100)
                    t.setPos(p[0], p[1])
                    t.setText(f"D{i} {e}%")
                    t.setVisible(True)
                else:
                    t.setVisible(False)

            # Targets
            tgts = getattr(env, "targets", [])
            if len(tgts) > 0:
                reached_ids = {tid for (_, tid) in getattr(env, "targets_reached", set())}
                colors = []
                for i in range(len(tgts)):
                    colors.append(pg.mkBrush(YELLOW if i in reached_ids else GREEN))
                self._map_targets.setData(
                    pos=np.array(tgts), brush=colors, size=14, symbol="star"
                )
            else:
                self._map_targets.clear()

            # Obstacles
            obs = getattr(env, "obstacles", [])
            if len(obs) > 0:
                self._map_obstacles.setData(
                    pos=np.array(obs), brush=pg.mkBrush(RED),
                    pen=pg.mkPen(RED, width=1.5), size=10, symbol="x"
                )
            else:
                self._map_obstacles.clear()

            # Base marker at origin
            self._map_base.setData(
                pos=np.array([[0, 0]]), brush=pg.mkBrush(TEXT3),
                pen=pg.mkPen(TEXT3, width=1), size=16, symbol="t"
            )

        # ── Update checkpoints ────────────────────────────────────────
        for fname, (check_lbl, status_lbl, size_lbl) in self._ckpt_labels.items():
            ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
            path = os.path.join(ckpt_dir, fname)
            # Also check project root
            root_path = os.path.join(os.path.dirname(__file__), "..", fname)
            actual = path if os.path.exists(path) else (root_path if os.path.exists(root_path) else None)

            if actual:
                age = time.time() - os.path.getmtime(actual)
                if age < 60:
                    age_s = "Just now"
                elif age < 3600:
                    age_s = f"{int(age / 60)}m ago"
                else:
                    age_s = f"{int(age / 3600)}h ago"
                sz = os.path.getsize(actual)
                sz_s = f"{sz / 1e6:.1f} MB" if sz > 1e6 else f"{sz / 1e3:.0f} KB"
                status_lbl.setText(f"Last Saved: {age_s}")
                size_lbl.setText(f"Size: {sz_s}")
                check_lbl.setStyleSheet(
                    f"background: {GREEN}22; color: {GREEN}; font-size: 12px;"
                    f"font-weight: 700; border-radius: 10px; border: none;"
                )
            else:
                status_lbl.setText("Pending")
                size_lbl.setText("")
                check_lbl.setStyleSheet(
                    f"background: {TEXT3}22; color: {TEXT3}; font-size: 12px;"
                    f"font-weight: 700; border-radius: 10px; border: none;"
                )

        # ── Update log ────────────────────────────────────────────────
        events = getattr(trainer, "log_events", [])
        if len(events) > self._log_count:
            new_lines = events[self._log_count:]
            self._log_count = len(events)
            for line in new_lines:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                if "Saved" in line or "Checkpoint" in line:
                    color = GREEN
                    badge = "●"
                elif "Reward +" in line or "best" in line.lower():
                    color = YELLOW
                    badge = "●"
                elif "Fail" in line or "Reward -" in line:
                    color = RED
                    badge = "●"
                else:
                    color = BLUE
                    badge = "●"
                self._log_text.append(
                    f'<span style="color:{color}; font-weight:bold">{badge}</span> '
                    f'<span style="color:{TEXT3}">{ts}</span>  '
                    f'<span style="color:{TEXT2}">{line}</span>'
                )
            sb = self._log_text.verticalScrollBar()
            sb.setValue(sb.maximum())

        # ── Update system overview ────────────────────────────────────
        try:
            import psutil
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            self._sys_bars["CPU Usage"][0].setText(f"{cpu:.0f}%")
            self._sys_bars["CPU Usage"][1].setValue(int(cpu))
            self._sys_bars["Memory Usage"][0].setText(f"{mem.used / 1e9:.1f} / {mem.total / 1e9:.0f} GB")
            self._sys_bars["Memory Usage"][1].setValue(int(mem.percent))
        except ImportError:
            elapsed_m = int((time.time() - self.start_time) / 60)
            self._sys_bars["CPU Usage"][0].setText(f"~{min(95, 20 + ep // 5)}%")
            self._sys_bars["CPU Usage"][1].setValue(min(95, 20 + ep // 5))
            self._sys_bars["Memory Usage"][0].setText(f"~{elapsed_m}m elapsed")
            self._sys_bars["Memory Usage"][1].setValue(min(80, 30 + ep // 10))

        buf_cap = getattr(cfg, "BUFFER_SIZE", 200000)
        buf_sz = len(trainer.buf) if hasattr(trainer, "buf") else 0
        buf_pct = int(buf_sz / max(1, buf_cap) * 100)
        self._sys_bars["Buffer Usage"][0].setText(f"{buf_sz:,} / {buf_cap:,}")
        self._sys_bars["Buffer Usage"][1].setValue(buf_pct)

        # ── Update current mission ────────────────────────────────────
        if planner and hasattr(planner, "state"):
            phase = getattr(planner.state, "phase", None)
            phase_name = phase.name if phase else "—"
            self._mission_labels["phase"].setText(phase_name)

            coord = getattr(planner.state, "coordination", None)
            if coord:
                prog = getattr(coord, "mission_progress", 0)
                self._mission_labels["progress"].setText(f"{int(prog * 100)}%")
                self._mission_bar.setValue(int(prog * 100))

            tgts = getattr(env, "targets_reached", set()) if env else set()
            total_t = len(getattr(env, "targets", [])) if env else 0
            self._mission_labels["targets"].setText(f"{len(tgts)} / {total_t}")

            remaining = max_ep - ep
            time_per_ep = elapsed / max(1, ep) if ep > 0 else 0
            rem_s = int(remaining * time_per_ep)
            self._mission_labels["time"].setText(str(datetime.timedelta(seconds=rem_s)))

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def run(self):
        """Show window and start the Qt event loop (blocks until closed)."""
        self.win.show()
        self.app.exec()
