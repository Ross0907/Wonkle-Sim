"""
Controls
--------
  GUI buttons       all parameters live-adjustable
  Click value       type exact numbers  (W, H, hover, etc.)
  Drag dividers     resize panels
  Mouse wheel       zoom grid / scroll metrics
  Right-drag        pan grid view
  R                 reset zoom & pan
  F11               toggle fullscreen
  Q / ESC           quit

Parameter Legend
----------------
  Rows / Cols       sensor grid dimensions
  Rad               sensor display radius (mm)
  HGap / SGap       hex / square inter-sensor gap (mm)
  MagH              magnet centre → sensor PCB distance inside pen (mm)
  PenD              pen nib → tablet surface air gap (mm)
  Eff.H             effective height = MagH + PenD (total field path)
  Tilt / Dir        magnet tilt angle (°) and direction (°)
  RSpd              random-tilt animation speed multiplier
  α                 IIR position filter smoothing (0 = no filter, 1 = instant)
"""

# ── platform fixes BEFORE pygame ──────────────────────────────────────
import os, sys, math, time
import numpy as np
from typing import List, Optional, Tuple

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "wonkle.hall.grid.compare.3")
    except Exception:
        pass
    os.environ.setdefault('SDL_VIDEO_WINDOW_POS', 'center')

import pygame

# ═══════════════════════════════════════════════════════════════════════
#  AH49HSC  &  ADC  PHYSICS
# ═══════════════════════════════════════════════════════════════════════

AH49HSC_VCC          = 3.3           # supply voltage (V)
AH49HSC_QUIESCENT_V  = AH49HSC_VCC / 2.0   # 1.65 V
AH49HSC_SENS_MV_G    = 1.4           # sensitivity (mV / Gauss)
AH49HSC_NOISE_MV     = 0.35          # RMS noise estimate

ADC_BITS             = 12
ADC_MAX              = (1 << ADC_BITS) - 1   # 4095
ADC_VREF             = 3.3
ADC_LSB_V            = ADC_VREF / ADC_MAX    # ≈ 0.806 mV

FW_THRESHOLD         = 2300                  # from sensor.rs
FW_THRESH_V          = FW_THRESHOLD * ADC_LSB_V         # ≈ 1.853 V
FW_THRESH_GAUSS      = ((FW_THRESH_V - AH49HSC_QUIESCENT_V) * 1000.0
                        / AH49HSC_SENS_MV_G)             # ≈ 145 G

# ── 15 mm N52 ring magnet ────────────────────────────────────────────
MAGNET_OD_MM   = 15.0
MAGNET_ID_MM   = 5.0
MAGNET_T_MM    = 5.0
MAGNET_Br_G    = 14500          # N52 remanence ≈ 1.45 T
MAGNET_R_OUT   = MAGNET_OD_MM / 2.0
MAGNET_R_IN    = MAGNET_ID_MM / 2.0

# ── defaults ─────────────────────────────────────────────────────────
INIT_W, INIT_H   = 1700, 1020
FPS               = 8000
DEFAULT_ROWS      = 11
DEFAULT_COLS      = 19
SENSOR_RADIUS_MM  = 5.0     # visual display radius
SPACING_HEX_MM    = 2.5     # hex gap  → pitch = 2*5+2.5 = 12.5 mm
SPACING_SQ_MM     = 0.0     # square gap → pitch = 2*5+0 = 10 mm
DEFAULT_HOVER_MM  = 10.0    # pen–sensor gap
DEFAULT_TILT      = 5.0
DEFAULT_TILT_DIR  = 0.0

SENSOR_SHAPES = ["Circle", "Square", "Hexagon", "Triangle", "Diamond"]
TILT_MODES    = ["Fixed", "Sweep", "Random"]

# ── colour palette ────────────────────────────────────────────────────
C_BG        = (18, 18, 28)
C_PANEL     = (28, 30, 42)
C_PANEL_HDR = (38, 40, 56)
C_BORDER    = (55, 58, 75)
C_DIVIDER   = (65, 68, 88)
C_DIV_HOV   = (95, 100, 135)
C_BTN       = (48, 50, 65)
C_BTN_HOV   = (62, 66, 88)
C_BTN_ACT   = (65, 115, 185)
C_BTN_TXT   = (210, 215, 225)
C_TEXT      = (190, 195, 210)
C_TEXT_DIM  = (115, 120, 138)
C_TEXT_BRT  = (235, 238, 250)
C_ACCENT    = (0, 185, 255)
C_ACCENT2   = (255, 85, 195)
C_CROSS     = (0, 255, 210)
C_OUTLINE   = (80, 85, 105)
C_HEX_C     = (60, 175, 255)
C_SQ_C      = (255, 155, 45)
C_WIN       = (75, 215, 115)
C_TIE       = (175, 175, 95)
C_EDIT_BG   = (35, 40, 60)

# Diverging heatmap for normalised [-1, 1] sensor values
# Positive:  dark-blue(0) → green(+0.5) → red(+1)   (matches reference)
# Negative:  dark-blue(0) → bright-blue(−1)
HEATMAP_POS = [(50, 50, 150), (0, 200, 0), (255, 0, 0)]
HEATMAP_NEG = [(50, 50, 150), (0, 120, 255)]


def heat_color(v: float):
    """Map normalised value [-1, 1] → RGB with gamma curve for visibility."""
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        a = v ** 0.45                          # gamma lift for low values
        if a < 0.5:
            t = a * 2.0
            c1, c2 = HEATMAP_POS[0], HEATMAP_POS[1]
        else:
            t = (a - 0.5) * 2.0
            c1, c2 = HEATMAP_POS[1], HEATMAP_POS[2]
    else:
        a = (-v) ** 0.45
        t = min(1.0, a)
        c1, c2 = HEATMAP_NEG[0], HEATMAP_NEG[1]
    return (int(c1[0]+(c2[0]-c1[0])*t),
            int(c1[1]+(c2[1]-c1[1])*t),
            int(c1[2]+(c2[2]-c1[2])*t))


# ═══════════════════════════════════════════════════════════════════════
#  RING-MAGNET FIELD MODEL
# ═══════════════════════════════════════════════════════════════════════

def _ring_Bz_onaxis(z, R_out=MAGNET_R_OUT, R_in=MAGNET_R_IN,
                    t=MAGNET_T_MM, Br=MAGNET_Br_G):
    """Exact on-axis Bz (Gauss) at axial distance *z* from ring centre."""
    def f(d):
        if abs(d) < 1e-9:
            return 0.0
        return (d / math.sqrt(d * d + R_out * R_out)
                - d / math.sqrt(d * d + R_in * R_in))
    return (Br / 2.0) * (f(z + t / 2.0) - f(z - t / 2.0))


def _field_at_sensor(sx, sy, mx, my, hover_h,
                     tilt_rad, tilt_dir_rad,
                     B0_cache):
    """
    Return Bz **in Gauss** at sensor (sx, sy, z=0) from ring magnet
    centred at (mx, my) at height *hover_h* above the sensor plane.

    Uses a magnetic-dipole model whose moment is rotated by
    (tilt_rad, tilt_dir_rad) and whose on-axis peak is calibrated
    to the exact ring-magnet formula.
    """
    ct = math.cos(tilt_rad)
    st = math.sin(tilt_rad)
    cp = math.cos(tilt_dir_rad)
    sp = math.sin(tilt_dir_rad)

    # Tilt shifts the projected centre on the sensor plane
    shift_x = hover_h * st * cp
    shift_y = hover_h * st * sp
    emx = mx + shift_x
    emy = my + shift_y

    dx = sx - emx
    dy = sy - emy
    rho_sq = dx * dx + dy * dy
    r_sq   = rho_sq + hover_h * hover_h
    r_mag  = math.sqrt(r_sq)
    if r_mag < 0.01:
        r_mag = 0.01

    # Tilted dipole moment direction: m̂ = (st·cp, st·sp, ct)
    inv_r  = 1.0 / r_mag
    rhat_x = dx * inv_r
    rhat_y = dy * inv_r
    rhat_z = -hover_h * inv_r       # sensor at z=0, magnet at +h

    m_dot_rhat = st * cp * rhat_x + st * sp * rhat_y + ct * rhat_z

    # Bz_dipole  ∝  (3·(m̂·r̂)·r̂_z  –  m̂_z) / r³
    Bz_rel = (3.0 * m_dot_rhat * rhat_z - ct) / (r_mag ** 3)

    # Normalise so that at (r=0, tilt=0) → B0_cache
    # At r=0, tilt=0: rhat=(0,0,-1), m̂=(0,0,1), m̂·r̂ = -1
    # Bz_rel = (3·(-1)·(-1) - 1)/h³ = 2/h³
    scale = B0_cache * hover_h ** 3 / 2.0
    return scale * Bz_rel


# Vectorised version for whole grid
def _field_all(sx_arr, sy_arr, mx, my, hover_h,
               tilt_rad, tilt_dir_rad, B0, add_noise=False):
    """Return Bz array in Gauss for all sensors (numpy)."""
    ct = math.cos(tilt_rad)
    st = math.sin(tilt_rad)
    cp = math.cos(tilt_dir_rad)
    sp = math.sin(tilt_dir_rad)

    shift_x = hover_h * st * cp
    shift_y = hover_h * st * sp
    dx = sx_arr - (mx + shift_x)
    dy = sy_arr - (my + shift_y)
    rho_sq = dx * dx + dy * dy
    h2 = hover_h * hover_h
    r_sq = rho_sq + h2
    r_mag = np.sqrt(r_sq)
    r_mag = np.maximum(r_mag, 0.01)

    inv_r = 1.0 / r_mag
    rz    = -hover_h * inv_r

    m_dot_r = st * cp * (dx * inv_r) + st * sp * (dy * inv_r) + ct * rz
    Bz_rel  = (3.0 * m_dot_r * rz - ct) / (r_mag ** 3)
    scale   = B0 * hover_h ** 3 / 2.0
    Bz_gauss = scale * Bz_rel

    if add_noise:
        Bz_gauss = Bz_gauss + np.random.normal(
            0, AH49HSC_NOISE_MV / AH49HSC_SENS_MV_G, Bz_gauss.shape)
    return Bz_gauss


def _gauss_to_adc(Bz_gauss):
    """Convert Gauss → AH49HSC voltage → 12-bit ADC counts."""
    voltage = AH49HSC_QUIESCENT_V + Bz_gauss * (AH49HSC_SENS_MV_G / 1000.0)
    adc = np.clip(voltage / ADC_VREF * ADC_MAX, 0, ADC_MAX).astype(np.int32)
    return adc


def _adc_threshold(adc):
    """Firmware-matching threshold (sensor.rs: THRESHOLD = 2300)."""
    return np.maximum(0, adc - FW_THRESHOLD)


# ═══════════════════════════════════════════════════════════════════════
#  SENSOR  &  GRID
# ═══════════════════════════════════════════════════════════════════════

class Sensor:
    __slots__ = ("row", "col", "x", "y", "value", "gauss", "adc_raw", "adc_thresh")

    def __init__(self, row, col, x, y):
        self.row = row; self.col = col
        self.x = x; self.y = y
        self.value = 0.0
        self.gauss = 0.0
        self.adc_raw = 0
        self.adc_thresh = 0


class SensorGrid:
    def __init__(self, rows, cols, radius, spacing, grid_type, hover_h=DEFAULT_HOVER_MM):
        self.rows      = rows
        self.cols      = cols
        self.radius    = radius
        self.spacing   = spacing
        self.grid_type = grid_type
        self.hover_h   = hover_h
        self.sensors: List[Sensor] = []
        self.est_pos:        Optional[Tuple[float, float]] = None
        self.est_pos_centroid: Optional[Tuple[float, float]] = None
        self.alpha   = 0.85         # IIR alpha (higher = more responsive)
        # --- smoothing state ---
        self._prev_raw:  Optional[Tuple[float, float]] = None
        self._prev_raw_c: Optional[Tuple[float, float]] = None
        self._vel:  float = 0.0
        self._vel_c: float = 0.0
        self._B0     = _ring_Bz_onaxis(hover_h)
        self._build()

    @property
    def pitch(self):
        return self.radius * 2 + self.spacing

    def _build(self):
        self.sensors.clear()
        cc = self.pitch
        if self.grid_type == "hex":
            dy = cc * math.sqrt(3) / 2
            for r in range(self.rows):
                for c in range(self.cols):
                    x = c * cc + (cc / 2.0 if r % 2 else 0)
                    self.sensors.append(Sensor(r, c, x, r * dy))
        else:
            for r in range(self.rows):
                for c in range(self.cols):
                    self.sensors.append(Sensor(r, c, c * cc, r * cc))
        # numpy acceleration arrays
        self._sx = np.array([s.x for s in self.sensors], dtype=np.float64)
        self._sy = np.array([s.y for s in self.sensors], dtype=np.float64)

    def refresh_B0(self):
        self._B0 = _ring_Bz_onaxis(self.hover_h)

    @property
    def bounds(self):
        xs = self._sx; ys = self._sy
        return (float(xs.min()), float(xs.max()),
                float(ys.min()), float(ys.max()))

    @property
    def width_mm(self):
        b = self.bounds; return b[1] - b[0] + 2 * self.radius

    @property
    def height_mm(self):
        b = self.bounds; return b[3] - b[2] + 2 * self.radius

    @property
    def footprint_mm2(self):
        return self.width_mm * self.height_mm

    @property
    def effective_width(self):
        """Center-to-center x span — the reliable tracking zone.
        Firmware maps col 0 … col (cols-1) → USB 0 … MAX.
        Edge sensors mark the boundary; positions beyond them are
        unreliable because there are no neighbours to interpolate."""
        b = self.bounds
        return b[1] - b[0]

    @property
    def effective_height(self):
        """Center-to-center y span — the reliable tracking zone."""
        b = self.bounds
        return b[3] - b[2]

    @property
    def effective_area_mm2(self):
        """Effective tracking area (center-to-center, excludes edge margins)."""
        return self.effective_width * self.effective_height

    # ── field ─────────────────────────────────────────────────────────
    def update_field(self, mx, my, tilt_deg=0.0, tilt_dir_deg=0.0,
                     add_noise=False):
        tilt_r = math.radians(tilt_deg)
        tdir_r = math.radians(tilt_dir_deg)
        Bz = _field_all(self._sx, self._sy, mx, my,
                        self.hover_h, tilt_r, tdir_r, self._B0, add_noise)
        adc_raw    = _gauss_to_adc(Bz)
        adc_thresh = _adc_threshold(adc_raw)
        # Normalise field to [-1, 1] by on-axis peak B0
        B0_abs = abs(self._B0) if abs(self._B0) > 1e-6 else 1.0
        for i, s in enumerate(self.sensors):
            s.gauss      = float(Bz[i])
            s.adc_raw    = int(adc_raw[i])
            s.adc_thresh = int(adc_thresh[i])
            s.value      = max(-1.0, min(1.0, float(Bz[i]) / B0_abs))

    def clear_field(self):
        for s in self.sensors:
            s.value = s.gauss = 0.0
            s.adc_raw = s.adc_thresh = 0

    # ── position estimation ───────────────────────────────────────────
    def find_local(self, top_n=9, thr=0.01):
        active = [s for s in self.sensors if s.value > thr]
        if not active:
            return []
        pk = max(active, key=lambda s: s.value)
        active.sort(key=lambda s: math.hypot(s.x-pk.x, s.y-pk.y))
        return active[:top_n]

    def estimate_pos_gauss_fit(self, local=None):
        """2D quadratic peak fit using raw Gauss values.

        Fits  z = a + bx + cy + dx² + exy + fy²  to all sensors with
        non-trivial field, weighted by Gauss².  The peak of the
        quadratic gives sub-pitch resolution.  Falls back to a
        Gauss-weighted centroid if the fit degenerates.
        Works identically for hex and square grids.
        """
        pts_x, pts_y, pts_z = [], [], []
        for s in self.sensors:
            if s.gauss > 0.1:
                pts_x.append(s.x)
                pts_y.append(s.y)
                pts_z.append(s.gauss)
        if len(pts_x) < 6:
            return self._gauss_centroid(pts_x, pts_y, pts_z)
        try:
            xs = np.array(pts_x)
            ys = np.array(pts_y)
            zs = np.array(pts_z)
            w = zs * zs
            # Design matrix: [1, x, y, x², xy, y²]
            A = np.column_stack([
                np.ones(len(xs)), xs, ys,
                xs*xs, xs*ys, ys*ys
            ])
            # Weighted least squares: (A'WA)^-1 A'Wz
            AtW = A.T * w          # broadcast: each col scaled by w
            p = np.linalg.solve(AtW @ A, AtW @ zs)
            # Peak: dz/dx=0, dz/dy=0
            det = 4*p[3]*p[5] - p[4]*p[4]
            if det >= 0:  # not a maximum
                return self._gauss_centroid(pts_x, pts_y, pts_z)
            ex = (p[4]*p[2] - 2*p[5]*p[1]) / det
            ey = (p[4]*p[1] - 2*p[3]*p[2]) / det
            # Reject estimates far outside grid
            b = self.bounds
            margin = self.pitch * 1.5
            if (ex < b[0] - margin or ex > b[1] + margin or
                    ey < b[2] - margin or ey > b[3] + margin):
                return self._gauss_centroid(pts_x, pts_y, pts_z)
            return (float(ex), float(ey))
        except Exception:
            return self._gauss_centroid(pts_x, pts_y, pts_z)

    @staticmethod
    def _gauss_centroid(px, py, pz):
        """Fallback: weighted centroid from pre-collected Gauss lists."""
        if not px:
            return None
        total = sum(pz)
        if total < 0.01:
            return None
        return (sum(x*z for x, z in zip(px, pz)) / total,
                sum(y*z for y, z in zip(py, pz)) / total)

    def estimate_pos_centroid(self):
        """Firmware-matching weighted centroid (sensor.rs find_center).

        Square grids use the exact firmware algorithm:
          sum_x += adc_thresh * (i % cols)
          sum_y += adc_thresh * (i / cols)
          result = (sum_x/total * pitch, sum_y/total * pitch)
        which maps grid indices to physical mm.

        Hex grids have no firmware analogue, so we use physical
        coordinates weighted by Gauss values for smooth interpolation.
        """
        if self.grid_type == "hex":
            # Hex: use Gauss-weighted physical coordinates
            total = 0.0
            sum_x = 0.0
            sum_y = 0.0
            for s in self.sensors:
                g = max(0.0, s.gauss)
                if g < 0.01:
                    continue
                total += g
                sum_x += g * s.x
                sum_y += g * s.y
            if total < 0.01:
                return None
            return (sum_x / total, sum_y / total)
        else:
            # Square: firmware-exact using thresholded ADC + grid indices
            total = 0
            sum_x = 0
            sum_y = 0
            for i, s in enumerate(self.sensors):
                v = s.adc_thresh
                if v == 0:
                    continue
                total += v
                sum_x += v * (i % self.cols)
                sum_y += v * (i // self.cols)
            if total == 0:
                return None
            # Firmware maps to 0..MAX; we map to physical mm
            cx = sum_x / total * self.pitch
            cy = sum_y / total * self.pitch
            return (cx, cy)

    # ── velocity-adaptive alpha ────────────────────────────────────
    def _adaptive_alpha(self, vel, base_alpha):
        """Return alpha that ramps from *base_alpha* (slow) → 1.0 (fast).

        Slow movement: smooth heavily (base_alpha ~0.85) to kill jitter.
        Fast movement: snap to 1.0 so cursor never lags behind the pen.
        Half-pitch/frame (~5 mm/frame) is considered "fast".
        """
        # Use half-pitch as the speed at which alpha saturates to 1.0
        fast_thresh = max(self.pitch * 0.5, 0.5)
        speed_norm = min(vel / fast_thresh, 1.0)
        # Linear ramp — reaches 1.0 quickly at moderate speed
        return base_alpha + (1.0 - base_alpha) * speed_norm

    def _iir_filter(self, prev, new, vel, base_alpha):
        """Velocity-adaptive IIR. At high speed, snaps directly to raw."""
        a = self._adaptive_alpha(vel, base_alpha)
        if a > 0.98 or prev is None:
            return new
        return (prev[0] + a * (new[0] - prev[0]),
                prev[1] + a * (new[1] - prev[1]))

    def apply_filter(self, new, mouse_mm=None):
        """Position filter for Gauss fit cursor."""
        if new is None:
            if self.est_pos is not None and mouse_mm is not None:
                dist = math.hypot(mouse_mm[0] - self.est_pos[0],
                                  mouse_mm[1] - self.est_pos[1])
                if dist > self.pitch * 4:
                    self.est_pos = None
                    self._prev_raw = None
                    self._vel = 0.0
            return self.est_pos

        # Track velocity from raw estimates
        if self._prev_raw is not None:
            raw_delta = math.hypot(new[0] - self._prev_raw[0],
                                   new[1] - self._prev_raw[1])
        else:
            raw_delta = 0.0
        self._prev_raw = new
        self._vel = self._vel * 0.4 + raw_delta * 0.6   # fast-tracking EMA

        self.est_pos = self._iir_filter(self.est_pos, new, self._vel, self.alpha)
        return self.est_pos

    def apply_filter_centroid(self, new):
        """Position filter for centroid cursor."""
        if new is None:
            if self.est_pos_centroid is not None:
                self.est_pos_centroid = None
                self._prev_raw_c = None
                self._vel_c = 0.0
            return None

        if self._prev_raw_c is not None:
            raw_d = math.hypot(new[0] - self._prev_raw_c[0],
                               new[1] - self._prev_raw_c[1])
        else:
            raw_d = 0.0
        self._prev_raw_c = new
        self._vel_c = self._vel_c * 0.4 + raw_d * 0.6

        self.est_pos_centroid = self._iir_filter(
            self.est_pos_centroid, new, self._vel_c, self.alpha)
        return self.est_pos_centroid


# ═══════════════════════════════════════════════════════════════════════
#  METRICS  &  COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def _simulate_rmse(grid, tilt_deg=0.0, n_angles=8, method="both"):
    """Monte-Carlo RMSE in mm."""
    n = len(grid.sensors)
    samples = 40 if n < 400 else (20 if n < 1000 else 10)
    b = grid.bounds
    mx = (b[1] - b[0]) * 0.20
    my = (b[3] - b[2]) * 0.20
    rng = np.random.RandomState(42)
    errs_gauss, errs_cent = [], []

    angles = [0.0] if tilt_deg < 0.1 else [
        i * 360.0 / n_angles for i in range(n_angles)]

    for _ in range(samples):
        tx = b[0] + mx + rng.random() * (b[1] - b[0] - 2 * mx)
        ty = b[2] + my + rng.random() * (b[3] - b[2] - 2 * my)
        for phi in angles:
            grid.update_field(tx, ty, tilt_deg=tilt_deg, tilt_dir_deg=phi)
            # Gaussian fit
            local = grid.find_local(12)
            est_g = grid.estimate_pos_gauss_fit(local)
            if est_g is not None:
                errs_gauss.append(math.hypot(est_g[0]-tx, est_g[1]-ty))
            # Weighted centroid
            est_c = grid.estimate_pos_centroid()
            if est_c is not None:
                errs_cent.append(math.hypot(est_c[0]-tx, est_c[1]-ty))

    grid.clear_field()

    def _rms(e):
        return math.sqrt(sum(x*x for x in e)/len(e)) if e else float("inf")

    if method == "gauss":
        return _rms(errs_gauss)
    elif method == "centroid":
        return _rms(errs_cent)
    return _rms(errs_gauss), _rms(errs_cent)


def _metrics(grid, tilt_deg=0.0):
    n   = len(grid.sensors)
    r   = grid.radius
    cc  = grid.pitch
    fp  = grid.footprint_mm2
    sa  = n * math.pi * r * r
    one = math.pi * r * r

    m = {}
    m["n"]        = n
    m["rows"]     = grid.rows
    m["cols"]     = grid.cols
    m["cc"]       = cc
    m["r"]        = r
    m["fp"]       = fp
    m["sa"]       = sa
    m["one"]      = one
    m["hover"]    = grid.hover_h
    m["cov"]      = 100 * sa / fp if fp > 0 else 0
    m["density"]  = n / (fp / 100) if fp > 0 else 0
    m["width"]    = grid.width_mm
    m["height"]   = grid.height_mm
    m["eff_width"]  = grid.effective_width
    m["eff_height"] = grid.effective_height
    m["eff_area"]   = grid.effective_area_mm2

    m["cell"] = (cc * cc * math.sqrt(3) / 2) if grid.grid_type == "hex" else (cc * cc)
    m["dead_cell"] = m["cell"] - one
    m["pack"]  = 100 * one / m["cell"] if m["cell"] > 0 else 0
    m["dead_max"] = (cc / math.sqrt(3)) if grid.grid_type == "hex" else (cc * math.sqrt(2) / 2)
    m["nn"]       = cc
    m["nyquist"]  = cc / 2

    # on-axis peak field & threshold info
    m["B0_gauss"] = _ring_Bz_onaxis(grid.hover_h)
    m["thresh_gauss"] = FW_THRESH_GAUSS
    m["active_radius_est"] = 0.0
    # estimate the radius at which field hits threshold
    for rr in np.linspace(0, 50, 200):
        Bz = m["B0_gauss"] * (2 * grid.hover_h**2 - rr**2) / (
            (grid.hover_h**2 + rr**2) ** 2.5) * grid.hover_h**3 / 2
        if Bz < m["thresh_gauss"]:
            m["active_radius_est"] = rr
            break

    rmse_g_up, rmse_c_up = _simulate_rmse(grid, tilt_deg=0.0, method="both")
    m["rmse_gauss"]    = rmse_g_up
    m["rmse_centroid"]  = rmse_c_up

    if tilt_deg > 0.1:
        rmse_g_t, rmse_c_t = _simulate_rmse(grid, tilt_deg=tilt_deg, method="both")
        m["rmse_gauss_tilt"]   = rmse_g_t
        m["rmse_centroid_tilt"] = rmse_c_t
    else:
        m["rmse_gauss_tilt"]    = rmse_g_up
        m["rmse_centroid_tilt"] = rmse_c_up

    return m


def _rows_cols_for_area(target_w, target_h, radius, spacing, grid_type):
    """Compute rows/cols so that effective area ≈ target.
    Effective area = center-to-center sensor span (firmware tracking zone).
    Square: eff_w = (cols-1)*pitch, eff_h = (rows-1)*pitch
    Hex:    eff_w ≈ (cols-1)*pitch + pitch/2, eff_h = (rows-1)*dy
    """
    cc = radius * 2 + spacing
    if cc <= 0:
        return 2, 2
    if grid_type == "hex":
        dy = cc * math.sqrt(3) / 2
        # eff_w = (cols-1)*cc + cc/2 (stagger offset for rows>1)
        cols = max(2, round((target_w - cc / 2) / cc) + 1)
        rows = max(2, round(target_h / dy) + 1)
    else:
        # eff_w = (cols-1)*cc
        cols = max(2, round(target_w / cc) + 1)
        rows = max(2, round(target_h / cc) + 1)
    return rows, cols


def build_comparison(rows, cols, radius, spacing_hex, spacing_sq, hover_h,
                     tilt_deg, mode="count", target_w=None, target_h=None):
    if mode == "area" and target_w and target_h:
        hr, hc = _rows_cols_for_area(target_w, target_h, radius, spacing_hex, "hex")
        sr, sc = _rows_cols_for_area(target_w, target_h, radius, spacing_sq, "square")
    else:
        hr = sr = rows; hc = sc = cols

    hg = SensorGrid(hr, hc, radius, spacing_hex, "hex",    hover_h)
    sg = SensorGrid(sr, sc, radius, spacing_sq,  "square", hover_h)
    h  = _metrics(hg, tilt_deg)
    s  = _metrics(sg, tilt_deg)

    f = lambda v, p=2: f"{v:.{p}f}" if isinstance(v, float) else str(v)

    rd: list = []

    if mode == "area" and target_w and target_h:
        rd.append(("COMPARISON MODE",
                   f"━ Same Eff. Area ━  {target_w:.0f}×{target_h:.0f} mm",
                   f"Hex {h['n']}  |  Sq {s['n']} sensors", "-"))
    else:
        rd.append(("COMPARISON MODE",
                   "━ Same Count ━",
                   f"{rows}×{cols} = {h['n']} sensors", "-"))

    rd += [
        ("Grid Layout",           "Hexagonal",                 "Square",               "-"),
        ("Rows × Cols",           f'{h["rows"]}×{h["cols"]}',  f'{s["rows"]}×{s["cols"]}', "-"),
        ("Total Sensors",         str(h["n"]),                 str(s["n"]),            "-"),
        ("Sensor (AH49HSC)",      f'{AH49HSC_SENS_MV_G} mV/G', f'{AH49HSC_SENS_MV_G} mV/G', "-"),
        ("ADC Threshold",         f'{FW_THRESHOLD} / {ADC_MAX}', f'{FW_THRESHOLD} / {ADC_MAX}', "-"),
        ("Threshold Field",       f'{h["thresh_gauss"]:.0f} G', f'{s["thresh_gauss"]:.0f} G', "-"),
        ("Sensor Radius (mm)",    f(h["r"], 1),              f(s["r"], 1),            "-"),
        ("Pitch / C-C (mm)",      f(h["cc"], 1),             f(s["cc"], 1),           "-"),
        ("Gap / Spacing (mm)",    f(h["cc"] - 2*h["r"], 1),  f(s["cc"] - 2*s["r"], 1), "-"),
        ("Hover Height (mm)",     f(h["hover"], 1),          f(s["hover"], 1),        "-"),
        ("", "", "", "-"),
    ]

    # ── field ──
    rd += [
        ("MAGNETIC FIELD (15mm N52 RING)", "", "", "-"),
        ("On-axis B₀ (Gauss)",   f(h["B0_gauss"], 1),  f(s["B0_gauss"], 1),  "-"),
        ("Active Sensor Radius",  f'{h["active_radius_est"]:.1f} mm',
                                                        f'{s["active_radius_est"]:.1f} mm', "-"),
        ("", "", "", "-"),
    ]

    # ── sensor ──
    rd += [
        ("SENSOR PROPERTIES", "", "", "-"),
        ("Active Area (mm²)",      f(h["one"], 2),      f(s["one"], 2),       "-"),
        ("  (π·r² – identical)",    "",                   "",                   "-"),
        ("Voronoi Cell (mm²)",     f(h["cell"], 2),      f(s["cell"], 2),      "l"),
        ("Dead / Cell (mm²)",      f(h["dead_cell"], 2), f(s["dead_cell"], 2), "l"),
        ("", "", "", "-"),
    ]

    # ── coverage ──
    rd += [
        ("AREA & COVERAGE", "", "", "-"),
        ("Eff. Width (mm)",        f(h["eff_width"], 1), f(s["eff_width"], 1), "-"),
        ("Eff. Height (mm)",       f(h["eff_height"], 1),f(s["eff_height"], 1),"-"),
        ("Eff. Area (mm²)",        f(h["eff_area"], 0),  f(s["eff_area"], 0),  "-"),
        ("Eff. Area (cm²)",        f(h["eff_area"]/100, 2), f(s["eff_area"]/100, 2), "-"),
        ("Footprint incl. edges",  f(h["fp"], 0)+" mm²", f(s["fp"], 0)+" mm²","-"),
        ("Total Sensor Area (mm²)",f(h["sa"], 0),        f(s["sa"], 0),        "h"),
        ("Coverage %",             f(h["cov"], 2),       f(s["cov"], 2),       "h"),
        ("Packing Efficiency %",   f(h["pack"], 2),      f(s["pack"], 2),      "h"),
        ("", "", "", "-"),
    ]

    # ── density ──
    rd += [
        ("DENSITY", "", "", "-"),
        ("Sensors / cm²",         f(h["density"], 3),   f(s["density"], 3),   "h"),
    ]
    if h["n"] != s["n"]:
        diff = s["n"] - h["n"]
        pct  = abs(diff) / max(s["n"], 1) * 100
        lbl  = "Sensors Saved by Hex" if diff > 0 else "Extra Sensors in Hex"
        rd.append((lbl, f"{abs(diff)}  ({pct:.1f}%)", "", "-"))
    rd.append(("", "", "", "-"))

    # ── spatial ──
    rd += [
        ("SPATIAL RESOLUTION", "", "", "-"),
        ("NN Distance (mm)",      f(h["nn"], 1),        f(s["nn"], 1),        "l"),
        ("Max Dead Zone (mm)",    f(h["dead_max"], 3),   f(s["dead_max"], 3),  "l"),
        ("Nyquist Limit (mm)",    f(h["nyquist"], 2),    f(s["nyquist"], 2),   "l"),
        ("", "", "", "-"),
    ]

    # ── accuracy – upright ──
    rd += [
        ("POSITION ACCURACY (UPRIGHT)", "", "", "-"),
        ("RMSE Gaussian Fit (mm)",  f(h["rmse_gauss"], 4),     f(s["rmse_gauss"], 4),     "l"),
        ("RMSE Centroid (mm)",      f(h["rmse_centroid"], 4),   f(s["rmse_centroid"], 4),   "l"),
    ]
    if h["rmse_gauss"] < 1e6 and s["rmse_gauss"] > 0 and s["rmse_gauss"] < 1e6:
        ag = (1 - h["rmse_gauss"]/s["rmse_gauss"]) * 100
        rd.append(("Hex Δ (Gauss Fit)", f"{ag:+.1f}%", "", "-"))
    if h["rmse_centroid"] < 1e6 and s["rmse_centroid"] > 0 and s["rmse_centroid"] < 1e6:
        ac = (1 - h["rmse_centroid"]/s["rmse_centroid"]) * 100
        rd.append(("Hex Δ (Centroid)", f"{ac:+.1f}%", "", "-"))
    rd.append(("", "", "", "-"))

    # ── accuracy – tilt ──
    rd += [
        (f"MAGNET TILT ({tilt_deg:.0f}°) — 15mm RING", "", "", "-"),
        ("Ring Magnet Ø (mm)",         f(MAGNET_OD_MM, 1), f(MAGNET_OD_MM, 1), "-"),
        ("Ring Inner Ø (mm)",          f(MAGNET_ID_MM, 1), f(MAGNET_ID_MM, 1), "-"),
        ("Ring Thickness (mm)",        f(MAGNET_T_MM, 1),  f(MAGNET_T_MM, 1),  "-"),
        ("Remanence (Gauss)",          f(MAGNET_Br_G, 0),  f(MAGNET_Br_G, 0),  "-"),
        (f"RMSE Gauss @ {tilt_deg:.0f}°",
            f(h["rmse_gauss_tilt"], 4),  f(s["rmse_gauss_tilt"], 4), "l"),
        (f"RMSE Centroid @ {tilt_deg:.0f}°",
            f(h["rmse_centroid_tilt"], 4), f(s["rmse_centroid_tilt"], 4), "l"),
    ]
    # tilt degradation
    for lbl, hk, sk in [("Gauss Fit", "rmse_gauss", "rmse_gauss_tilt"),
                         ("Centroid",  "rmse_centroid", "rmse_centroid_tilt")]:
        hu = h[hk]; su = s[hk]
        ht = h[sk]; st_ = s[sk]
        if hu > 0 and hu < 1e6:
            hd = (ht / hu - 1) * 100
            sd = (st_ / su - 1) * 100 if su > 0 and su < 1e6 else 0
            rd.append((f"Tilt Degrad. ({lbl})",
                       f"Hex: +{hd:.1f}%", f"Sq: +{sd:.1f}%", "-"))
    rd.append(("", "", "", "-"))

    # ── angular ──
    rd += [
        ("ANGULAR DIVERSITY", "", "", "-"),
        ("Nearest Neighbours",    "6 equidistant",        "4 + 4 diagonal",   "-"),
        ("Best for Gaussian Fit", "✓ 6 directions",      "4 primary dirs",   "-"),
        ("Tilt Robustness",       "Better (6-axis)",      "Weaker (4-axis)",  "-"),
        ("", "", "", "-"),
    ]

    # ── scorecard ──
    result: list = []
    for label, hv, sv, d in rd:
        if d == "-" or label == "":
            result.append((label, hv, sv, ""))
            continue
        try:
            hn = float(hv.split()[0].replace(",", ""))
            sn = float(sv.split()[0].replace(",", ""))
            if abs(hn - sn) < 1e-9:
                w = "Tie"
            elif d == "h":
                w = "◆ Hex" if hn > sn else "◆ Square"
            else:
                w = "◆ Hex" if hn < sn else "◆ Square"
        except (ValueError, IndexError, AttributeError):
            w = ""
        result.append((label, hv, sv, w))

    hex_w = sum(1 for *_, w in result if "Hex" in w)
    sq_w  = sum(1 for *_, w in result if "Square" in w)
    ties  = sum(1 for *_, w in result if w == "Tie")
    result += [
        ("", "", "", ""),
        ("FINAL SCORE",
         f"Hex wins: {hex_w}", f"Square wins: {sq_w}", f"Ties: {ties}"),
        ("", "", "", ""),
    ]

    pk = h["pack"] - s["pack"]
    result.append(("Hex Packing Advantage", f"+{pk:.1f}%", "", ""))
    if s["dead_max"] > 0:
        dz = (1 - h["dead_max"] / s["dead_max"]) * 100
        result.append(("Dead-Zone Advantage", f"{dz:.1f}% smaller", "", ""))
    result += [
        ("", "", "", ""),
        ("NOTE: AH49HSC 1.4 mV/G,",
         "Threshold 2300 ADC,",
         "15mm N52 ring magnet.", ""),
        ("Hex 6-neighbour sampling",
         "gives more angular data",
         "for Gaussian peak fit.", ""),
    ]
    return result


# ═══════════════════════════════════════════════════════════════════════
#  WIDGETS
# ═══════════════════════════════════════════════════════════════════════

def _draw_sensor_shape(surface, shape, color, outline,
                       cx, cy, sr, draw_outline=True):
    icx, icy, isr = int(cx), int(cy), max(1, int(sr))
    if shape == "Circle":
        pygame.draw.circle(surface, color, (icx, icy), isr)
        if draw_outline and sr > 3:
            pygame.draw.circle(surface, outline, (icx, icy), isr, 1)
    elif shape == "Square":
        r = pygame.Rect(cx-sr, cy-sr, sr*2, sr*2)
        pygame.draw.rect(surface, color, r)
        if draw_outline and sr > 3:
            pygame.draw.rect(surface, outline, r, 1)
    elif shape == "Hexagon":
        pts = [(cx+sr*math.cos(math.pi/6+i*math.pi/3),
                cy+sr*math.sin(math.pi/6+i*math.pi/3)) for i in range(6)]
        pygame.draw.polygon(surface, color, pts)
        if draw_outline and sr > 3:
            pygame.draw.polygon(surface, outline, pts, 1)
    elif shape == "Triangle":
        pts = [(cx+sr*math.cos(-math.pi/2+i*2*math.pi/3),
                cy+sr*math.sin(-math.pi/2+i*2*math.pi/3)) for i in range(3)]
        pygame.draw.polygon(surface, color, pts)
        if draw_outline and sr > 3:
            pygame.draw.polygon(surface, outline, pts, 1)
    elif shape == "Diamond":
        pts = [(cx, cy-sr), (cx+sr, cy), (cx, cy+sr), (cx-sr, cy)]
        pygame.draw.polygon(surface, color, pts)
        if draw_outline and sr > 3:
            pygame.draw.polygon(surface, outline, pts, 1)


class Btn:
    def __init__(self, x, y, w, h, text, font, cb=None, toggle=False, active=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text; self.font = font
        self.cb = cb; self.toggle = toggle
        self.active = active; self.hovered = False

    def draw(self, surf):
        bg = C_BTN_ACT if self.active else (C_BTN_HOV if self.hovered else C_BTN)
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, C_BORDER, self.rect, 1, border_radius=4)
        rendered = self.font.render(self.text, True, C_BTN_TXT)
        surf.blit(rendered, rendered.get_rect(center=self.rect.center))

    def handle(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(ev.pos)
        elif (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
              and self.rect.collidepoint(ev.pos)):
            if self.cb:
                self.cb()
            return True
        return False


class EditableValue:
    """Click-to-type exact value field."""

    def __init__(self, font, on_change, fmt="{:.0f}", lo=0, hi=9999):
        self.font = font
        self.on_change = on_change
        self.fmt = fmt
        self.lo, self.hi = lo, hi
        self.editing = False
        self.buf = ""
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, surf, value, pos):
        txt = self.fmt.format(value)
        if self.editing:
            txt = self.buf + "▌"
            rendered = self.font.render(txt, True, C_TEXT_BRT)
            r = rendered.get_rect(topleft=pos)
            pygame.draw.rect(surf, C_EDIT_BG, r.inflate(6, 2))
            pygame.draw.rect(surf, C_ACCENT, r.inflate(6, 2), 1)
        else:
            rendered = self.font.render(txt, True, C_TEXT_BRT)
            r = rendered.get_rect(topleft=pos)
        self.rect = r.inflate(8, 4)
        self.rect.topleft = (pos[0]-4, pos[1]-2)
        surf.blit(rendered, pos)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.editing = True
                self.buf = ""
                return True
            elif self.editing:
                self._submit()
                return True
        if ev.type == pygame.KEYDOWN and self.editing:
            if ev.key == pygame.K_RETURN or ev.key == pygame.K_KP_ENTER:
                self._submit(); return True
            elif ev.key == pygame.K_ESCAPE:
                self.editing = False; return True
            elif ev.key == pygame.K_BACKSPACE:
                self.buf = self.buf[:-1]; return True
            elif ev.unicode in "0123456789.-":
                self.buf += ev.unicode; return True
        return False

    def _submit(self):
        self.editing = False
        try:
            v = float(self.buf)
            v = max(self.lo, min(self.hi, v))
            self.on_change(v)
        except ValueError:
            pass

    @property
    def is_editing(self):
        return self.editing


class Divider:
    def __init__(self, orient):
        self.orient  = orient
        self.dragging = False
        self.hovered  = False

    def hit_rect(self, x, y, length, thickness=6):
        if self.orient == "h":
            return pygame.Rect(x, y - thickness//2, length, thickness)
        return pygame.Rect(x - thickness//2, y, thickness, length)


# ═══════════════════════════════════════════════════════════════════════
#  APPLICATION
# ═══════════════════════════════════════════════════════════════════════

class App:
    def __init__(self):
        pygame.init()
        # Set caption & icon BEFORE set_mode to avoid double window-show
        pygame.display.set_caption(
            "Wonkle — Hall-Effect Grid Comparison (AH49HSC · 15mm Ring Magnet)")
        ico = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(ico, C_ACCENT, (16, 16), 14)
        pygame.display.set_icon(ico)

        self.sw, self.sh = INIT_W, INIT_H
        self.screen = pygame.display.set_mode(
            (self.sw, self.sh), pygame.RESIZABLE)
        # Render background immediately — prevents blank-window flash on taskbar
        self.screen.fill(C_BG)
        pygame.display.flip()
        self.clock = pygame.time.Clock()

        self.f_title = pygame.font.SysFont("Consolas", 22, bold=True)
        self.f_label = pygame.font.SysFont("Consolas", 16, bold=True)
        self.f_small = pygame.font.SysFont("Consolas", 15)
        self.f_tiny  = pygame.font.SysFont("Consolas", 14)
        self.f_btn   = pygame.font.SysFont("Consolas", 14, bold=True)

        # state
        self.grid_type   = "square"
        self.rows        = DEFAULT_ROWS
        self.cols        = DEFAULT_COLS
        self.radius      = SENSOR_RADIUS_MM
        self.spacing_hex = SPACING_HEX_MM
        self.spacing_sq  = SPACING_SQ_MM
        self.hover_h     = DEFAULT_HOVER_MM   # magnet centre → sensor PCB distance (mm)
        self.pen_dist    = 1.0                  # pen nib → tablet surface air gap (mm)
        self.tilt        = DEFAULT_TILT
        self.tilt_dir    = DEFAULT_TILT_DIR
        self.tilt_mode   = 0            # 0=Fixed 1=Sweep 2=Random
        self.rand_speed  = 0.2          # random tilt speed multiplier
        self._rand_target = 0.0         # random tilt target angle
        self._rand_timer  = 0.0         # random tilt interpolation timer
        self.grid        = self._make_grid()

        self.shape_idx = 0
        self.cmp_mode  = 1            # 0=Same Count  1=Same Area
        self.show_gauss   = True       # show Gaussian fit crosshair
        self.show_centroid = False     # show centroid marker
        self.target_w  = 180.0
        self.target_h  = 100.0

        self.zoom  = 1.0
        self.pan_x = self.pan_y = 0.0
        self.panning = False
        self.pan_start     = (0, 0)
        self.pan_start_off = (0.0, 0.0)

        self.comparison: list = []
        self.metrics_scroll = 0
        self.metrics_dirty  = True
        self._sweep_angle   = 0.0     # animated sweep
        self._viz_tilt_dir  = self.tilt_dir  # active tilt dir for drawing

        # panel split  (h=grid/metrics, v=grid/right, h2=controls/minimap)
        self.v_split  = 0.73
        self.h_split  = 0.72
        self.h2_split = 0.62
        self.div_h  = Divider("h")
        self.div_v  = Divider("v")
        self.div_h2 = Divider("h")

        self.running       = True
        self.mouse_in_grid = False
        self.cursor_mm     = (0.0, 0.0)
        self._last_size    = (self.sw, self.sh)

        # editable value widgets — every slider gets one for exact entry
        def _ev(cb, fmt, lo, hi):
            return EditableValue(self.f_small, cb, fmt, lo, hi)

        self.ev_rows     = _ev(lambda v: self._set_val("rows", int(v)), "{:.0f}", 2, 80)
        self.ev_cols     = _ev(lambda v: self._set_val("cols", int(v)), "{:.0f}", 2, 80)
        self.ev_rad      = _ev(lambda v: self._set_val("radius", v),   "{:.2f}", 0.5, 25)
        self.ev_hex_gap  = _ev(lambda v: self._set_val("spacing_hex", v), "{:.2f}", 0, 30)
        self.ev_sq_gap   = _ev(lambda v: self._set_val("spacing_sq", v),  "{:.2f}", 0, 30)
        self.ev_hover    = _ev(lambda v: self._set_hover(v),  "{:.2f}", 1, 50)
        self.ev_pen_dist = _ev(lambda v: self._set_pen_dist(v), "{:.2f}", 0, 30)
        self.ev_tilt     = _ev(lambda v: self._set_val("tilt", v),     "{:.1f}", 0, 45)
        self.ev_tdir     = _ev(lambda v: self._set_val("tilt_dir", v % 360), "{:.1f}", 0, 360)
        self.ev_rspd     = _ev(lambda v: self._set_val("rand_speed", v), "{:.2f}", 0.1, 5.0)
        self.ev_zoom     = _ev(lambda v: self._set_zoom(v),   "{:.2f}", 0.1, 30)
        self.ev_alpha    = _ev(lambda v: self._set_alpha(v),  "{:.2f}", 0, 1)
        self.ev_tw       = _ev(lambda v: self._set_target_w(v), "{:.1f}", 20, 5000)
        self.ev_th       = _ev(lambda v: self._set_target_h(v), "{:.1f}", 20, 5000)
        self.editables = [
            self.ev_rows, self.ev_cols, self.ev_rad,
            self.ev_hex_gap, self.ev_sq_gap,
            self.ev_hover, self.ev_pen_dist,
            self.ev_tilt, self.ev_tdir, self.ev_rspd,
            self.ev_zoom, self.ev_alpha,
            self.ev_tw, self.ev_th,
        ]

        self._layout()
        self._create_buttons()
        # Default is Same Area → size grid to target area
        if self.cmp_mode == 1:
            self._rebuild_for_area()

    @property
    def spacing(self):
        """Active spacing for the currently displayed grid type."""
        return self.spacing_hex if self.grid_type == "hex" else self.spacing_sq

    @spacing.setter
    def spacing(self, v):
        if self.grid_type == "hex":
            self.spacing_hex = v
        else:
            self.spacing_sq = v

    @property
    def effective_h(self):
        """Total magnet-to-sensor distance = recess + air gap."""
        return self.hover_h + self.pen_dist

    def _make_grid(self):
        sp = self.spacing_hex if self.grid_type == "hex" else self.spacing_sq
        return SensorGrid(self.rows, self.cols, self.radius,
                          sp, self.grid_type, self.effective_h)

    # ── layout ────────────────────────────────────────────────────────
    def _layout(self):
        sb_h = 34
        rp_w   = max(220, int(self.sw * (1 - self.v_split)))
        top_h  = max(160, int((self.sh - sb_h) * self.h_split))
        ctrl_h = max(90, int(top_h * self.h2_split))

        self.grid_rect   = pygame.Rect(0, 0, self.sw - rp_w, top_h)
        self.ctrl_rect   = pygame.Rect(self.sw - rp_w, 0, rp_w, ctrl_h)
        self.mini_rect   = pygame.Rect(self.sw - rp_w, ctrl_h, rp_w, top_h - ctrl_h)
        self.met_rect    = pygame.Rect(0, top_h, self.sw, self.sh - top_h - sb_h)
        self.status_rect = pygame.Rect(0, self.sh - sb_h, self.sw, sb_h)

    # ── buttons ───────────────────────────────────────────────────────
    def _create_buttons(self):
        self.buttons: list = []
        cx = self.ctrl_rect.x + 14
        cy = self.ctrl_rect.y + 34
        bw, bh, gap, vg = 100, 24, 8, 2
        vx  = cx + 36
        prx = cx + 145

        # grid type
        self.btn_hex = Btn(cx, cy, bw, bh, "Hexagonal", self.f_btn,
                           lambda: self._set_grid("hex"), True, self.grid_type == "hex")
        self.btn_sq  = Btn(cx+bw+gap, cy, bw, bh, "Square", self.f_btn,
                           lambda: self._set_grid("square"), True, self.grid_type == "square")
        self.buttons += [self.btn_hex, self.btn_sq]
        cy += bh + 6

        def _pm(cy_, dec, inc):
            bm = Btn(cx, cy_, 30, bh, "−", self.f_btn, dec)
            bp = Btn(prx, cy_, 30, bh, "+", self.f_btn, inc)
            self.buttons += [bm, bp]
            return (vx, cy_)

        self.lbl_rows  = _pm(cy, lambda: self._adj("rows", -1),  lambda: self._adj("rows", 1));  cy += bh+vg
        self.lbl_cols  = _pm(cy, lambda: self._adj("cols", -1),  lambda: self._adj("cols", 1));  cy += bh+vg
        self.lbl_rad   = _pm(cy, lambda: self._adj("radius", -.5), lambda: self._adj("radius", .5)); cy += bh+vg
        self.lbl_hex_gap = _pm(cy, lambda: self._adj("spacing_hex", -.5), lambda: self._adj("spacing_hex", .5)); cy += bh+vg
        self.lbl_sq_gap  = _pm(cy, lambda: self._adj("spacing_sq", -.5), lambda: self._adj("spacing_sq", .5)); cy += bh+vg
        self.lbl_hover = _pm(cy, lambda: self._adj("hover_h", -1), lambda: self._adj("hover_h", 1)); cy += bh+vg
        self.lbl_pdist = _pm(cy, lambda: self._adj("pen_dist", -.5), lambda: self._adj("pen_dist", .5)); cy += bh+vg
        self.lbl_tilt  = _pm(cy, lambda: self._adj("tilt", -5),  lambda: self._adj("tilt", 5));  cy += bh+vg
        self.lbl_tdir  = _pm(cy, lambda: self._adj("tilt_dir", -15), lambda: self._adj("tilt_dir", 15)); cy += bh+vg
        self.lbl_rspd  = _pm(cy, lambda: self._adj("rand_speed", -.25), lambda: self._adj("rand_speed", .25)); cy += bh+vg
        self.lbl_zoom  = _pm(cy, lambda: self._adj_zoom(-.2), lambda: self._adj_zoom(.2)); cy += bh+vg
        self.lbl_alpha = _pm(cy, lambda: self._adj_alpha(-.05), lambda: self._adj_alpha(.05)); cy += bh+5

        # shape
        self.btn_shape = Btn(cx, cy, 210, bh,
                             f"Shape: {SENSOR_SHAPES[self.shape_idx]}",
                             self.f_btn, self._next_shape)
        self.buttons.append(self.btn_shape); cy += bh+4

        # tilt mode
        self.btn_tmode = Btn(cx, cy, 210, bh,
                             f"Tilt: {TILT_MODES[self.tilt_mode]}",
                             self.f_btn, self._next_tilt_mode)
        self.buttons.append(self.btn_tmode); cy += bh+5

        # compare mode
        self.btn_cmp0 = Btn(cx, cy, bw, bh, "Same Count", self.f_btn,
                            lambda: self._set_cmp(0), True, self.cmp_mode == 0)
        self.btn_cmp1 = Btn(cx+bw+gap, cy, bw, bh, "Same Area", self.f_btn,
                            lambda: self._set_cmp(1), True, self.cmp_mode == 1)
        self.buttons += [self.btn_cmp0, self.btn_cmp1]; cy += bh+vg

        self.lbl_tw = _pm(cy, lambda: self._adj_tw(-10), lambda: self._adj_tw(10)); cy += bh+vg
        self.lbl_th = _pm(cy, lambda: self._adj_th(-10), lambda: self._adj_th(10)); cy += bh+5

        self.btn_reset = Btn(cx, cy, 210, bh, "Reset View (R)", self.f_btn, self._reset_view)
        self.buttons.append(self.btn_reset); cy += bh + 4

        # cursor visibility toggles
        self.btn_show_gauss = Btn(cx, cy, bw, bh, "+ Gauss", self.f_btn,
            self._toggle_gauss, True, self.show_gauss)
        self.btn_show_cent  = Btn(cx+bw+gap, cy, bw, bh, "□ Centroid", self.f_btn,
            self._toggle_centroid, True, self.show_centroid)
        self.buttons += [self.btn_show_gauss, self.btn_show_cent]

    # ── callbacks ─────────────────────────────────────────────────────
    def _set_grid(self, t):
        if self.grid_type == t:
            return
        self.grid_type = t
        self.btn_hex.active = (t == "hex")
        self.btn_sq.active  = (t == "square")
        self._rebuild()

    def _adj(self, attr, d):
        limits = {"rows": (2, 80), "cols": (2, 80),
                  "radius": (0.5, 25), "spacing_hex": (0, 30),
                  "spacing_sq": (0, 30),
                  "hover_h": (1, 50), "pen_dist": (0, 30),
                  "tilt": (0, 45),
                  "tilt_dir": (-360, 720), "rand_speed": (0.1, 5.0)}
        lo, hi = limits.get(attr, (0, 9999))
        v = getattr(self, attr) + d
        v = round(max(lo, min(hi, v)), 2)
        if attr == "tilt_dir":
            v = v % 360
        setattr(self, attr, v)
        if attr in ("rows", "cols", "radius", "spacing_hex", "spacing_sq"):
            self._rebuild()
        elif attr in ("hover_h", "pen_dist"):
            self.grid.hover_h = self.effective_h
            self.grid.refresh_B0()
            self.metrics_dirty = True
        else:
            self.metrics_dirty = True

    def _adj_zoom(self, d):
        self.zoom = max(0.1, min(30, self.zoom + d))

    def _adj_alpha(self, d):
        self.grid.alpha = round(max(0, min(1, self.grid.alpha + d)), 2)

    def _next_shape(self):
        self.shape_idx = (self.shape_idx + 1) % len(SENSOR_SHAPES)
        self.btn_shape.text = f"Shape: {SENSOR_SHAPES[self.shape_idx]}"

    def _next_tilt_mode(self):
        self.tilt_mode = (self.tilt_mode + 1) % len(TILT_MODES)
        self.btn_tmode.text = f"Tilt: {TILT_MODES[self.tilt_mode]}"
        self.metrics_dirty = True

    def _toggle_gauss(self):
        self.show_gauss = not self.show_gauss
        self.btn_show_gauss.active = self.show_gauss

    def _toggle_centroid(self):
        self.show_centroid = not self.show_centroid
        self.btn_show_cent.active = self.show_centroid

    def _set_cmp(self, m):
        if self.cmp_mode == m:
            return
        self.cmp_mode = m
        self.btn_cmp0.active = (m == 0)
        self.btn_cmp1.active = (m == 1)
        if m == 1:
            # auto-sync target to effective area (center-to-center span)
            self.target_w = round(self.grid.effective_width, 0)
            self.target_h = round(self.grid.effective_height, 0)
        self.metrics_dirty = True

    def _adj_tw(self, d):
        self.target_w = max(20, min(5000, self.target_w + d))
        if self.cmp_mode == 1:
            self._rebuild_for_area()
        self.metrics_dirty = True

    def _adj_th(self, d):
        self.target_h = max(20, min(5000, self.target_h + d))
        if self.cmp_mode == 1:
            self._rebuild_for_area()
        self.metrics_dirty = True

    def _set_target_w(self, v):
        self.target_w = max(20, min(5000, v))
        if self.cmp_mode == 1:
            self._rebuild_for_area()
        self.metrics_dirty = True

    def _set_target_h(self, v):
        self.target_h = max(20, min(5000, v))
        if self.cmp_mode == 1:
            self._rebuild_for_area()
        self.metrics_dirty = True

    def _set_val(self, attr, v):
        """Generic setter for editable values."""
        limits = {"rows": (2, 80), "cols": (2, 80),
                  "radius": (0.5, 25), "spacing_hex": (0, 30),
                  "spacing_sq": (0, 30), "tilt": (0, 45),
                  "tilt_dir": (0, 360), "rand_speed": (0.1, 5.0)}
        lo, hi = limits.get(attr, (0, 9999))
        v = max(lo, min(hi, v))
        if attr in ("rows", "cols"):
            v = int(v)
        else:
            v = round(v, 2)
        setattr(self, attr, v)
        if attr in ("rows", "cols", "radius", "spacing_hex", "spacing_sq"):
            self._rebuild()
        else:
            self.metrics_dirty = True

    def _set_hover(self, v):
        self.hover_h = round(max(1, min(50, v)), 2)
        self.grid.hover_h = self.effective_h
        self.grid.refresh_B0()
        self.metrics_dirty = True

    def _set_pen_dist(self, v):
        self.pen_dist = round(max(0, min(30, v)), 2)
        self.grid.hover_h = self.effective_h
        self.grid.refresh_B0()
        self.metrics_dirty = True

    def _set_zoom(self, v):
        self.zoom = max(0.1, min(30, round(v, 2)))

    def _set_alpha(self, v):
        self.grid.alpha = max(0, min(1, round(v, 2)))

    def _reset_view(self):
        self.zoom = 1.0; self.pan_x = self.pan_y = 0.0

    def _rebuild(self):
        a = self.grid.alpha
        if self.cmp_mode == 1:
            self._rebuild_for_area()
        self.grid = self._make_grid()
        self.grid.alpha = a
        self.metrics_dirty = True
        self._reset_view()

    def _rebuild_for_area(self):
        """In Same Area mode, adjust rows/cols to fit target area."""
        sp = self.spacing_hex if self.grid_type == "hex" else self.spacing_sq
        r, c = _rows_cols_for_area(
            self.target_w, self.target_h,
            self.radius, sp, self.grid_type)
        self.rows = r
        self.cols = c

    def _recompute_metrics(self):
        mode = "area" if self.cmp_mode == 1 else "count"
        self.comparison = build_comparison(
            self.rows, self.cols, self.radius,
            self.spacing_hex, self.spacing_sq,
            self.effective_h, self.tilt,
            mode=mode, target_w=self.target_w, target_h=self.target_h)
        self.metrics_dirty = False

    # ── transforms ────────────────────────────────────────────────────
    def _grid_transform(self):
        vp = self.grid_rect
        da = pygame.Rect(vp.x, vp.y + 30, vp.width, vp.height - 34)
        pad = 30
        gw, gh = self.grid.width_mm, self.grid.height_mm
        if gw <= 0 or gh <= 0:
            return 1, da.centerx, da.centery, 0, 0
        sx = (da.width  - 2*pad) / gw
        sy = (da.height - 2*pad) / gh
        base = min(sx, sy)
        b = self.grid.bounds
        return base, da.centerx, da.centery, (b[0]+b[1])/2, (b[2]+b[3])/2

    def mm2scr(self, mx, my):
        base, cx, cy, gcx, gcy = self._grid_transform()
        eff = base * self.zoom
        return (cx + (mx - gcx + self.pan_x)*eff,
                cy + (my - gcy + self.pan_y)*eff)

    def scr2mm(self, sx, sy):
        base, cx, cy, gcx, gcy = self._grid_transform()
        eff = base * self.zoom
        if eff == 0: return (0.0, 0.0)
        return ((sx - cx)/eff + gcx - self.pan_x,
                (sy - cy)/eff + gcy - self.pan_y)

    def _divider_rects(self):
        top_h = self.grid_rect.height
        rp_w  = self.ctrl_rect.width
        dh  = self.div_h.hit_rect(0, top_h, self.sw)
        dv  = self.div_v.hit_rect(self.sw - rp_w, 0, top_h)
        dh2 = self.div_h2.hit_rect(self.ctrl_rect.x, self.ctrl_rect.bottom, rp_w)
        return dh, dv, dh2

    # ── events ────────────────────────────────────────────────────────
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False; return

            # Resize – NEVER call set_mode() here to prevent taskbar flicker
            if ev.type == pygame.VIDEORESIZE:
                # Use actual surface size — never call set_mode() here
                surf = pygame.display.get_surface()
                if surf is not None:
                    self.screen = surf
                    nw, nh = surf.get_size()
                else:
                    nw, nh = ev.w, ev.h
                if (nw, nh) != self._last_size:
                    self.sw, self.sh = nw, nh
                    self._last_size = (nw, nh)
                    self._layout()
                    self._create_buttons()

            # editable values (handle before buttons to capture keypresses)
            any_editing = any(e.is_editing for e in self.editables)
            for editable in self.editables:
                editable.handle(ev)

            if ev.type == pygame.KEYDOWN and not any_editing:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                elif ev.key == pygame.K_r:
                    self._reset_view()
                elif ev.key == pygame.K_UP:
                    self._adj_alpha(0.05)
                elif ev.key == pygame.K_DOWN:
                    self._adj_alpha(-0.05)
                elif ev.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()

            # dividers
            dh_r, dv_r, dh2_r = self._divider_rects()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if dh_r.collidepoint(mx, my):
                    self.div_h.dragging = True
                elif dv_r.collidepoint(mx, my):
                    self.div_v.dragging = True
                elif dh2_r.collidepoint(mx, my):
                    self.div_h2.dragging = True
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.div_h.dragging = self.div_v.dragging = self.div_h2.dragging = False
            if ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                sb_h = self.status_rect.height
                usable = self.sh - sb_h
                if self.div_h.dragging:
                    self.h_split = max(0.2, min(0.85, my / usable))
                    self._layout(); self._create_buttons()
                elif self.div_v.dragging:
                    self.v_split = max(0.35, min(0.90, mx / self.sw))
                    self._layout(); self._create_buttons()
                elif self.div_h2.dragging:
                    rel = (my - self.ctrl_rect.y) / max(1, self.grid_rect.height)
                    self.h2_split = max(0.20, min(0.85, rel))
                    self._layout(); self._create_buttons()
                self.div_h.hovered  = dh_r.collidepoint(mx, my)
                self.div_v.hovered  = dv_r.collidepoint(mx, my)
                self.div_h2.hovered = dh2_r.collidepoint(mx, my)

            # scroll
            if ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if self.grid_rect.collidepoint(mx, my):
                    old = self.scr2mm(mx, my)
                    fac = 1.15 if ev.y > 0 else 1/1.15
                    self.zoom = max(0.1, min(30, self.zoom * fac))
                    new = self.scr2mm(mx, my)
                    self.pan_x += new[0] - old[0]
                    self.pan_y += new[1] - old[1]
                elif self.met_rect.collidepoint(mx, my):
                    self.metrics_scroll = max(0, self.metrics_scroll - ev.y * 3)

            # pan
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                if self.grid_rect.collidepoint(ev.pos):
                    self.panning = True
                    self.pan_start = ev.pos
                    self.pan_start_off = (self.pan_x, self.pan_y)
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                self.panning = False
            if ev.type == pygame.MOUSEMOTION and self.panning:
                base, *_ = self._grid_transform()
                eff = base * self.zoom
                if eff > 0:
                    self.pan_x = self.pan_start_off[0] + (ev.pos[0]-self.pan_start[0])/eff
                    self.pan_y = self.pan_start_off[1] + (ev.pos[1]-self.pan_start[1])/eff

            for btn in self.buttons:
                btn.handle(ev)

    # ── update ────────────────────────────────────────────────────────
    def update(self) -> List[Sensor]:
        mx, my = pygame.mouse.get_pos()
        self.mouse_in_grid = self.grid_rect.collidepoint(mx, my)

        # Determine tilt direction for real-time viz
        if self.tilt_mode == 0:
            viz_dir = self.tilt_dir
        elif self.tilt_mode == 1:
            self._sweep_angle = (self._sweep_angle + 3 * self.rand_speed) % 360
            viz_dir = self._sweep_angle
        else:
            # Random mode – interpolate toward a new random target
            self._rand_timer += self.rand_speed
            if self._rand_timer >= 1.0:
                self._rand_target = float(np.random.uniform(0, 360))
                self._rand_timer = 0.0
            viz_dir = self._rand_target

        self._viz_tilt_dir = viz_dir

        if self.mouse_in_grid:
            self.cursor_mm = self.scr2mm(mx, my)
            self.grid.update_field(*self.cursor_mm,
                                   tilt_deg=self.tilt,
                                   tilt_dir_deg=viz_dir)
        else:
            # Mouse left the grid → clear estimates & filter state
            self.grid.est_pos = None
            self.grid.est_pos_centroid = None
            self.grid._prev_raw = self.grid._prev_raw_c = None
            self.grid._vel = self.grid._vel_c = 0.0
            self.grid.clear_field()
        local = self.grid.find_local(12)
        # Gauss quadratic peak fit + velocity-adaptive IIR
        mouse_hint = self.cursor_mm if self.mouse_in_grid else None
        self.grid.apply_filter(self.grid.estimate_pos_gauss_fit(), mouse_hint)
        # Firmware-matching centroid + IIR
        self.grid.apply_filter_centroid(self.grid.estimate_pos_centroid())
        if self.metrics_dirty:
            self._recompute_metrics()
        return local

    # ── render ────────────────────────────────────────────────────────
    def render(self, local):
        self.screen.fill(C_BG)
        self._draw_grid_view()
        self._draw_controls()
        self._draw_minimap()
        self._draw_metrics()
        self._draw_dividers()
        self._draw_status(local)
        pygame.display.flip()

    def _draw_dividers(self):
        for div, rect in zip(
                [self.div_h, self.div_v, self.div_h2],
                self._divider_rects()):
            c = C_DIV_HOV if (div.hovered or div.dragging) else C_DIVIDER
            pygame.draw.rect(self.screen, c, rect)

    # ·· grid view ·····················································
    def _draw_grid_view(self):
        vp = self.grid_rect
        pygame.draw.rect(self.screen, C_PANEL, vp)
        pygame.draw.rect(self.screen, C_BORDER, vp, 1)

        gt = "Hexagonal" if self.grid_type == "hex" else "Square"
        ns = len(self.grid.sensors)
        ew = self.grid.effective_width
        eh = self.grid.effective_height
        title_parts = f"{gt} Grid — {self.rows}×{self.cols} ({ns} AH49HSC)  Eff: {ew:.0f}×{eh:.0f} mm"
        if self.cmp_mode == 1:
            other_type_name = "square" if self.grid_type == "hex" else "hex"
            other_sp = self.spacing_sq if self.grid_type == "hex" else self.spacing_hex
            or_, oc = _rows_cols_for_area(
                self.target_w, self.target_h, self.radius, other_sp,
                other_type_name)
            other_n = or_ * oc
            other_type = "Sq" if self.grid_type == "hex" else "Hex"
            title_parts += f"  |  {other_type}: {other_n}"
        title_parts += f"  Tilt: {self.tilt:.0f}°  Eff.H: {self.effective_h:.1f} mm"
        title = self.f_title.render(title_parts, True, C_TEXT_BRT)
        self.screen.blit(title, (vp.x + 10, vp.y + 4))

        clip = pygame.Rect(vp.x+1, vp.y+28, vp.width-2, vp.height-30)
        self.screen.set_clip(clip)

        base, *_ = self._grid_transform()
        eff   = base * self.zoom
        sr    = max(1.0, self.grid.radius * eff)
        shape = SENSOR_SHAPES[self.shape_idx]

        for s in self.grid.sensors:
            sx, sy = self.mm2scr(s.x, s.y)
            if sx+sr < clip.x or sx-sr > clip.right or sy+sr < clip.y or sy-sr > clip.bottom:
                continue
            col = heat_color(s.value)
            _draw_sensor_shape(self.screen, shape, col, C_OUTLINE, sx, sy, sr)
            if sr > 22:
                val_txt = f"{s.value:.2f}" if abs(s.value) > 0.005 else ""
                if val_txt:
                    t = self.f_tiny.render(val_txt, True, C_TEXT)
                    self.screen.blit(t, t.get_rect(center=(int(sx), int(sy))))

        # Crosshair cursor at magnet position
        if self.mouse_in_grid:
            cx2, cy2 = self.mm2scr(*self.cursor_mm)
            cs = 15
            pygame.draw.line(self.screen, C_ACCENT,
                             (cx2-cs, cy2), (cx2+cs, cy2), 2)
            pygame.draw.line(self.screen, C_ACCENT,
                             (cx2, cy2-cs), (cx2, cy2+cs), 2)
            pygame.draw.circle(self.screen, C_ACCENT2,
                               (int(cx2), int(cy2)), 6)
            pygame.draw.circle(self.screen, C_ACCENT,
                               (int(cx2), int(cy2)), 6, 2)

        if self.show_gauss and self.grid.est_pos:
            ex, ey = self.mm2scr(*self.grid.est_pos)
            cs = 14
            pygame.draw.line(self.screen, C_CROSS, (ex-cs, ey), (ex+cs, ey), 3)
            pygame.draw.line(self.screen, C_CROSS, (ex, ey-cs), (ex, ey+cs), 3)
            pygame.draw.circle(self.screen, C_ACCENT2, (int(ex), int(ey)), 7)
            pygame.draw.circle(self.screen, C_CROSS, (int(ex), int(ey)), 7, 2)

        if self.show_centroid and self.grid.est_pos_centroid:
            ex, ey = self.mm2scr(*self.grid.est_pos_centroid)
            pygame.draw.rect(self.screen, C_SQ_C,
                             (int(ex)-4, int(ey)-4, 8, 8), 2)

        # tilt direction arrow
        if self.tilt > 0.5 and self.mouse_in_grid:
            cx2, cy2 = self.mm2scr(*self.cursor_mm)
            arr_len = 30
            ax = cx2 + arr_len * math.cos(math.radians(self._viz_tilt_dir))
            ay = cy2 + arr_len * math.sin(math.radians(self._viz_tilt_dir))
            pygame.draw.line(self.screen, (255, 200, 50),
                             (int(cx2), int(cy2)), (int(ax), int(ay)), 2)
            pygame.draw.circle(self.screen, (255, 200, 50), (int(ax), int(ay)), 4)

        # scale bar
        sc_mm = 10; sc_px = sc_mm * eff
        if sc_px > 20:
            bx = clip.right - 22 - sc_px
            by = clip.bottom - 22
            pygame.draw.line(self.screen, C_TEXT, (bx, by), (bx+sc_px, by), 2)
            pygame.draw.line(self.screen, C_TEXT, (bx, by-4), (bx, by+4), 1)
            pygame.draw.line(self.screen, C_TEXT, (bx+sc_px, by-4), (bx+sc_px, by+4), 1)
            lbl = self.f_tiny.render(f"{sc_mm} mm", True, C_TEXT)
            self.screen.blit(lbl, (bx+sc_px/2-lbl.get_width()/2, by-18))
        self.screen.set_clip(None)

    # ·· controls ······················································
    def _draw_controls(self):
        r = self.ctrl_rect
        pygame.draw.rect(self.screen, C_PANEL, r)
        pygame.draw.rect(self.screen, C_BORDER, r, 1)
        hdr = pygame.Rect(r.x, r.y, r.width, 28)
        pygame.draw.rect(self.screen, C_PANEL_HDR, hdr)
        self.screen.blit(self.f_label.render("  Controls", True, C_TEXT_BRT),
                         (r.x+6, r.y+5))

        clip = pygame.Rect(r.x+1, r.y+29, r.width-2, r.height-30)
        self.screen.set_clip(clip)
        for btn in self.buttons:
            btn.draw(self.screen)

        def lbl(pos, name, val=None, editable=None, value=None, unit=""):
            if pos is None:
                return
            t1 = self.f_small.render(f"{name}: ", True, C_TEXT_DIM)
            self.screen.blit(t1, (pos[0], pos[1]+5))
            vx = pos[0] + t1.get_width()
            if editable is not None and value is not None:
                editable.draw(self.screen, value, (vx, pos[1]+5))
                if unit:
                    ux = editable.rect.right + 2
                    tu = self.f_small.render(unit, True, C_TEXT_DIM)
                    self.screen.blit(tu, (ux, pos[1]+5))
            elif val is not None:
                t2 = self.f_small.render(str(val), True, C_TEXT_BRT)
                self.screen.blit(t2, (vx, pos[1]+5))

        lbl(self.lbl_rows,    "Rows",   editable=self.ev_rows,     value=self.rows)
        lbl(self.lbl_cols,    "Cols",   editable=self.ev_cols,     value=self.cols)
        lbl(self.lbl_rad,     "Rad",    editable=self.ev_rad,      value=self.radius, unit="mm")
        lbl(self.lbl_hex_gap, "HGap",   editable=self.ev_hex_gap,  value=self.spacing_hex, unit="mm")
        lbl(self.lbl_sq_gap,  "SGap",   editable=self.ev_sq_gap,   value=self.spacing_sq, unit="mm")
        lbl(self.lbl_hover,   "MagH",   editable=self.ev_hover,    value=self.hover_h, unit="mm")
        lbl(self.lbl_pdist,   "PenD",   editable=self.ev_pen_dist, value=self.pen_dist, unit="mm")
        lbl(self.lbl_tilt,    "Tilt",   editable=self.ev_tilt,     value=self.tilt, unit="°")
        lbl(self.lbl_tdir,    "Dir",    editable=self.ev_tdir,     value=self.tilt_dir, unit="°")
        lbl(self.lbl_rspd,    "RSpd",   editable=self.ev_rspd,     value=self.rand_speed, unit="×")
        lbl(self.lbl_zoom,    "Zoom",   editable=self.ev_zoom,     value=self.zoom, unit="×")
        lbl(self.lbl_alpha,   "α",      editable=self.ev_alpha,    value=self.grid.alpha)
        lbl(self.lbl_tw,      "W",      editable=self.ev_tw,       value=self.target_w, unit="mm")
        lbl(self.lbl_th,      "H",      editable=self.ev_th,       value=self.target_h, unit="mm")
        self.screen.set_clip(None)

    # ·· minimap ·······················································
    def _draw_minimap(self):
        r = self.mini_rect
        if r.height < 50:
            return
        pygame.draw.rect(self.screen, C_PANEL, r)
        pygame.draw.rect(self.screen, C_BORDER, r, 1)
        hdr = pygame.Rect(r.x, r.y, r.width, 24)
        pygame.draw.rect(self.screen, C_PANEL_HDR, hdr)
        self.screen.blit(self.f_label.render("  Stylus Tracker", True, C_TEXT_BRT),
                         (r.x+6, r.y+3))

        g = self.grid
        b = g.bounds
        pad = 14
        inner = pygame.Rect(r.x+pad, r.y+30, r.width-2*pad, r.height-38-pad)
        if inner.width < 10 or inner.height < 10:
            return
        gw, gh = g.width_mm, g.height_mm
        if gw <= 0 or gh <= 0:
            return
        sc = min(inner.width / gw, inner.height / gh)
        ow = inner.x + (inner.width - gw*sc) / 2
        oh = inner.y + (inner.height - gh*sc) / 2
        rr = max(1, g.radius * sc)

        self.screen.set_clip(inner)
        for s in g.sensors:
            sx = ow + (s.x - b[0] + g.radius) * sc
            sy = oh + (s.y - b[2] + g.radius) * sc
            col = heat_color(s.value)
            pygame.draw.circle(self.screen, col, (int(sx), int(sy)), max(1, int(rr)))
            pygame.draw.circle(self.screen, (50, 52, 68), (int(sx), int(sy)), max(1, int(rr)), 1)

        def _cross(est, col1, col2, sz):
            if est is None:
                return
            px = ow + (est[0] - b[0] + g.radius) * sc
            py = oh + (est[1] - b[2] + g.radius) * sc
            pygame.draw.line(self.screen, col1, (px-sz, py), (px+sz, py), 2)
            pygame.draw.line(self.screen, col1, (px, py-sz), (px, py+sz), 2)
            pygame.draw.circle(self.screen, col2, (int(px), int(py)), 5)
            lbl = self.f_tiny.render(f"({est[0]:.1f},{est[1]:.1f}) mm", True, C_TEXT)
            self.screen.blit(lbl, (px+8, py-8))

        if self.show_gauss:
            _cross(g.est_pos, C_CROSS, C_ACCENT2, 8)
        if self.show_centroid:
            _cross(g.est_pos_centroid, C_SQ_C, C_SQ_C, 6)

        # legend
        ly = inner.bottom - 14
        self.screen.blit(self.f_tiny.render("+ Gauss", True, C_CROSS),
                         (inner.x, ly))
        self.screen.blit(self.f_tiny.render("□ Centroid", True, C_SQ_C),
                         (inner.x + 70, ly))

        self.screen.set_clip(None)

    # ·· metrics ·······················································
    def _draw_metrics(self):
        r = self.met_rect
        pygame.draw.rect(self.screen, C_PANEL, r)
        pygame.draw.rect(self.screen, C_BORDER, r, 1)
        hdr = pygame.Rect(r.x, r.y, r.width, 28)
        pygame.draw.rect(self.screen, C_PANEL_HDR, hdr)
        ml = ["Same Count", "Same Eff. Area"][self.cmp_mode]
        self.screen.blit(self.f_label.render(
            f"  Comparison — {ml} — AH49HSC · 15 mm N52 Ring · "
            f"Eff.H {self.effective_h:.1f} mm · Tilt {self.tilt:.0f}°",
            True, C_TEXT_BRT), (r.x+6, r.y+5))

        if not self.comparison:
            return

        cw = [r.width*0.31, r.width*0.22, r.width*0.22, r.width*0.21]
        cxs = [r.x+14]
        for w in cw[:-1]:
            cxs.append(cxs[-1]+w)

        hy = r.y + 32
        for i, (hd, hc) in enumerate(zip(
                ["Metric", "Hexagonal", "Square", "Winner"],
                [C_TEXT_BRT, C_HEX_C, C_SQ_C, C_TEXT_BRT])):
            self.screen.blit(self.f_label.render(hd, True, hc), (cxs[i], hy))
        pygame.draw.line(self.screen, C_BORDER, (r.x+6, hy+22), (r.right-6, hy+22))

        row_h = 22
        start_y = hy + 26
        vis_h = r.bottom - start_y - 6
        max_vis = max(1, vis_h // row_h)
        total = len(self.comparison)
        max_scr = max(0, total - max_vis)
        self.metrics_scroll = min(self.metrics_scroll, max_scr)

        clip = pygame.Rect(r.x+1, start_y, r.width-2, vis_h)
        self.screen.set_clip(clip)

        for idx in range(total):
            y = start_y + (idx - self.metrics_scroll) * row_h
            if y + row_h < start_y or y > clip.bottom:
                continue
            label, hv, sv, winner = self.comparison[idx]
            if label == "":
                pygame.draw.line(self.screen, C_BORDER,
                                 (r.x+12, y+row_h//2), (r.right-12, y+row_h//2))
                continue
            if idx % 2 == 0:
                pygame.draw.rect(self.screen, (22, 24, 35),
                                 (r.x+2, y, r.width-4, row_h))
            is_hdr = label.isupper() and not label.startswith("NOTE")
            is_note = label.startswith("NOTE") or label.startswith("Hex ")
            lc = C_ACCENT if is_hdr else (C_TEXT_DIM if is_note else C_TEXT)
            self.screen.blit(self.f_small.render(label, True, lc), (cxs[0], y+3))
            hvc = C_WIN if "Hex" in winner else C_TEXT
            self.screen.blit(self.f_small.render(str(hv), True, hvc), (cxs[1], y+3))
            svc = C_WIN if "Square" in winner else C_TEXT
            self.screen.blit(self.f_small.render(str(sv), True, svc), (cxs[2], y+3))
            wc = (C_HEX_C if "Hex" in winner else
                  C_SQ_C if "Square" in winner else
                  C_TIE if winner == "Tie" else C_TEXT_DIM)
            self.screen.blit(self.f_small.render(winner, True, wc), (cxs[3], y+3))

        self.screen.set_clip(None)
        if total > max_vis and max_vis > 0:
            sb_x = r.right - 12
            sb_h = max(20, int(vis_h * max_vis / total))
            sb_y = (start_y + int((vis_h-sb_h)*self.metrics_scroll/max_scr)
                    if max_scr > 0 else start_y)
            pygame.draw.rect(self.screen, (38, 40, 52), (sb_x, start_y, 8, vis_h))
            pygame.draw.rect(self.screen, (80, 85, 110), (sb_x, sb_y, 8, sb_h), border_radius=3)

    # ·· status ········································
    def _draw_status(self, local):
        r = self.status_rect
        pygame.draw.rect(self.screen, (25, 27, 38), r)
        pygame.draw.rect(self.screen, C_BORDER, r, 1)
        fps = self.clock.get_fps()
        est = "N/A"
        if self.grid.est_pos:
            est = f"({self.grid.est_pos[0]:.2f},{self.grid.est_pos[1]:.2f}) mm"
        cent = "N/A"
        if self.grid.est_pos_centroid:
            cent = f"({self.grid.est_pos_centroid[0]:.1f},{self.grid.est_pos_centroid[1]:.1f}) mm"
        above = sum(1 for s in self.grid.sensors if s.adc_thresh > 0)
        parts = [
            f"Cursor: ({self.cursor_mm[0]:.1f},{self.cursor_mm[1]:.1f}) mm",
            f"GaussFit: {est}",
            f"Centroid: {cent}",
            f"Above Thr: {above}/{len(self.grid.sensors)}",
            f"Eff.H: {self.effective_h:.1f} mm",
            f"Tilt: {self.tilt:.0f}°→{self.tilt_dir:.0f}° ({TILT_MODES[self.tilt_mode]})",
            f"FPS: {fps:.0f}",
            "[R] Reset  [F11] Full  [↑↓] α  [RMB] Pan  [Q] Quit",
        ]
        txt = self.f_tiny.render("  │  ".join(parts), True, C_TEXT_DIM)
        self.screen.blit(txt, (r.x+10, r.y+8))

    # ── cursor ────────────────────────────────────────────────────────
    def _update_cursor(self):
        if any(d.hovered or d.dragging for d in (self.div_h, self.div_v, self.div_h2)):
            any_h = (self.div_h.hovered or self.div_h.dragging
                     or self.div_h2.hovered or self.div_h2.dragging)
            pygame.mouse.set_cursor(
                pygame.SYSTEM_CURSOR_SIZENS if any_h else pygame.SYSTEM_CURSOR_SIZEWE)
        elif self.mouse_in_grid:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    # ── main loop ─────────────────────────────────────────────────────
    def run(self):
        while self.running:
            self.handle_events()
            local = self.update()
            self._update_cursor()
            self.render(local)
            self.clock.tick(0)
        pygame.quit()


def main():
    App().run()


if __name__ == "__main__":
    main()
