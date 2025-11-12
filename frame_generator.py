# frame_generator_v2.py
# Přepis frame_generator_v1 -> v2
# - profily berou reálné délky z výpočtu řezů (self.piece_lengths)
# - úhly profilů jsou spočteny z vrcholů outer_poly (nejbližší vrchol k začátku/k konci běhu)

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box, MultiPolygon, LineString, Point
from shapely.ops import unary_union
import math

# ---------- pomocné funkce ----------
def format_length(value):
    if value is None:
        return ""
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}"
    else:
        return f"{value:.1f}"

def format_angle(value):
    if value is None:
        return ""
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}°"
    else:
        return f"{value:.1f}°"

def compute_inner_angles(points):
    pts = list(points)
    if len(pts) >= 2 and (pts[0][0] == pts[-1][0] and pts[0][1] == pts[-1][1]):
        pts = pts[:-1]
    n = len(pts)
    angles = []
    if n < 3:
        return angles
    for i in range(n):
        p_prev = np.array(pts[(i-1) % n])
        p_curr = np.array(pts[i])
        p_next = np.array(pts[(i+1) % n])
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        # úhel mezi v1 a v2 (vnitřní)
        a = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-12), -1.0, 1.0)))
        angles.append(a)
    return angles

def is_ccw(points):
    s = 0.0
    n = len(points)-1
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[i+1]
        s += (x1 - x0)*(y1 + y0)
    return s < 0

def draw_polygon_annotations(points, lengths, ax, outward_offset=None, fontsize=9):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color='black', linewidth=1.0)  # obrys

    n = len(points) - 1
    if n <= 0:
        return ax

    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)
    max_dim = max(bbox_width, bbox_height)
    if max_dim <= 0:
        max_dim = 1.0

    try:
        lens_arr = np.array([max(1.0, float(l)) for l in lengths])
        char_len = float(np.exp(np.mean(np.log(lens_arr))))
    except Exception:
        char_len = max_dim

    min_out = 12.0
    max_out = max_dim * 0.15
    if outward_offset is None:
        outward_offset = float(np.clip(char_len * 0.08, min_out, max_out))

    poly_ccw = is_ccw(points)

    for i in range(n):
        p0 = np.array(points[i]); p1 = np.array(points[i+1])
        dx_line, dy_line = p1[0]-p0[0], p1[1]-p0[1]
        line_len = math.hypot(dx_line, dy_line)
        if line_len < 1e-9:
            continue
        ux, uy = dx_line/line_len, dy_line/line_len
        perp = np.array([-uy, ux])
        outward = -perp if poly_ccw else perp
        outward /= np.linalg.norm(outward) + 1e-12
        mid = (p0 + p1)/2
        text_pos = mid + outward * outward_offset
        length_text = format_length(lengths[i])
        ax.text(text_pos[0], text_pos[1], length_text,
                fontsize=fontsize, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1),
                rotation=0, zorder=5)
    # angles
    inner_angles = compute_inner_angles(points)
    n_edges = len(points)-1
    edge_out_normals = []
    for i in range(n_edges):
        pa = np.array(points[i])
        pb = np.array(points[i+1])
        edge_vec = pb - pa
        L = np.linalg.norm(edge_vec)
        if L < 1e-9:
            edge_out_normals.append(np.array([0.0, 0.0]))
            continue
        ux, uy = edge_vec[0]/L, edge_vec[1]/L
        perp = np.array([-uy, ux])
        outward = -perp if poly_ccw else perp
        outward /= np.linalg.norm(outward) + 1e-12
        edge_out_normals.append(outward)
    angle_offset = float(np.clip(char_len * 0.07, 20.0, max_dim * 0.25))
    verts = list(points)
    if len(verts) >= 2 and (verts[0][0] == verts[-1][0] and verts[0][1] == verts[-1][1]):
        verts = verts[:-1]
    for i, ang in enumerate(inner_angles):
        p_curr = np.array(verts[i % len(verts)])
        n_in = edge_out_normals[(i-1) % n_edges]
        n_out = edge_out_normals[i % n_edges]
        out_dir = n_in + n_out
        out_len = np.linalg.norm(out_dir)
        if out_len < 1e-9:
            out_dir = n_out if np.linalg.norm(n_out) > 1e-9 else np.array([1.0,0.0])
            out_len = np.linalg.norm(out_dir)
        out_dir /= out_len
        text_pos = p_curr + out_dir * angle_offset
        ax.text(text_pos[0], text_pos[1], format_angle(ang),
                fontsize=fontsize, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1),
                zorder=5)
    return ax

def clean_coords(coords, eps=1e-8):
    pts = [tuple(p) for p in coords]
    new = []
    for p in pts:
        if not new or (abs(p[0] - new[-1][0]) > eps or abs(p[1] - new[-1][1]) > eps):
            new.append(p)
    if len(new) > 1 and (abs(new[0][0] - new[-1][0]) < eps and abs(new[0][1] - new[-1][1]) < eps):
        new.pop()
    if len(new) < 3:
        return np.array(new)
    final = [new[0]]
    for i in range(1, len(new) - 1):
        A, B, C = np.array(final[-1]), np.array(new[i]), np.array(new[i + 1])
        cross = (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
        if abs(cross) <= eps:
            continue
        final.append(new[i])
    final.append(new[-1])
    return np.array(final)

def compute_offset_polygon(outer_poly, offset):
    if offset <= 0:
        return outer_poly
    try:
        inner = outer_poly.buffer(-float(offset))
    except Exception:
        return None
    if inner is None or inner.is_empty:
        return None
    if inner.geom_type == "MultiPolygon":
        largest = max(inner.geoms, key=lambda p: p.area)
        return largest
    if inner.geom_type == "Polygon":
        return inner
    return None

def rotate_coords(coords, R):
    arr = np.array(coords)
    pts = (arr @ R.T)
    return [tuple(p) for p in pts]

def exact_rotation_matrix(angle_deg):
    a = int(angle_deg) % 360
    if a % 90 == 0:
        if a == 0:
            return np.array([[1.0, 0.0], [0.0, 1.0]])
        if a == 90:
            return np.array([[0.0, -1.0], [1.0, 0.0]])
        if a == 180:
            return np.array([[-1.0, 0.0], [0.0, -1.0]])
        if a == 270:
            return np.array([[0.0, 1.0], [-1.0, 0.0]])
    rad = np.radians(angle_deg)
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

# ---------------- hlavní aplikace ----------------
class SklenenyPanelApp:
    def __init__(self, parent, back_callback=None):
        self.root = parent
        self.back_callback = back_callback
        try:
            self.root.winfo_toplevel().title("Skleněný panel – drážky a řezy (v2)")
        except Exception:
            pass

        # params
        self.rotation_angle = 0
        self.frame_depth = 30.0
        self.glass_groove = 6.0
        self.show_profiles = True
        self.debug = False

        self.cut_order = {}
        self._cut_counter = 1

        # GUI
        tk.Label(self.root, text="Délka (mm):").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.base_entry = tk.Entry(self.root, width=8); self.base_entry.insert(0, "500"); self.base_entry.grid(row=0, column=1)

        tk.Label(self.root, text="Výška vlevo (mm):").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        self.left_entry = tk.Entry(self.root, width=8); self.left_entry.insert(0, "500"); self.left_entry.grid(row=0, column=3)

        tk.Label(self.root, text="Výška vpravo (mm):").grid(row=0, column=4, padx=4, pady=4, sticky="e")
        self.right_entry = tk.Entry(self.root, width=8); self.right_entry.insert(0, "2000"); self.right_entry.grid(row=0, column=5)

        tk.Button(self.root, text="Nakreslit", command=self.draw_shape, bg="#4CAF50", fg="white").grid(row=0, column=6, padx=6)
        tk.Button(self.root, text="Otočit", command=self.rotate_shape, bg="#2196F3", fg="white").grid(row=0, column=7, padx=6)
        tk.Button(self.root, text="Reset", command=self.reset_canvas, bg="#f44336", fg="white").grid(row=0, column=8, padx=6)
        tk.Button(self.root, text="Profily", command=self.toggle_profiles).grid(row=0, column=9, padx=6)

        tk.Label(self.root, text="Řezy:").grid(row=1, column=0, sticky="e", padx=4)
        self.cut_vars = {
            "left": tk.BooleanVar(value=True),
            "right": tk.BooleanVar(value=True),
            "top": tk.BooleanVar(value=False),
            "bottom": tk.BooleanVar(value=False),
        }
        for i, name in enumerate(["left", "right", "top", "bottom"], start=1):
            tk.Checkbutton(self.root, text=name.capitalize(), variable=self.cut_vars[name],
                           command=lambda n=name: self.on_toggle_cut(n)).grid(row=1, column=i)

        self._create_figure_layout()

        for name in ["left", "right"]:
            if self.cut_vars[name].get():
                self.cut_order[name] = self._cut_counter
                self._cut_counter += 1

    def go_back(self):
        for w in self.root.winfo_children():
            w.destroy()
        if self.back_callback:
            self.back_callback()

    def on_toggle_cut(self, name):
        if self.cut_vars[name].get():
            if name not in self.cut_order:
                self.cut_order[name] = self._cut_counter
                self._cut_counter += 1
        else:
            if name in self.cut_order:
                del self.cut_order[name]
        if getattr(self, 'debug', False):
            print(f"[DEBUG] on_toggle_cut: {name} -> {self.cut_vars[name].get()}, cut_order={self.cut_order}")
        self.draw_shape()

    def draw_shape(self):
        try:
            base = float(self.base_entry.get())
            left_h = float(self.left_entry.get())
            right_h = float(self.right_entry.get())
        except ValueError:
            print("Zadej platné číselné hodnoty.")
            return

        frame = self.frame_depth
        inset = frame - self.glass_groove
        R = exact_rotation_matrix(self.rotation_angle)

        self.ax_main.clear(); self.ax_glass.clear()
        for ax in self.profile_axes:
            ax.clear(); ax.axis('off')

        self.ax_main.set_aspect('equal'); self.ax_main.grid(False); self.ax_main.axis('off')
        self.ax_glass.set_aspect('equal'); self.ax_glass.axis('off')

        self._draw_window(self.ax_main, 0.0, base, left_h, right_h, frame, inset, R, draw_frame=True)
        self._draw_window(self.ax_glass, 0.0, base, left_h, right_h, frame, inset, R, draw_frame=False, fit_to_axis=True)

        profiles = [("Levý", left_h), ("Horní", base), ("Pravý", right_h), ("Dolní", base)]
        for i, ax in enumerate(self.profile_axes):
            ax.clear(); ax.axis('off')
            if i < len(profiles):
                name, real_len = profiles[i]
                ax.set_aspect('equal')
                self._draw_profile(ax, name, real_len, frame)

        try:
            self.canvas.draw_idle()
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.update_idletasks()
            bbox = canvas_widget.bbox('all')
            if bbox:
                canvas_widget.config(scrollregion=bbox)
        except Exception:
            try:
                self.canvas.draw()
            except Exception:
                pass

    def _create_figure_layout(self):
        try:
            if hasattr(self, 'canvas'):
                widget = self.canvas.get_tk_widget()
                try:
                    widget.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            screen_w = self.root.winfo_screenwidth(); screen_h = self.root.winfo_screenheight()
            safe_h = min(screen_h * 0.85, 900); safe_w = min(screen_w * 0.9, 1100)
            fig_h = safe_h / 96.0; fig_w = safe_w / 96.0
        except Exception:
            fig_w, fig_h = 11.0, 9.0

        self.fig = plt.Figure(figsize=(fig_w, fig_h))

        if self.show_profiles:
            height_ratios = [3, 3, 0.05, 1.2]
        else:
            height_ratios = [4.5, 4.5, 0.05, 0.1]

        hspace = 0.25; wspace = 0.25
        gs = self.fig.add_gridspec(4, 4, height_ratios=height_ratios, width_ratios=[1,1,1,1], hspace=hspace, wspace=wspace)

        self.ax_main = self.fig.add_subplot(gs[0, 0:4])
        self.ax_glass = self.fig.add_subplot(gs[1, 0:4])

        self.ax_p0 = self.fig.add_subplot(gs[3, 0])
        self.ax_p1 = self.fig.add_subplot(gs[3, 1])
        self.ax_p2 = self.fig.add_subplot(gs[3, 2])
        self.ax_p3 = self.fig.add_subplot(gs[3, 3])
        self.profile_axes = [self.ax_p0, self.ax_p1, self.ax_p2, self.ax_p3]

        self.ax_main.set_aspect('equal'); self.ax_main.grid(False); self.ax_main.axis('off')
        self.ax_glass.set_aspect('equal'); self.ax_glass.axis('off')
        for ax in self.profile_axes:
            ax.set_aspect('equal'); ax.axis('off')

        canvas_frame = tk.Frame(self.root)
        canvas_frame.grid(row=2, column=0, columnspan=11, sticky='nsew', padx=8, pady=8)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(10, weight=1)

        vscroll = tk.Scrollbar(canvas_frame, orient='vertical')
        hscroll = tk.Scrollbar(canvas_frame, orient='horizontal')

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        canvas_widget = self.canvas.get_tk_widget()

        canvas_widget.grid(row=0, column=0, sticky='nsew')
        vscroll.grid(row=0, column=1, sticky='ns')
        hscroll.grid(row=1, column=0, sticky='ew')

        vscroll.config(command=canvas_widget.yview)
        hscroll.config(command=canvas_widget.xview)
        canvas_widget.config(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

    def toggle_profiles(self):
        self.show_profiles = not getattr(self, 'show_profiles', True)
        self._create_figure_layout()
        try:
            self.draw_shape()
        except Exception:
            try:
                self.canvas.draw_idle()
            except Exception:
                try:
                    self.canvas.draw()
                except Exception:
                    pass

    def _draw_window(self, ax, offset_x, base, left_h, right_h, frame, inset, R, draw_frame=True, fit_to_axis=False):
        shrink = 0.0
        A = np.array([offset_x + shrink, 0 + shrink])
        B = np.array([offset_x + base - shrink, 0 + shrink])
        C = np.array([offset_x + base - shrink, right_h - shrink])
        D = np.array([offset_x + shrink, left_h - shrink])
        pts = np.array([A, B, C, D, A])

        pts_rot = pts @ R.T
        if draw_frame:
            self.last_points = pts_rot
            self.outer_poly = Polygon(pts_rot)
            self.rotation_matrix = R

        outer_poly = Polygon(pts_rot)
        inner_poly = compute_offset_polygon(outer_poly, frame)
        glass_poly = compute_offset_polygon(outer_poly, frame - self.glass_groove)

        frame_poly = outer_poly
        if inner_poly is not None:
            try:
                frame_poly = outer_poly.difference(inner_poly).buffer(0)
            except Exception:
                frame_poly = outer_poly

        cuts = {
            "left": box(offset_x, -10000, offset_x + frame, 10000),
            "right": box(offset_x + base - frame, -10000, offset_x + base, 10000),
            "top": box(offset_x, max(left_h, right_h) - frame, offset_x + base, max(left_h, right_h)),
            "bottom": box(offset_x, 0, offset_x + base, frame)
        }
        for k in list(cuts.keys()):
            cuts[k] = Polygon(rotate_coords(list(cuts[k].exterior.coords), R))

        side_pieces = {}
        for cname in ['left', 'right', 'top', 'bottom']:
            try:
                pc = frame_poly.intersection(cuts[cname])
                if not pc.is_empty:
                    side_pieces[cname] = pc
            except Exception:
                pass

        pieces = {}
        if draw_frame:
            existing = None
            for name, _ in sorted(self.cut_order.items(), key=lambda kv: kv[1]):
                if not self.cut_vars[name].get():
                    continue
                piece = side_pieces.get(name)
                if piece is None:
                    continue
                if existing is not None and not existing.is_empty:
                    piece = piece.difference(existing)
                if not piece.is_empty:
                    pieces[name] = piece
                    existing = piece if existing is None else unary_union([existing, piece])

        # compute piece_runs and piece_lengths (reálné délky podle výsledných geometrií)
        try:
            applied_union = None
            for name, _ in sorted(self.cut_order.items(), key=lambda kv: kv[1]):
                if not self.cut_vars.get(name, tk.BooleanVar()).get():
                    continue
                p = pieces.get(name)
                if p is None:
                    continue
                applied_union = p if applied_union is None else unary_union([applied_union, p])

            rest = frame_poly
            if applied_union is not None:
                try:
                    rest = frame_poly.difference(applied_union)
                except Exception:
                    rest = frame_poly

            self.piece_runs = {}
            self.piece_lengths = {}
            for side in ['left', 'right', 'top', 'bottom']:
                target_poly = None
                if side in pieces:
                    target_poly = pieces[side]
                else:
                    try:
                        target_poly = rest.intersection(cuts[side])
                    except Exception:
                        target_poly = None

                if target_poly is None or getattr(target_poly, 'is_empty', False):
                    self.piece_runs[side] = None
                    self.piece_lengths[side] = None
                    continue

                run = self._main_edge_run(target_poly)
                self.piece_runs[side] = run
                plen = None
                try:
                    if run is not None:
                        start_pt, end_pt, run_len, ux, uy = run
                        midx = (start_pt[0] + end_pt[0]) / 2.0
                        midy = (start_pt[1] + end_pt[1]) / 2.0
                        BIG = max(20000.0, run_len * 10.0)
                        p0 = (midx - ux * BIG, midy - uy * BIG)
                        p1 = (midx + ux * BIG, midy + uy * BIG)
                        test_line = LineString([p0, p1])
                        inter = target_poly.intersection(test_line)
                        total = 0.0
                        if inter is None or getattr(inter, 'is_empty', False):
                            total = run_len
                        else:
                            if inter.geom_type == 'LineString':
                                total = inter.length
                            elif inter.geom_type == 'MultiLineString':
                                total = sum(seg.length for seg in inter.geoms)
                            else:
                                total = run_len
                        plen = total
                except Exception:
                    plen = run[2] if run is not None else None
                self.piece_lengths[side] = float(plen) if plen is not None else None
        except Exception:
            self.piece_runs = {}
            self.piece_lengths = {}

        if getattr(self, 'debug', False):
            print("[DEBUG] pieces:", list(pieces.keys()))
            print("[DEBUG] piece_lengths:", self.piece_lengths)

        # vykreslení rámu a pieceů
        if draw_frame:
            rest_area = frame_poly
            for geom in pieces.values():
                rest_area = rest_area.difference(geom)
            for g in (rest_area.geoms if hasattr(rest_area, "geoms") else [rest_area]):
                ax.fill(*g.exterior.xy, color="#d3d3d3")
                ax.plot(*g.exterior.xy, "k-", lw=0.8)
            for g in pieces.values():
                for sub in (g.geoms if hasattr(g, "geoms") else [g]):
                    ax.fill(*sub.exterior.xy, color="#b0b0b0")
                    ax.plot(*sub.exterior.xy, "k-", lw=0.8)

        # glass
        if glass_poly is not None and not glass_poly.is_empty:
            for g in (glass_poly.geoms if hasattr(glass_poly, "geoms") else [glass_poly]):
                ax.fill(*g.exterior.xy, color="#aee8ff", alpha=0.7)
                ax.plot(*g.exterior.xy, "k-", lw=0.6)

        # annotations
        try:
            if draw_frame and outer_poly is not None:
                poly = outer_poly
                if getattr(poly, "geom_type", "") == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda p: p.area)
                coords = list(poly.exterior.coords)
                lens = []
                for i in range(len(coords)-1):
                    p0 = coords[i]; p1 = coords[i+1]
                    lens.append(math.hypot(p1[0]-p0[0], p1[1]-p0[1]))
                draw_polygon_annotations(coords, lens, ax, outward_offset=None, fontsize=9)

            if (not draw_frame) and glass_poly is not None:
                gpoly = glass_poly
                if getattr(gpoly, "geom_type", "") == "MultiPolygon":
                    gpoly = max(gpoly.geoms, key=lambda p: p.area)
                coords_g = list(gpoly.exterior.coords)
                lens_g = []
                for i in range(len(coords_g)-1):
                    q0 = coords_g[i]; q1 = coords_g[i+1]
                    lens_g.append(math.hypot(q1[0]-q0[0], q1[1]-q0[1]))
                draw_polygon_annotations(coords_g, lens_g, ax, outward_offset=None, fontsize=9)
        except Exception:
            pass

        ax.axis("off")

        # nastav rozsah
        bboxes = []
        for poly in (outer_poly, frame_poly, glass_poly):
            if poly is None:
                continue
            try:
                if not poly.is_empty:
                    bboxes.append(poly.bounds)
            except Exception:
                pass
        for p in pieces.values():
            try:
                if not p.is_empty:
                    bboxes.append(p.bounds)
            except Exception:
                pass

        if bboxes:
            minx = min(b[0] for b in bboxes)
            miny = min(b[1] for b in bboxes)
            maxx = max(b[2] for b in bboxes)
            maxy = max(b[3] for b in bboxes)
            padx = (maxx - minx) * 0.12 + 1.0
            pady = (maxy - miny) * 0.12 + 1.0
            ax.set_xlim(minx - padx, maxx + padx)
            ax.set_ylim(miny - pady, maxy + pady)

        try:
            ax.relim()
            ax.autoscale_view()
        except Exception:
            pass

        # spočti corner_angles pro pozdější použití v profilech
        try:
            if draw_frame and hasattr(self, "outer_poly") and self.outer_poly is not None:
                coords = list(self.outer_poly.exterior.coords)
                if np.allclose(coords[0], coords[-1]):
                    coords = coords[:-1]
                n = len(coords)
                corner_angles = {}
                # vypočti úhly (vnitřní) pro každý vrchol
                for i in range(n):
                    p_prev = np.array(coords[(i-1)%n])
                    p_curr = np.array(coords[i])
                    p_next = np.array(coords[(i+1)%n])
                    v1 = p_prev - p_curr
                    v2 = p_next - p_curr
                    if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                        ang = 90.0
                    else:
                        ang = math.degrees(math.acos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)))
                    corner_angles[i] = ang
                self.corner_angles = {"coords": coords, "angles": corner_angles}
        except Exception:
            self.corner_angles = None

        if fit_to_axis:
            try:
                self.canvas.draw_idle()
            except Exception:
                try:
                    self.canvas.draw()
                except Exception:
                    pass

    def _edge_direction_for_side(self, outer_poly, side_cut_poly):
        import numpy as np
        if outer_poly is None:
            return np.array([1.0, 0.0])
        try:
            exterior = list(outer_poly.exterior.coords)
        except Exception:
            return np.array([1.0, 0.0])
        try:
            c = np.array(side_cut_poly.centroid.coords[0])
        except Exception:
            c = np.array(outer_poly.centroid.coords[0])
        best_idx = 0
        best_dist = float('inf')
        n = len(exterior) - 1
        for i in range(n):
            a = np.array(exterior[i]); b = np.array(exterior[i+1])
            mid = (a + b) / 2.0
            d = np.linalg.norm(mid - c)
            if d < best_dist:
                best_dist = d; best_idx = i
        a = np.array(exterior[best_idx]); b = np.array(exterior[(best_idx + 1) % n])
        v = b - a
        L = np.linalg.norm(v)
        if L < 1e-9:
            return np.array([1.0, 0.0])
        return v / L

    def _main_edge_run(self, poly):
        import math, numpy as np
        if poly is None:
            return None
        try:
            geoms = poly.geoms if hasattr(poly, 'geoms') else [poly]
            # combine exteriors
            coords = []
            for sub in geoms:
                coords += list(sub.exterior.coords)
            if len(coords) < 2:
                return None
            n = len(coords) - 1
            best = (None, None, 0.0, 0.0, 0.0)
            i = 0
            while i < n:
                p0 = coords[i]; p1 = coords[i+1]
                v0 = (p1[0]-p0[0], p1[1]-p0[1])
                L0 = math.hypot(v0[0], v0[1])
                if L0 < 1e-9:
                    i += 1; continue
                ux0, uy0 = v0[0]/L0, v0[1]/L0
                run_len = L0
                start_pt = p0; end_pt = p1
                j = i+1
                while j < n:
                    pnext = coords[j+1]
                    v1 = (pnext[0]-coords[j][0], pnext[1]-coords[j][1])
                    L1 = math.hypot(v1[0], v1[1])
                    if L1 < 1e-9:
                        break
                    ux1, uy1 = v1[0]/L1, v1[1]/L1
                    dot = ux0*ux1 + uy0*uy1
                    if abs(abs(dot) - 1.0) < 1e-3:
                        run_len += L1
                        end_pt = pnext
                        j += 1
                    else:
                        break
                if run_len > best[2]:
                    best = (start_pt, end_pt, run_len, ux0, uy0)
                i = max(i+1, j)
            return best if best[2] > 1e-6 else None
        except Exception:
            return None

    def _find_nearest_vertex_index(self, point, coords):
        # coords: list of (x,y) closed or open (we handle both)
        arr = np.array(coords)
        if arr.shape[0] == 0:
            return None
        dists = np.hypot(arr[:,0] - point[0], arr[:,1] - point[1])
        return int(np.argmin(dists))

    def _draw_profile(self, ax, name, real_len, frame):
        if real_len is None or real_len <= 0:
            return
        PROFILE_HEIGHT = 30.0
        name_key = {'Levý':'left','Pravý':'right','Horní':'top','Dolní':'bottom'}.get(name, None)
        if name_key is None:
            return

        # prefer reálnou délku z výpočtu (piece_lengths)
        length_val = None
        if hasattr(self, "piece_lengths"):
            length_val = self.piece_lengths.get(name_key)
        if length_val is None or length_val <= 0:
            # fallback: use provided real_len (geometric side length)
            length_val = real_len

        # run for angles: use piece_runs if available
        run = None
        if hasattr(self, "piece_runs"):
            run = self.piece_runs.get(name_key)

        # compute end angles by finding nearest vertices to run endpoints (if possible)
        ang_start = 90.0; ang_end = 90.0
        try:
            if run is not None and run[0] is not None and run[1] is not None and hasattr(self, "outer_poly"):
                start_pt = run[0]; end_pt = run[1]
                coords = list(self.outer_poly.exterior.coords)
                if np.allclose(coords[0], coords[-1]):
                    coords = coords[:-1]
                idx_s = self._find_nearest_vertex_index(start_pt, coords)
                idx_e = self._find_nearest_vertex_index(end_pt, coords)
                n = len(coords)
                def angle_at_index(i):
                    p_prev = np.array(coords[(i-1)%n]); p_curr = np.array(coords[i]); p_next = np.array(coords[(i+1)%n])
                    v1 = p_prev - p_curr; v2 = p_next - p_curr
                    if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                        return 90.0
                    return math.degrees(math.acos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)))
                if idx_s is not None:
                    ang_start = angle_at_index(idx_s)
                if idx_e is not None:
                    ang_end = angle_at_index(idx_e)
        except Exception:
            ang_start = ang_end = 90.0

        cut_angles = (ang_start, ang_end)

        # vykreslení profilu (dvojí - jednoduché a vizualizační, podobně jako v původním kódu)
        scale = min(800.0 / max(length_val, 1.0), 25.0)
        w = length_val * scale
        h = PROFILE_HEIGHT * scale
        x0, y0 = 20.0, 20.0

        def dx_from_angle(a):
            if a is None or a <= 0 or a >= 90:
                return 0.0
            return h * math.tan(math.radians(90 - a))

        dx1 = dx_from_angle(cut_angles[0])
        dx2 = dx_from_angle(cut_angles[1])

        points = [
            (x0 + dx1, y0),
            (x0, y0 + h),
            (x0 + w, y0 + h),
            (x0 + w - dx2, y0)
        ]

        poly = plt.Polygon(points, facecolor='white', edgecolor='black', linewidth=1.2)
        ax.add_patch(poly)
        ax.text(x0 + w/2, y0 - 12, f"{length_val:.1f}", ha='center', va='top', fontsize=9)
        ax.text(x0 + w/2, y0 - 28, name, ha='center', va='top', fontsize=9)
        ax.text(points[0][0], y0 - 6, f"{cut_angles[0]:.1f}°", ha='right', va='top', fontsize=8)
        ax.text(points[3][0], y0 - 6, f"{cut_angles[1]:.1f}°", ha='left', va='top', fontsize=8)

        ax.set_xlim(x0 - 50, x0 + w + 50)
        ax.set_ylim(y0 - 50, y0 + h + 50)
        ax.set_aspect('equal')
        ax.axis('off')

        # doplňkové vykreslení (stejný vizuální blok, zachováno pro konzistenci)
        # (zbytek duplikátu je záměrně zachován, aby vizuál byl konzistentní)
        # (nelze dále zkracovat — pro čitelnost)
        if getattr(self, 'debug', False):
            print(f"[PROFILE] {name}: len={length_val:.2f}  angles=({cut_angles[0]:.2f},{cut_angles[1]:.2f})")

    def rotate_shape(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.draw_shape()

    def reset_canvas(self):
        self.rotation_angle = 0
        self.ax_main.clear(); self.ax_glass.clear()
        for ax in self.profile_axes:
            ax.clear(); ax.axis('off')
        self.ax_main.grid(False)
        try:
            self.canvas.draw()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SklenenyPanelApp(root)
    # app.debug = True  # odkomentuj pro debug výpisy
    root.mainloop() 
