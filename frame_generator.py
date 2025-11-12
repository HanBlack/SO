import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom

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
    ax.plot(xs, ys, color='black', linewidth=1.0)

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
            self.root.winfo_toplevel().title("Skleněný panel – XML Export souřadnic")
        except Exception:
            pass

        # params
        self.rotation_angle = 0
        self.frame_depth = 30.0
        self.glass_groove = 6.0
        self.debug = False

        # GUI
        tk.Label(self.root, text="Délka (mm):").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.base_entry = tk.Entry(self.root, width=8)
        self.base_entry.insert(0, "500")
        self.base_entry.grid(row=0, column=1)

        tk.Label(self.root, text="Výška vlevo (mm):").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        self.left_entry = tk.Entry(self.root, width=8)
        self.left_entry.insert(0, "500")
        self.left_entry.grid(row=0, column=3)

        tk.Label(self.root, text="Výška vpravo (mm):").grid(row=0, column=4, padx=4, pady=4, sticky="e")
        self.right_entry = tk.Entry(self.root, width=8)
        self.right_entry.insert(0, "2000")
        self.right_entry.grid(row=0, column=5)

        tk.Button(self.root, text="Nakreslit", command=self.draw_shape, bg="#4CAF50", fg="white").grid(row=0, column=6, padx=6)
        tk.Button(self.root, text="Otočit", command=self.rotate_shape, bg="#2196F3", fg="white").grid(row=0, column=7, padx=6)
        tk.Button(self.root, text="Reset", command=self.reset_canvas, bg="#f44336", fg="white").grid(row=0, column=8, padx=6)
        tk.Button(self.root, text="Export XML", command=self.export_xml, bg="#FF9800", fg="white").grid(row=0, column=9, padx=6)

        self._create_figure_layout()

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
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            safe_h = min(screen_h * 0.85, 900)
            safe_w = min(screen_w * 0.9, 1100)
            fig_h = safe_h / 96.0
            fig_w = safe_w / 96.0
        except Exception:
            fig_w, fig_h = 11.0, 9.0

        self.fig = plt.Figure(figsize=(fig_w, fig_h))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        self.ax_frame = self.fig.add_subplot(gs[0, 0])
        self.ax_glass = self.fig.add_subplot(gs[1, 0])

        self.ax_frame.set_aspect('equal')
        self.ax_frame.grid(False)
        self.ax_frame.axis('off')
        self.ax_frame.set_title("Rám okna", fontsize=10)
        
        self.ax_glass.set_aspect('equal')
        self.ax_glass.axis('off')
        self.ax_glass.set_title("Sklo", fontsize=10)

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

    def draw_shape(self):
        try:
            base = float(self.base_entry.get())
            left_h = float(self.left_entry.get())
            right_h = float(self.right_entry.get())
        except ValueError:
            messagebox.showerror("Chyba", "Zadej platné číselné hodnoty.")
            return

        frame = self.frame_depth
        R = exact_rotation_matrix(self.rotation_angle)

        self.ax_frame.clear()
        self.ax_glass.clear()

        self.ax_frame.set_aspect('equal')
        self.ax_frame.grid(False)
        self.ax_frame.axis('off')
        self.ax_frame.set_title("Rám okna", fontsize=10)
        
        self.ax_glass.set_aspect('equal')
        self.ax_glass.axis('off')
        self.ax_glass.set_title("Sklo", fontsize=10)

        # Vytvoření základních bodů
        A = np.array([0.0, 0.0])
        B = np.array([base, 0.0])
        C = np.array([base, right_h])
        D = np.array([0.0, left_h])
        pts = np.array([A, B, C, D, A])

        # Rotace
        pts_rot = pts @ R.T
        
        # Uložení pro XML export
        self.frame_vertices = pts_rot[:-1]  # bez duplikátu posledního bodu
        
        # Výpočet středu
        center_x = np.mean(self.frame_vertices[:, 0])
        center_y = np.mean(self.frame_vertices[:, 1])
        self.center = np.array([center_x, center_y])
        
        # Přepočet souřadnic relativně ke středu
        self.frame_vertices_centered = self.frame_vertices - self.center

        # Vytvoření polygonů
        outer_poly = Polygon(pts_rot)
        inner_poly = compute_offset_polygon(outer_poly, frame)
        glass_poly = compute_offset_polygon(outer_poly, frame - self.glass_groove)

        # Uložení pro XML export
        if glass_poly is not None and not glass_poly.is_empty:
            glass_coords = np.array(glass_poly.exterior.coords[:-1])
            self.glass_vertices_centered = glass_coords - self.center
        else:
            self.glass_vertices_centered = None

        # Vykreslení rámu
        frame_poly = outer_poly
        if inner_poly is not None:
            try:
                frame_poly = outer_poly.difference(inner_poly).buffer(0)
            except Exception:
                frame_poly = outer_poly

        for g in (frame_poly.geoms if hasattr(frame_poly, "geoms") else [frame_poly]):
            self.ax_frame.fill(*g.exterior.xy, color="#d3d3d3")
            self.ax_frame.plot(*g.exterior.xy, "k-", lw=0.8)

        # Vykreslení skla
        if glass_poly is not None and not glass_poly.is_empty:
            for g in (glass_poly.geoms if hasattr(glass_poly, "geoms") else [glass_poly]):
                self.ax_glass.fill(*g.exterior.xy, color="#aee8ff", alpha=0.7)
                self.ax_glass.plot(*g.exterior.xy, "k-", lw=0.6)

        # Anotace pro rám
        coords = list(outer_poly.exterior.coords)
        lens = []
        for i in range(len(coords)-1):
            p0 = coords[i]
            p1 = coords[i+1]
            lens.append(math.hypot(p1[0]-p0[0], p1[1]-p0[1]))
        draw_polygon_annotations(coords, lens, self.ax_frame, outward_offset=None, fontsize=9)

        # Anotace pro sklo
        if glass_poly is not None:
            coords_g = list(glass_poly.exterior.coords)
            lens_g = []
            for i in range(len(coords_g)-1):
                q0 = coords_g[i]
                q1 = coords_g[i+1]
                lens_g.append(math.hypot(q1[0]-q0[0], q1[1]-q0[1]))
            draw_polygon_annotations(coords_g, lens_g, self.ax_glass, outward_offset=None, fontsize=9)

        # Vykreslení os X,Y procházejících středem
        for ax in [self.ax_frame, self.ax_glass]:
            # Určení rozsahu
            all_x = list(pts_rot[:, 0])
            all_y = list(pts_rot[:, 1])
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            # Osa X (horizontální)
            ax.plot([min_x - 50, max_x + 50], [center_y, center_y], 'r--', linewidth=0.5, alpha=0.5, label='Osa X')
            # Osa Y (vertikální)
            ax.plot([center_x, center_x], [min_y - 50, max_y + 50], 'b--', linewidth=0.5, alpha=0.5, label='Osa Y')
            
            # Označení středu
            ax.plot(center_x, center_y, 'ko', markersize=4)
            ax.text(center_x + 10, center_y + 10, f'Střed\n({center_x:.1f}, {center_y:.1f})', 
                   fontsize=8, ha='left', va='bottom',
                   bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='black', pad=2))

        # Nastavení rozsahu
        bboxes = []
        for poly in (outer_poly, frame_poly, glass_poly):
            if poly is None:
                continue
            try:
                if not poly.is_empty:
                    bboxes.append(poly.bounds)
            except Exception:
                pass

        if bboxes:
            minx = min(b[0] for b in bboxes)
            miny = min(b[1] for b in bboxes)
            maxx = max(b[2] for b in bboxes)
            maxy = max(b[3] for b in bboxes)
            padx = (maxx - minx) * 0.12 + 1.0
            pady = (maxy - miny) * 0.12 + 1.0
            
            for ax in [self.ax_frame, self.ax_glass]:
                ax.set_xlim(minx - padx, maxx + padx)
                ax.set_ylim(miny - pady, maxy + pady)

        try:
            self.canvas.draw_idle()
        except Exception:
            try:
                self.canvas.draw()
            except Exception:
                pass

    def export_xml(self):
        if not hasattr(self, 'frame_vertices_centered'):
            messagebox.showwarning("Varování", "Nejprve nakresli okno pomocí tlačítka 'Nakreslit'.")
            return

        # Vytvoření XML struktury
        root = ET.Element("Window")
        root.set("version", "1.0")
        
        # Informace o středu
        center_elem = ET.SubElement(root, "Center")
        center_elem.set("x", f"{self.center[0]:.2f}")
        center_elem.set("y", f"{self.center[1]:.2f}")
        
        # Rám okna
        frame_elem = ET.SubElement(root, "Frame")
        vertices_frame = ET.SubElement(frame_elem, "Vertices")
        
        for i, vertex in enumerate(self.frame_vertices_centered):
            v_elem = ET.SubElement(vertices_frame, "Vertex")
            v_elem.set("id", str(i))
            v_elem.set("x", f"{vertex[0]:.2f}")
            v_elem.set("y", f"{vertex[1]:.2f}")
        
        # Úhly rámu
        angles_frame = compute_inner_angles(self.frame_vertices_centered)
        angles_elem = ET.SubElement(frame_elem, "Angles")
        for i, angle in enumerate(angles_frame):
            a_elem = ET.SubElement(angles_elem, "Angle")
            a_elem.set("vertex_id", str(i))
            a_elem.set("degrees", f"{angle:.2f}")
        
        # Sklo
        if self.glass_vertices_centered is not None:
            glass_elem = ET.SubElement(root, "Glass")
            vertices_glass = ET.SubElement(glass_elem, "Vertices")
            
            for i, vertex in enumerate(self.glass_vertices_centered):
                v_elem = ET.SubElement(vertices_glass, "Vertex")
                v_elem.set("id", str(i))
                v_elem.set("x", f"{vertex[0]:.2f}")
                v_elem.set("y", f"{vertex[1]:.2f}")
            
            # Úhly skla
            angles_glass = compute_inner_angles(self.glass_vertices_centered)
            angles_elem = ET.SubElement(glass_elem, "Angles")
            for i, angle in enumerate(angles_glass):
                a_elem = ET.SubElement(angles_elem, "Angle")
                a_elem.set("vertex_id", str(i))
                a_elem.set("degrees", f"{angle:.2f}")
        
        # Formátování XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
        # Uložení do souboru
        filename = filedialog.asksaveasfilename(
            defaultextension=".xml",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
            title="Uložit XML souřadnice"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(xml_str)
                messagebox.showinfo("Úspěch", f"XML soubor byl úspěšně uložen:\n{filename}")
            except Exception as e:
                messagebox.showerror("Chyba", f"Nepodařilo se uložit soubor:\n{str(e)}")

    def rotate_shape(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.draw_shape()

    def reset_canvas(self):
        self.rotation_angle = 0
        self.ax_frame.clear()
        self.ax_glass.clear()
        self.ax_frame.grid(False)
        self.ax_frame.set_title("Rám okna", fontsize=10)
        self.ax_glass.set_title("Sklo", fontsize=10)
        try:
            self.canvas.draw()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SklenenyPanelApp(root)
    root.mainloop()
