# frame_generator_with_profiles_final.py
# Finální verze s SPRÁVNÝM výpočtem profilů

import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Import původních funkcí
from frame_generator import (
    format_length, format_angle, compute_inner_angles, is_ccw,
    draw_polygon_annotations, compute_offset_polygon, exact_rotation_matrix
)

# Import SPRÁVNÉHO modulu pro profily
from profile_generator import calculate_profile_dimensions, draw_all_profiles, draw_profile_dimensions_table


class SklenenyPanelAppWithProfiles:
    def __init__(self, parent, back_callback=None):
        self.root = parent
        self.back_callback = back_callback
        try:
            self.root.winfo_toplevel().title("Skleněný panel s profily - SPRÁVNÁ verze")
        except Exception:
            pass

        # Parametry
        self.rotation_angle = 0
        self.frame_depth = 30.0
        self.glass_groove = 6.0

        # GUI - První řádek
        tk.Label(self.root, text="Délka (mm):").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.base_entry = tk.Entry(self.root, width=8)
        self.base_entry.insert(0, "1934")
        self.base_entry.grid(row=0, column=1)

        tk.Label(self.root, text="Výška vlevo (mm):").grid(row=0, column=2, padx=4, pady=4, sticky="e")
        self.left_entry = tk.Entry(self.root, width=8)
        self.left_entry.insert(0, "862")
        self.left_entry.grid(row=0, column=3)

        tk.Label(self.root, text="Výška vpravo (mm):").grid(row=0, column=4, padx=4, pady=4, sticky="e")
        self.right_entry = tk.Entry(self.root, width=8)
        self.right_entry.insert(0, "1115")
        self.right_entry.grid(row=0, column=5)

        # Tlačítka
        tk.Button(self.root, text="Nakreslit", command=self.draw_shape, bg="#4CAF50", fg="white").grid(row=0, column=6, padx=6)
        tk.Button(self.root, text="Otočit", command=self.rotate_shape, bg="#2196F3", fg="white").grid(row=0, column=7, padx=6)
        tk.Button(self.root, text="Reset", command=self.reset_canvas, bg="#f44336", fg="white").grid(row=0, column=8, padx=6)
        tk.Button(self.root, text="Export XML", command=self.export_xml, bg="#FF9800", fg="white").grid(row=0, column=9, padx=6)

        # Přepínač zobrazení profilů
        self.show_profiles_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Zobrazit profily", variable=self.show_profiles_var, 
                      command=self.toggle_profiles).grid(row=0, column=10, padx=6)

        # Druhý řádek - informace o vypočtených rozměrech
        info_frame = tk.LabelFrame(self.root, text="Vypočtené rozměry profilů", padx=10, pady=5)
        info_frame.grid(row=1, column=0, columnspan=11, padx=10, pady=5, sticky="ew")
        
        self.profile_info_text = tk.Text(info_frame, height=6, width=120, font=("Courier", 9))
        self.profile_info_text.pack(fill="both", expand=True)
        
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
            safe_h = min(screen_h * 0.85, 1000)
            safe_w = min(screen_w * 0.9, 1400)
            fig_h = safe_h / 96.0
            fig_w = safe_w / 96.0
        except Exception:
            fig_w, fig_h = 14.0, 10.0

        self.fig = plt.Figure(figsize=(fig_w, fig_h))
        
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], 
                                   hspace=0.3, wspace=0.3)

        self.ax_frame = self.fig.add_subplot(gs[0, 0])
        self.ax_glass = self.fig.add_subplot(gs[1, 0])
        self.ax_profiles = self.fig.add_subplot(gs[0, 1])
        self.ax_table = self.fig.add_subplot(gs[1, 1])

        for ax in [self.ax_frame, self.ax_glass, self.ax_profiles]:
            ax.set_aspect('equal')
            ax.grid(False)
            ax.axis('off')

        self.ax_table.axis('off')
        
        self.ax_frame.set_title("Rám okna", fontsize=10)
        self.ax_glass.set_title("Sklo", fontsize=10)
        self.ax_profiles.set_title("Boční profily", fontsize=10)
        self.ax_table.set_title("Rozměry profilů", fontsize=10)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.grid(row=3, column=0, columnspan=11, sticky='nsew', padx=8, pady=8)
        self.root.grid_rowconfigure(3, weight=1)
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

    def update_profile_info(self, profiles):
        """Aktualizuje textové pole s informacemi o profilech"""
        self.profile_info_text.delete(1.0, tk.END)
        
        info = f"Tvar: {'Pravidelný (obdélník)' if profiles['is_rectangular'] else f'Atypický (sklon {profiles['top_angle']:.1f}°)'}\n"
        info += f"Tloušťka rámu: {profiles['frame_depth']:.1f} mm\n"
        info += f"Prodloužení: {profiles['extension']:.2f} mm\n\n"
        
        for name, label in [('left', 'Levý'), ('right', 'Pravý'), ('bottom', 'Dolní'), ('top', 'Horní')]:
            p = profiles[name]
            info += f"{label} profil: {p['outer_length']:.1f} x {p['width']:.0f} ({p.get('miter_start', 0):.1f}°) x {p['inner_length']:.1f} x {p['width']:.0f} ({p.get('miter_end', 0):.1f}°)\n"
        
        self.profile_info_text.insert(1.0, info)

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
        self.ax_profiles.clear()
        self.ax_table.clear()

        for ax in [self.ax_frame, self.ax_glass, self.ax_profiles]:
            ax.set_aspect('equal')
            ax.grid(False)
            ax.axis('off')

        self.ax_table.axis('off')
        
        self.ax_frame.set_title("Rám okna", fontsize=10)
        self.ax_glass.set_title("Sklo", fontsize=10)
        self.ax_profiles.set_title("Boční profily", fontsize=10)
        self.ax_table.set_title("Rozměry profilů", fontsize=10)

        # Vytvoření základních bodů
        A = np.array([0.0, 0.0])
        B = np.array([base, 0.0])
        C = np.array([base, right_h])
        D = np.array([0.0, left_h])
        pts = np.array([A, B, C, D, A])

        # Rotace
        pts_rot = pts @ R.T
        
        # Uložení pro výpočet profilů
        self.frame_vertices = pts_rot[:-1]
        
        # Výpočet středu
        center_x = np.mean(self.frame_vertices[:, 0])
        center_y = np.mean(self.frame_vertices[:, 1])
        self.center = np.array([center_x, center_y])
        
        self.frame_vertices_centered = self.frame_vertices - self.center

        # Vytvoření polygonů
        outer_poly = Polygon(pts_rot)
        inner_poly = compute_offset_polygon(outer_poly, frame)
        glass_poly = compute_offset_polygon(outer_poly, frame - self.glass_groove)

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

        # Anotace
        coords = list(outer_poly.exterior.coords)
        lens = []
        for i in range(len(coords)-1):
            p0 = coords[i]
            p1 = coords[i+1]
            lens.append(math.hypot(p1[0]-p0[0], p1[1]-p0[1]))
        draw_polygon_annotations(coords, lens, self.ax_frame, outward_offset=None, fontsize=9)

        if glass_poly is not None:
            coords_g = list(glass_poly.exterior.coords)
            lens_g = []
            for i in range(len(coords_g)-1):
                q0 = coords_g[i]
                q1 = coords_g[i+1]
                lens_g.append(math.hypot(q1[0]-q0[0], q1[1]-q0[1]))
            draw_polygon_annotations(coords_g, lens_g, self.ax_glass, outward_offset=None, fontsize=9)

        # Vykreslení os
        for ax in [self.ax_frame, self.ax_glass]:
            all_x = list(pts_rot[:, 0])
            all_y = list(pts_rot[:, 1])
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            ax.plot([min_x - 50, max_x + 50], [center_y, center_y], 'r--', linewidth=0.5, alpha=0.5)
            ax.plot([center_x, center_x], [min_y - 50, max_y + 50], 'b--', linewidth=0.5, alpha=0.5)
            
            ax.plot(center_x, center_y, 'ko', markersize=4)
            ax.text(center_x + 10, center_y + 10, f'Střed\n({center_x:.1f}, {center_y:.1f})', 
                   fontsize=8, ha='left', va='bottom',
                   bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='black', pad=2))

        # Výpočet a vykreslení profilů
        if self.show_profiles_var.get():
            try:
                profiles = calculate_profile_dimensions(self.frame_vertices, self.frame_depth)
                self.profiles = profiles
                
                self.update_profile_info(profiles)
                
                draw_all_profiles(self.ax_profiles, profiles)
                draw_profile_dimensions_table(self.ax_table, profiles)
            except Exception as e:
                print(f"Chyba při vykreslování profilů: {e}")
                import traceback
                traceback.print_exc()

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

    def toggle_profiles(self):
        self.draw_shape()

    def export_xml(self):
        if not hasattr(self, 'frame_vertices_centered'):
            messagebox.showwarning("Varování", "Nejprve nakresli okno pomocí tlačítka 'Nakreslit'.")
            return

        root = ET.Element("Window")
        root.set("version", "2.0")
        
        center_elem = ET.SubElement(root, "Center")
        center_elem.set("x", f"{self.center[0]:.2f}")
        center_elem.set("y", f"{self.center[1]:.2f}")
        
        frame_elem = ET.SubElement(root, "Frame")
        vertices_frame = ET.SubElement(frame_elem, "Vertices")
        
        for i, vertex in enumerate(self.frame_vertices_centered):
            v_elem = ET.SubElement(vertices_frame, "Vertex")
            v_elem.set("id", str(i))
            v_elem.set("x", f"{vertex[0]:.2f}")
            v_elem.set("y", f"{vertex[1]:.2f}")
        
        angles_frame = compute_inner_angles(self.frame_vertices_centered)
        angles_elem = ET.SubElement(frame_elem, "Angles")
        for i, angle in enumerate(angles_frame):
            a_elem = ET.SubElement(angles_elem, "Angle")
            a_elem.set("vertex_id", str(i))
            a_elem.set("degrees", f"{angle:.2f}")
        
        if self.glass_vertices_centered is not None:
            glass_elem = ET.SubElement(root, "Glass")
            vertices_glass = ET.SubElement(glass_elem, "Vertices")
            
            for i, vertex in enumerate(self.glass_vertices_centered):
                v_elem = ET.SubElement(vertices_glass, "Vertex")
                v_elem.set("id", str(i))
                v_elem.set("x", f"{vertex[0]:.2f}")
                v_elem.set("y", f"{vertex[1]:.2f}")
            
            angles_glass = compute_inner_angles(self.glass_vertices_centered)
            angles_elem = ET.SubElement(glass_elem, "Angles")
            for i, angle in enumerate(angles_glass):
                a_elem = ET.SubElement(angles_elem, "Angle")
                a_elem.set("vertex_id", str(i))
                a_elem.set("degrees", f"{angle:.2f}")
        
        if hasattr(self, 'profiles'):
            profiles_elem = ET.SubElement(root, "Profiles")
            profiles_elem.set("frame_depth", f"{self.frame_depth:.1f}")
            profiles_elem.set("top_angle", f"{self.profiles['top_angle']:.2f}")
            profiles_elem.set("extension", f"{self.profiles['extension']:.2f}")
            
            for name in ['left', 'right', 'bottom', 'top']:
                profile = self.profiles[name]
                p_elem = ET.SubElement(profiles_elem, "Profile")
                p_elem.set("name", name)
                p_elem.set("outer_length", f"{profile['outer_length']:.2f}")
                p_elem.set("inner_length", f"{profile['inner_length']:.2f}")
                p_elem.set("width", f"{profile['width']:.1f}")
                p_elem.set("angle", f"{profile['angle']:.2f}")
                p_elem.set("miter_start", f"{profile.get('miter_start', 0):.2f}")
                p_elem.set("miter_end", f"{profile.get('miter_end', 0):.2f}")
        
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
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
        self.ax_profiles.clear()
        self.ax_table.clear()
        self.profile_info_text.delete(1.0, tk.END)
        
        for ax in [self.ax_frame, self.ax_glass, self.ax_profiles]:
            ax.grid(False)
        
        self.ax_frame.set_title("Rám okna", fontsize=10)
        self.ax_glass.set_title("Sklo", fontsize=10)
        self.ax_profiles.set_title("Boční profily", fontsize=10)
        self.ax_table.set_title("Rozměry profilů", fontsize=10)
        
        try:
            self.canvas.draw()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = SklenenyPanelAppWithProfiles(root)
    root.mainloop()
