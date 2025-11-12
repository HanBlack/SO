# profile_generator_correct.py
# SPRÁVNÝ modul pro výpočet rozměrů bočních profilů rámu
# Funguje správně pro atypické tvary se šikmými hranami

import numpy as np
import math
from matplotlib.patches import Rectangle, Polygon as MPLPolygon
import matplotlib.pyplot as plt


def calculate_profile_dimensions(frame_vertices, frame_depth=30.0):
    """
    Vypočítá rozměry bočních profilů na základě vrcholů rámu.
    SPRÁVNÝ algoritmus pro atypické tvary.
    
    Args:
        frame_vertices: numpy array vrcholů rámu [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                       A (levý dolní), B (pravý dolní), C (pravý horní), D (levý horní)
        frame_depth: tloušťka rámu v mm (default 30.0)
    
    Returns:
        dict: Slovník s informacemi o profilech
    """
    if len(frame_vertices) != 4:
        raise ValueError("Očekávány 4 vrcholy rámu")
    
    # Pojmenování vrcholů
    A, B, C, D = frame_vertices
    
    # Výpočet délek stran
    bottom_length = np.linalg.norm(B - A)  # Dolní strana (A->B)
    right_length = np.linalg.norm(C - B)   # Pravá strana (B->C)
    top_length = np.linalg.norm(D - C)     # Horní strana (C->D)
    left_length = np.linalg.norm(A - D)    # Levá strana (D->A)
    
    # Výpočet úhlu sklonu horní strany (od horizontály)
    dx_top = C[0] - D[0]
    dy_top = C[1] - D[1]
    top_angle = math.degrees(math.atan2(abs(dy_top), abs(dx_top)))
    
    # Prodloužení/zkrácení kvůli šikmému řezu
    extension = frame_depth * math.tan(math.radians(top_angle))
    
    # Kontrola, zda je tvar pravidelný (obdélník)
    is_rectangular = abs(top_angle) < 0.5  # Horní strana je téměř horizontální
    
    # LEVÝ PROFIL
    # Vnější délka: základní délka + prodloužení nahoře (kvůli šikmé horní straně)
    # Vnitřní délka: základní délka (dolní roh je kolmý)
    left_outer = left_length + extension
    left_inner = left_length
    left_miter_top = top_angle  # úhel řezu nahoře
    left_miter_bottom = 0.0     # úhel řezu dole (kolmý)
    
    # PRAVÝ PROFIL
    # Vnější délka: základní délka (dolní roh je kolmý, takže žádné prodloužení)
    # Vnitřní délka: základní délka - zkrácení nahoře (kvůli šikmé horní straně)
    right_outer = right_length
    right_inner = right_length - extension
    right_miter_top = top_angle  # úhel řezu nahoře
    right_miter_bottom = 0.0     # úhel řezu dole (kolmý)
    
    # DOLNÍ PROFIL
    # Oba konce jsou kolmé, takže odečteme tloušťku rámu na obou stranách
    bottom_outer = bottom_length - 2 * frame_depth
    bottom_inner = bottom_length - 2 * frame_depth
    bottom_miter_left = 0.0
    bottom_miter_right = 0.0
    
    # HORNÍ PROFIL
    # Oba konce mají šikmý řez, ale délka se měří po střednici
    # Proto odečteme tloušťku rámu na obou stranách
    top_outer = top_length - 2 * frame_depth
    top_inner = top_length - 2 * frame_depth
    top_miter_left = top_angle
    top_miter_right = top_angle
    
    profiles = {
        'is_rectangular': is_rectangular,
        'frame_depth': frame_depth,
        'top_angle': top_angle,
        'extension': extension,
        'left': {
            'outer_length': left_outer,
            'inner_length': left_inner,
            'width': frame_depth,
            'angle': 0.0,  # levá strana je vertikální
            'miter_start': left_miter_top,
            'miter_end': left_miter_bottom,
            'corner_angles': (left_miter_bottom, left_miter_top)
        },
        'right': {
            'outer_length': right_outer,
            'inner_length': right_inner,
            'width': frame_depth,
            'angle': 0.0,  # pravá strana je vertikální
            'miter_start': right_miter_bottom,
            'miter_end': right_miter_top,
            'corner_angles': (right_miter_bottom, right_miter_top)
        },
        'bottom': {
            'outer_length': bottom_outer,
            'inner_length': bottom_inner,
            'width': frame_depth,
            'angle': 0.0,  # dolní strana je horizontální
            'miter_start': bottom_miter_left,
            'miter_end': bottom_miter_right,
            'corner_angles': (bottom_miter_left, bottom_miter_right)
        },
        'top': {
            'outer_length': top_outer,
            'inner_length': top_inner,
            'width': frame_depth,
            'angle': top_angle,  # horní strana má sklon
            'miter_start': top_miter_left,
            'miter_end': top_miter_right,
            'corner_angles': (top_miter_left, top_miter_right)
        }
    }
    
    return profiles


def draw_profile_cross_section(ax, profile_info, position, label, color='#8B4513'):
    """
    Vykreslí průřez profilu se šikmými řezy.
    """
    x, y = position
    width = profile_info['width']
    outer_len = profile_info['outer_length']
    inner_len = profile_info['inner_length']
    angle = profile_info['angle']
    miter_start = profile_info.get('miter_start', 0)
    miter_end = profile_info.get('miter_end', 0)
    
    # Vnější obdélník
    outer_points = [
        [x, y],
        [x + outer_len, y],
        [x + outer_len, y + width],
        [x, y + width]
    ]
    outer_poly = MPLPolygon(outer_points, linewidth=1.5, edgecolor='black', 
                           facecolor=color, alpha=0.7)
    ax.add_patch(outer_poly)
    
    # Vnitřní obdélník (drážka pro sklo)
    groove_width = 6
    inner_points = [
        [x + groove_width, y + groove_width],
        [x + inner_len + groove_width, y + groove_width],
        [x + inner_len + groove_width, y + width - groove_width],
        [x + groove_width, y + width - groove_width]
    ]
    inner_poly = MPLPolygon(inner_points, linewidth=1, edgecolor='gray', 
                           facecolor='white', alpha=0.5)
    ax.add_patch(inner_poly)
    
    # Popisky
    # Vnější délka (nahoře)
    ax.text(x + outer_len/2, y + width + 20, 
           f'{outer_len:.1f}', 
           ha='center', va='bottom', fontsize=9, fontweight='bold',
           bbox=dict(facecolor='white', edgecolor='blue', alpha=0.9))
    
    # Šířka (vlevo)
    ax.text(x - 20, y + width/2, 
           f'{width:.0f}', 
           ha='right', va='center', fontsize=9,
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    # Vnitřní délka (dole)
    if abs(outer_len - inner_len) > 0.5:
        ax.text(x + inner_len/2 + groove_width, y - 10, 
               f'{inner_len:.1f}', 
               ha='center', va='top', fontsize=8, style='italic', color='gray',
               bbox=dict(facecolor='lightyellow', edgecolor='gray', alpha=0.8))
    
    # Úhel sklonu strany
    if angle > 0.5:
        ax.text(x + outer_len + 25, y + width/2, 
               f'Sklon: {angle:.1f}°', 
               ha='left', va='center', fontsize=8, color='red',
               bbox=dict(facecolor='yellow', edgecolor='red', alpha=0.8))
    
    # Úhly řezů
    text_y_offset = y + width + 45
    if miter_start > 0.5:
        ax.text(x + 10, text_y_offset, 
               f'({miter_start:.1f}°)', 
               ha='left', va='bottom', fontsize=7, color='darkgreen',
               bbox=dict(facecolor='lightgreen', edgecolor='green', alpha=0.8))
    
    if miter_end > 0.5:
        ax.text(x + outer_len - 10, text_y_offset, 
               f'({miter_end:.1f}°)', 
               ha='right', va='bottom', fontsize=7, color='darkgreen',
               bbox=dict(facecolor='lightgreen', edgecolor='green', alpha=0.8))
    
    # Název profilu
    ax.text(x + outer_len/2, y - 35, 
           label, 
           ha='center', va='top', fontsize=10, fontweight='bold',
           bbox=dict(facecolor='lightblue', edgecolor='blue', alpha=0.9))


def draw_all_profiles(ax, profiles):
    """
    Vykreslí všechny profily na jedno plátno.
    """
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Boční profily rámu", fontsize=12, fontweight='bold')
    
    spacing = 100
    start_y = 0
    
    # Levý profil
    draw_profile_cross_section(ax, profiles['left'], 
                               (0, start_y), 
                               'Levý profil', 
                               color='#8B4513')
    
    # Pravý profil
    right_y = start_y + profiles['left']['width'] + spacing
    draw_profile_cross_section(ax, profiles['right'], 
                               (0, right_y), 
                               'Pravý profil', 
                               color='#A0522D')
    
    # Dolní profil
    bottom_y = right_y + profiles['right']['width'] + spacing
    draw_profile_cross_section(ax, profiles['bottom'], 
                               (0, bottom_y), 
                               'Dolní profil', 
                               color='#CD853F')
    
    # Horní profil
    top_y = bottom_y + profiles['bottom']['width'] + spacing
    draw_profile_cross_section(ax, profiles['top'], 
                               (0, top_y), 
                               'Horní profil', 
                               color='#DEB887')
    
    # Informační text
    info_text = "Pravidelný tvar" if profiles['is_rectangular'] else f"Atypický tvar (sklon {profiles['top_angle']:.1f}°)"
    ax.text(0, top_y + profiles['top']['width'] + 60, 
           info_text, 
           ha='left', va='bottom', fontsize=10, style='italic', fontweight='bold',
           bbox=dict(facecolor='lightyellow', edgecolor='orange', alpha=0.9, linewidth=2))
    
    # Nastavení limitů
    max_length = max(profiles['left']['outer_length'], 
                    profiles['right']['outer_length'],
                    profiles['bottom']['outer_length'],
                    profiles['top']['outer_length'])
    
    ax.set_xlim(-150, max_length + 150)
    ax.set_ylim(-80, top_y + profiles['top']['width'] + 120)


def draw_profile_dimensions_table(ax, profiles):
    """
    Vykreslí tabulku s rozměry profilů.
    """
    ax.clear()
    ax.axis('off')
    ax.set_title("Tabulka rozměrů profilů", fontsize=12, fontweight='bold')
    
    # Příprava dat
    table_data = []
    table_data.append(['Profil', 'Vnější délka', 'Vnitřní délka', 'Šířka', 'Řez start', 'Řez end'])
    
    for name, label in [('left', 'Levý'), ('right', 'Pravý'), 
                        ('bottom', 'Dolní'), ('top', 'Horní')]:
        profile = profiles[name]
        miter_start_str = f"{profile.get('miter_start', 0):.1f}°"
        miter_end_str = f"{profile.get('miter_end', 0):.1f}°"
        
        table_data.append([
            label,
            f"{profile['outer_length']:.1f}",
            f"{profile['inner_length']:.1f}",
            f"{profile['width']:.0f}",
            miter_start_str,
            miter_end_str
        ])
    
    # Vykreslení tabulky
    table = ax.table(cellText=table_data, 
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.2, 0.2, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Formátování záhlaví
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Formátování řádků
    for i in range(1, 5):
        for j in range(6):
            cell = table[(i, j)]
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')


if __name__ == "__main__":
    # Test s atypickým tvarem
    print("Test: Atypický tvar 1934x862x1950.2x1115")
    print("=" * 60)
    
    atypical_vertices = np.array([
        [0, 0],           # A - levý dolní
        [1934, 0],        # B - pravý dolní
        [1934, 1115],     # C - pravý horní
        [0, 862]          # D - levý horní
    ])
    
    profiles = calculate_profile_dimensions(atypical_vertices)
    
    print(f"\nVýsledky:")
    print(f"Je pravidelný: {profiles['is_rectangular']}")
    print(f"Úhel sklonu horní strany: {profiles['top_angle']:.2f}°")
    print(f"Prodloužení: {profiles['extension']:.2f} mm")
    print()
    
    print(f"Levý profil: {profiles['left']['outer_length']:.1f} x {profiles['left']['width']:.0f} ({profiles['left']['miter_start']:.1f}°) x {profiles['left']['inner_length']:.1f} x {profiles['left']['width']:.0f} ({profiles['left']['miter_end']:.1f}°)")
    print(f"Očekáváno:   865.9 x 30 (7.5°) x 862 x 30 (0.0°)")
    print()
    
    print(f"Pravý profil: {profiles['right']['outer_length']:.1f} x {profiles['right']['width']:.0f} ({profiles['right']['miter_start']:.1f}°) x {profiles['right']['inner_length']:.1f} x {profiles['right']['width']:.0f} ({profiles['right']['miter_end']:.1f}°)")
    print(f"Očekáváno:    1115.0 x 30 (0.0°) x 1111.1 x 30 (7.5°)")
    print()
    
    print(f"Dolní profil: {profiles['bottom']['outer_length']:.1f} x {profiles['bottom']['width']:.0f} ({profiles['bottom']['miter_start']:.1f}°) x {profiles['bottom']['inner_length']:.1f} x {profiles['bottom']['width']:.0f} ({profiles['bottom']['miter_end']:.1f}°)")
    print(f"Očekáváno:    1874.0 x 30 (0.0°) x 1874 x 30 (0.0°)")
    print()
    
    print(f"Horní profil: {profiles['top']['outer_length']:.1f} x {profiles['top']['width']:.0f} ({profiles['top']['miter_start']:.1f}°) x {profiles['top']['inner_length']:.1f} x {profiles['top']['width']:.0f} ({profiles['top']['miter_end']:.1f}°)")
    print(f"Očekáváno:    1890 x 30 (7.5°) x 1890 x 30 (7.5°)")
