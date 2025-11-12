# xml_exporter.py
# Export souřadnic rámu a okna do XML formátu

import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime


def prettify_xml(elem):
    """Vrátí pěkně formátovaný XML string"""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def export_to_xml(frame_points, window_points, filename=None):
    """
    Export souřadnic rámu a okna do XML souboru
    
    Args:
        frame_points: List bodů rámu [[x1,y1], [x2,y2], ...]
        window_points: List bodů okna [[x1,y1], [x2,y2], ...]
        filename: Název souboru (volitelné, default: frame_export_TIMESTAMP.xml)
    
    Returns:
        str: Název vytvořeného souboru
    """
    
    # Vytvoření root elementu
    root = ET.Element("FrameWindowData")
    root.set("version", "1.0")
    root.set("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Přidání informací o rámu
    frame_elem = ET.SubElement(root, "Frame")
    frame_elem.set("unit", "mm")
    frame_elem.set("point_count", str(len(frame_points) - 1))  # -1 protože poslední bod je duplicitní
    
    points_frame = ET.SubElement(frame_elem, "Points")
    for i, point in enumerate(frame_points[:-1]):  # Vynecháme poslední duplicitní bod
        point_elem = ET.SubElement(points_frame, "Point")
        point_elem.set("id", str(i))
        point_elem.set("x", f"{point[0]:.3f}")
        point_elem.set("y", f"{point[1]:.3f}")
    
    # Výpočet délek stran rámu
    edges_frame = ET.SubElement(frame_elem, "Edges")
    for i in range(len(frame_points) - 1):
        p1 = frame_points[i]
        p2 = frame_points[i + 1]
        length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        
        edge_elem = ET.SubElement(edges_frame, "Edge")
        edge_elem.set("id", str(i))
        edge_elem.set("from_point", str(i))
        edge_elem.set("to_point", str((i + 1) % (len(frame_points) - 1)))
        edge_elem.set("length", f"{length:.3f}")
    
    # Přidání informací o okně
    window_elem = ET.SubElement(root, "Window")
    window_elem.set("unit", "mm")
    window_elem.set("point_count", str(len(window_points) - 1))
    
    points_window = ET.SubElement(window_elem, "Points")
    for i, point in enumerate(window_points[:-1]):
        point_elem = ET.SubElement(points_window, "Point")
        point_elem.set("id", str(i))
        point_elem.set("x", f"{point[0]:.3f}")
        point_elem.set("y", f"{point[1]:.3f}")
    
    # Výpočet délek stran okna
    edges_window = ET.SubElement(window_elem, "Edges")
    for i in range(len(window_points) - 1):
        p1 = window_points[i]
        p2 = window_points[i + 1]
        length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        
        edge_elem = ET.SubElement(edges_window, "Edge")
        edge_elem.set("id", str(i))
        edge_elem.set("from_point", str(i))
        edge_elem.set("to_point", str((i + 1) % (len(window_points) - 1)))
        edge_elem.set("length", f"{length:.3f}")
    
    # Vytvoření názvu souboru
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_export_{timestamp}.xml"
    
    # Uložení do souboru
    xml_string = prettify_xml(root)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_string)
    
    print(f"\nXML export vytvořen: {filename}")
    print(f"  - Rám: {len(frame_points)-1} bodů")
    print(f"  - Okno: {len(window_points)-1} bodů")
    
    return filename


def read_from_xml(filename):
    """
    Načtení souřadnic z XML souboru
    
    Args:
        filename: Název XML souboru
    
    Returns:
        tuple: (frame_points, window_points)
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    
    # Načtení bodů rámu
    frame_points = []
    frame_elem = root.find("Frame")
    points_frame = frame_elem.find("Points")
    
    for point_elem in points_frame.findall("Point"):
        x = float(point_elem.get("x"))
        y = float(point_elem.get("y"))
        frame_points.append([x, y])
    
    # Uzavření polygonu
    frame_points.append(frame_points[0])
    
    # Načtení bodů okna
    window_points = []
    window_elem = root.find("Window")
    points_window = window_elem.find("Points")
    
    for point_elem in points_window.findall("Point"):
        x = float(point_elem.get("x"))
        y = float(point_elem.get("y"))
        window_points.append([x, y])
    
    # Uzavření polygonu
    window_points.append(window_points[0])
    
    print(f"\nXML načteno: {filename}")
    print(f"  - Rám: {len(frame_points)-1} bodů")
    print(f"  - Okno: {len(window_points)-1} bodů")
    
    return frame_points, window_points


if __name__ == "__main__":
    # Test
    test_frame = [[0, 0], [1560, 0], [1560, 221.3], [0, 0], [0, 0]]
    test_window = [[30, 30], [1530, 30], [1530, 191.3], [30, 0], [30, 30]]
    
    filename = export_to_xml(test_frame, test_window, "test_export.xml")
    
    # Test načtení
    loaded_frame, loaded_window = read_from_xml(filename)
    print("\nNačtené body rámu:", loaded_frame)
    print("Načtené body okna:", loaded_window)
