# main_final.py
# Finální hlavní aplikace s opraveným výpočtem profilů

import tkinter as tk
from frame_generator import SklenenyPanelApp
from frame_generator_with_profiles import SklenenyPanelAppWithProfiles


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Volba režimu - Generátor oken")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.menu_frame = tk.Frame(root)
        self.menu_frame.pack(padx=20, pady=20)

        tk.Label(self.menu_frame, text="Vyber režim:", font=("Arial", 14, "bold")).pack(pady=15)

        tk.Button(self.menu_frame, text="Jedno okno (základní)", bg="#4CAF50", fg="white",
                  font=("Arial", 11),
                  command=self.launch_single).pack(padx=10, pady=10, fill='x')
        
        tk.Button(self.menu_frame, text="Jedno okno s profily", bg="#2196F3", fg="white",
                  font=("Arial", 11),
                  command=self.launch_with_profiles).pack(padx=10, pady=10, fill='x')
        
        tk.Button(self.menu_frame, text="Sestava oken", bg="#FF9800", fg="white",
                  font=("Arial", 11),
                  command=self.launch_sestava).pack(padx=10, pady=10, fill='x')

        self.active_frame = None

    def clear_active(self):
        if self.active_frame is not None:
            self.active_frame.destroy()
            self.active_frame = None

    def show_menu(self):
        self.clear_active()
        self.menu_frame.pack(padx=20, pady=20)

    def launch_single(self):
        """Spustí základní verzi bez profilů"""
        self.menu_frame.pack_forget()
        self.active_frame = tk.Frame(self.root)
        self.active_frame.pack(fill="both", expand=True)
        SklenenyPanelApp(self.active_frame, back_callback=self.show_menu)

    def launch_with_profiles(self):
        """Spustí rozšířenou verzi s profily"""
        self.menu_frame.pack_forget()
        self.active_frame = tk.Frame(self.root)
        self.active_frame.pack(fill="both", expand=True)
        SklenenyPanelAppWithProfiles(self.active_frame, back_callback=self.show_menu)

    def launch_sestava(self):
        """Placeholder pro sestavu oken"""
        self.menu_frame.pack_forget()
        self.active_frame = tk.Frame(self.root)
        self.active_frame.pack(fill="both", expand=True)
        tk.Label(self.active_frame, text="Sestava oken – ve vývoji", 
                font=("Arial", 12)).pack(pady=50)
        tk.Button(self.active_frame, text="Zpět", command=self.show_menu).pack()

    def on_close(self):
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
