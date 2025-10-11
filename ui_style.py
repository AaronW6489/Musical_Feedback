import tkinter as tk
from tkinter import ttk

def setup_styles():
    style = ttk.Style()
    # prefer clam for better widget control, fallback handled by Tk
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Color palette
    primary = "#3b82f6"   # blue
    accent = "#06b6d4"    # teal
    bg = "#f4f7fb"        # very light
    card = "#ffffff"
    text = "#1f2937"

    # General button style
    style.configure("TButton",
                    font=("Segoe UI", 11),
                    padding=8,
                    background=primary,
                    foreground="white")
    style.map("TButton",
              background=[("active", "#2b6ad6")],
              foreground=[("active", "white")])

    # Label style
    style.configure("TLabel",
                    font=("Segoe UI", 11),
                    background=bg,
                    foreground=text)

    # Title label
    style.configure("Title.TLabel",
                    font=("Segoe UI", 20, "bold"),
                    foreground="#0f172a",
                    background=bg)

    # Frame style - use card like background for sections
    style.configure("Card.TFrame", background=card, relief="flat")
    style.configure("App.TFrame", background=bg)

    # Section title
    style.configure("Section.TLabel", font=("Segoe UI", 14, "bold"), background=card, foreground=text)

    # Small muted label
    style.configure("Muted.TLabel", font=("Segoe UI", 10), background=card, foreground="#6b7280")

    # Bottom navigation style
    style.configure("BottomNav.TFrame", background=bg)
    style.configure("Nav.TButton", font=("Segoe UI", 10), padding=6)

def create_title(parent, text):
    """Create a styled title label"""
    lbl = ttk.Label(parent, text=text, style="Title.TLabel")
    lbl.pack(pady=(8, 12))
    return lbl

def create_section(parent, title):
    """Create a section frame with a title inside"""
    frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
    lbl = ttk.Label(frame, text=title, style="Section.TLabel")
    lbl.pack(anchor="w", pady=(0, 8))
    return frame
