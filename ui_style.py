import tkinter as tk
from tkinter import ttk

def setup_styles():
    style = ttk.Style()
    style.theme_use("clam")

    # General button style
    style.configure("TButton",
                    font=("Segoe UI", 12),
                    padding=8)

    # Hover effect for buttons
    style.map("TButton",
              background=[("active", "#2980b9")],
              foreground=[("active", "white")])

    # Label style
    style.configure("TLabel",
                    font=("Segoe UI", 12))

    # Title label
    style.configure("Title.TLabel",
                    font=("Segoe UI", 20, "bold"),
                    foreground="#2c3e50")

    # Radiobutton style
    style.configure("TRadiobutton",
                    font=("Segoe UI", 12))

    # Frame style
    style.configure("TFrame", background="#ecf0f1")

def create_title(parent, text):
    """Create a styled title label"""
    lbl = ttk.Label(parent, text=text, style="Title.TLabel")
    lbl.pack(pady=(10, 20))
    return lbl

def create_section(parent, title):
    """Create a section frame with a title inside"""
    frame = ttk.Frame(parent, padding=15, relief="ridge")
    lbl = ttk.Label(frame, text=title, font=("Segoe UI", 14, "bold"))
    lbl.pack(anchor="w", pady=(0, 10))
    return frame
