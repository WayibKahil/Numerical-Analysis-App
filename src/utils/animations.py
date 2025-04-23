import customtkinter as ctk

def fade_in(widget: ctk.CTkBaseClass, duration: int = 1000):
    steps = 20
    for i in range(steps + 1):
        alpha = i / steps
        widget.after(int(i * duration / steps), lambda a=alpha: widget.configure(text_color=f"{widget.cget('text_color')[:7]}{int(a*255):02x}"))

def slide_in(widget: ctk.CTkBaseClass, start_x: int, end_x: int, duration: int = 200):
    steps = 20
    for i in range(steps + 1):
        x = start_x + (end_x - start_x) * i / steps
        widget.after(int(i * duration / steps), lambda pos=x: widget.place(x=pos, y=0))