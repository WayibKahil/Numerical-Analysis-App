import customtkinter as ctk
from typing import Dict, Callable
import logging

class Sidebar:
    def __init__(self, parent, theme: Dict[str, str], home_cb: Callable, history_cb: Callable, 
                 settings_cb: Callable, about_cb: Callable):
        """
        Initialize the sidebar with navigation buttons.
        
        Args:
            parent: The parent widget
            theme: Dictionary containing theme colors
            home_cb: Callback for home button
            history_cb: Callback for history button
            settings_cb: Callback for settings button
            about_cb: Callback for about button
        """
        self.logger = logging.getLogger(__name__)
        self.theme = theme
        
        try:
            # Create sidebar with a slightly elevated appearance
            self.widget = ctk.CTkFrame(parent, width=200, fg_color=theme["fg"], corner_radius=10)
            self.widget.pack(side="left", fill="y", padx=(10, 10), pady=10)
            
            # Add a title label at the top
            title_label = ctk.CTkLabel(
                self.widget, 
                text="Numerical Analysis", 
                font=("Helvetica", 16, "bold"),
                text_color=theme["text"]
            )
            title_label.pack(pady=(20, 30))
            
            # Navigation buttons with consistent styling
            buttons = [
                ("Home", home_cb),
                ("History", history_cb),
                ("Settings", settings_cb),
                ("About", about_cb)
            ]
            
            for text, command in buttons:
                btn = ctk.CTkButton(
                    self.widget,
                    text=text,
                    command=command,
                    fg_color=theme["button"],
                    hover_color=theme["button_hover"],
                    text_color="#FFFFFF",  # White text for better contrast
                    font=("Helvetica", 14, "bold"),
                    width=160,
                    height=40,
                    corner_radius=8  # Rounded corners for modern look
                )
                btn.pack(pady=10, padx=20)
                
        except Exception as e:
            self.logger.error(f"Error initializing sidebar: {str(e)}")
            raise

    def update_theme(self, theme: Dict[str, str]) -> None:
        """
        Update the theme colors of the sidebar.
        
        Args:
            theme: Dictionary containing new theme colors
        """
        try:
            self.theme = theme
            self.widget.configure(fg_color=theme["fg"])
            
            # Update all widgets in the sidebar
            for widget in self.widget.winfo_children():
                if isinstance(widget, ctk.CTkButton):
                    widget.configure(
                        fg_color=theme["button"],
                        hover_color=theme["button_hover"],
                        text_color="#FFFFFF"  # White text for better contrast
                    )
                elif isinstance(widget, ctk.CTkLabel):
                    widget.configure(text_color=theme["text"])
        except Exception as e:
            self.logger.error(f"Error updating sidebar theme: {str(e)}")
            raise