import customtkinter as ctk
from tkinter import ttk
from src.ui.widgets.input_form import InputForm
from src.ui.widgets.table import ResultTable
from src.ui.widgets.sidebar import Sidebar
from src.core.solver import Solver
from src.core.history import HistoryManager
from src.ui.theme import ThemeManager
from src.utils.export import export_to_pdf
import logging
from collections import OrderedDict
from src.core.methods.newton_raphson import ConvergenceStatus as NewtonConvergenceStatus
from src.core.methods.secant import ConvergenceStatus as SecantConvergenceStatus
import threading
import time

class NumericalApp:
    def __init__(self):
        """Initialize the application."""
        # Initialize logging before any other operations
        self.logger = logging.getLogger(__name__)
        self.version = "1.0.0"  # Add version attribute
        
        # Create the main window
        self.root = ctk.CTk()
        self.root.title("Numerical Analysis App")
        self.root.geometry("1000x700")
        
        # Initialize attributes to track after events
        self.after_ids = {}
        
        # Flag to indicate if the application is shutting down
        self.is_shutting_down = False
        
        # Add threading lock to prevent race conditions
        self.calculation_lock = threading.Lock()
        self.calculation_thread = None
        
        try:
            self.theme_manager = ThemeManager()
            self.history_manager = HistoryManager()
            self.solver = Solver()
            self.theme = self.theme_manager.apply_theme()
            self.configure_table_style()
            
            # Check DPI scaling once at initialization directly instead of using after
            self.check_dpi_scaling()
            self.setup_welcome_screen()
        except Exception as e:
            self.logger.error(f"Error initializing app: {str(e)}")
            raise

    def configure_table_style(self):
        """Configure the table style for better visibility."""
        try:
            style = ttk.Style()
            
            # Configure the main table style
            style.configure("Custom.Treeview",
                          background=self.theme["table_bg"],
                          foreground=self.theme["table_fg"],
                          fieldbackground=self.theme["table_bg"],
                          rowheight=25)
            
            # Configure the table heading style
            style.configure("Custom.Treeview.Heading",
                          background=self.theme["table_heading_bg"],
                          foreground=self.theme["table_heading_fg"],
                          font=("Helvetica", 10, "bold"))
            
            # Configure selection colors
            style.map("Custom.Treeview",
                     background=[("selected", self.theme["button"])],
                     foreground=[("selected", self.theme["text"])])
            
            # Configure row colors
            style.map("Custom.Treeview",
                     background=[("selected", self.theme["button"])],
                     foreground=[("selected", self.theme["text"])])
            
            self.logger.info("Table style configured successfully")
        except Exception as e:
            self.logger.error(f"Error configuring table style: {str(e)}")

    def setup_welcome_screen(self):
        """Initialize and display the welcome screen."""
        try:
            self.welcome_frame = ctk.CTkFrame(self.root, fg_color=self.theme["bg"])
            self.welcome_frame.pack(fill="both", expand=True)
            
            # Create a container for welcome content
            welcome_container = ctk.CTkFrame(self.welcome_frame, fg_color=self.theme["bg"])
            welcome_container.pack(expand=True)
            
            # Add application title
            title_label = ctk.CTkLabel(
                welcome_container, 
                text="Numerical Analysis App", 
                font=("Helvetica", 48, "bold"), 
                text_color=self.theme["accent"]
            )
            title_label.pack(pady=(0, 20))
            
            # Add version info
            version_label = ctk.CTkLabel(
                welcome_container, 
                text=f"Version {self.version}", 
                font=("Helvetica", 16), 
                text_color=self.theme["text"]
            )
            version_label.pack(pady=(0, 40))
            
            # Add loading indicator
            loading_label = ctk.CTkLabel(
                welcome_container, 
                text="Loading...", 
                font=("Helvetica", 14), 
                text_color=self.theme["text"]
            )
            loading_label.pack()
            
            # Schedule transition to main window and track the after ID using safe_after
            self.safe_after(2000, self.show_main_window, "welcome_transition")
            
        except Exception as e:
            self.logger.error(f"Error setting up welcome screen: {str(e)}")
            raise

    def show_main_window(self):
        """Transition from welcome screen to main window."""
        try:
            # Cancel any pending welcome screen after events
            if "welcome_transition" in self.after_ids:
                try:
                    self.root.after_cancel(self.after_ids["welcome_transition"])
                except Exception as e:
                    self.logger.debug(f"Error canceling welcome transition: {e}")
                # Remove from tracking dictionary
                del self.after_ids["welcome_transition"]
            
            # Destroy welcome frame if it exists
            if hasattr(self, "welcome_frame") and self.welcome_frame.winfo_exists():
                self.welcome_frame.destroy()
            
            # Create main frame
            self.main_frame = ctk.CTkFrame(self.root, fg_color=self.theme["bg"])
            self.main_frame.pack(fill="both", expand=True)

            # Create header
            self.header = ctk.CTkFrame(self.main_frame, height=60, fg_color=self.theme["fg"])
            self.header.pack(fill="x", pady=(0, 10))
            
            # Add title to header
            ctk.CTkLabel(
                self.header, 
                text="Numerical Analysis", 
                font=("Helvetica", 24, "bold"), 
                text_color=self.theme["accent"]
            ).pack(side="left", padx=20)
            
            # Create sidebar
            self.sidebar = Sidebar(
                self.main_frame, 
                self.theme, 
                self.show_home, 
                self.show_history, 
                self.show_settings, 
                self.show_about
            )
            
            # Create content frame
            self.content_frame = ctk.CTkFrame(self.main_frame, fg_color=self.theme["bg"])
            self.content_frame.pack(side="left", fill="both", expand=True)
            
            # Show home screen
            self.show_home()
            
        except Exception as e:
            self.logger.error(f"Error showing main window: {str(e)}")
            raise

    def change_theme(self, theme_name: str):
        self.theme = self.theme_manager.set_theme(theme_name)
        self.update_ui_theme()

    def update_ui_theme(self):
        """Update UI elements with the current theme."""
        try:
            # If app is shutting down, don't proceed with updates
            if getattr(self, 'is_shutting_down', False):
                return
                
            # Check if the window still exists before proceeding
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            # Cancel any existing update_ui_theme callbacks
            if "update_ui_theme" in self.after_ids:
                try:
                    self.root.after_cancel(self.after_ids["update_ui_theme"])
                    del self.after_ids["update_ui_theme"]
                except Exception as e:
                    self.logger.debug(f"Error canceling update_ui_theme callback: {e}")
            
            # Update the main window background
            self.root.configure(fg_color=self.theme.get("bg", "#F0F4F8"))
            
            # Update the sidebar
            if hasattr(self, "sidebar"):
                try:
                    self.sidebar.update_theme(self.theme)
                except Exception as sidebar_error:
                    self.logger.error(f"Error updating sidebar theme: {str(sidebar_error)}")
            
            # Update tables if they exist
            if hasattr(self, "result_table") and self.result_table is not None:
                try:
                    self.result_table.update_theme(self.theme)
                except Exception as table_error:
                    self.logger.error(f"Error updating result table theme: {str(table_error)}")
                    
            if hasattr(self, "history_table") and self.history_table is not None:
                try:
                    self.history_table.update_theme(self.theme)
                except Exception as table_error:
                    self.logger.error(f"Error updating history table theme: {str(table_error)}")
                    
            # Update forms if they exist
            if hasattr(self, "input_form") and self.input_form is not None:
                try:
                    self.input_form.update_theme(self.theme)
                except Exception as form_error:
                    self.logger.error(f"Error updating input form theme: {str(form_error)}")
                
            # Configure the Table Style
            self.configure_table_style()
                
        except Exception as e:
            self.logger.error(f"Error updating UI theme: {str(e)}")

    def show_home(self):
        """Display the home screen with input form and results table."""
        try:
            self.clear_content()
            
            # Create a canvas and scrollbar for scrolling
            canvas = ctk.CTkCanvas(self.content_frame, bg=self.theme.get("bg", "#F0F4F8"), highlightthickness=0)
            scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
            
            # Create the main frame that will be scrolled
            home_frame = ctk.CTkFrame(canvas, fg_color=self.theme.get("bg", "#F0F4F8"))
            
            # Configure the canvas
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Pack the scrollbar and canvas
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            
            # Create a window in the canvas for the frame
            canvas_window = canvas.create_window((0, 0), window=home_frame, anchor="nw", width=canvas.winfo_width())
            
            # Update the scroll region when the frame changes size
            def configure_scroll_region(event):
                canvas.configure(scrollregion=canvas.bbox("all"))
            
            home_frame.bind("<Configure>", configure_scroll_region)
            
            # Update the canvas window width when the canvas is resized
            def configure_canvas_window(event):
                canvas.itemconfig(canvas_window, width=event.width)
            
            canvas.bind("<Configure>", configure_canvas_window)
            
            # Add mousewheel scrolling
            def _on_mousewheel(event):
                """Handle mouse wheel scrolling with improved cross-platform support."""
                try:
                    # Quick validation of event and widgets
                    if not canvas.winfo_exists():
                        return
                        
                    # Use a safer method to get widget attributes
                    try:
                        # Get current scroll position to detect boundaries
                        current_pos = canvas.yview()
                        
                        # Calculate appropriate scroll amount based on platform detection
                        scroll_amount = 0
                        
                        # Windows (delta is typically multiples of 120)
                        if hasattr(event, "delta") and abs(event.delta) >= 120:
                            # Scale for smoother scrolling
                            scroll_factor = 2  # Adjust for smoother scrolling
                            scroll_amount = -1 * (event.delta // (120 / scroll_factor))
                        
                        # macOS (delta with smaller values)
                        elif hasattr(event, "delta") and abs(event.delta) < 20:
                            scroll_amount = -1 * event.delta
                        
                        # Linux/Unix (Button-4/Button-5)
                        elif hasattr(event, "num"):
                            if event.num == 4:
                                scroll_amount = -1
                            elif event.num == 5:
                                scroll_amount = 1
                        
                        # Apply the scroll if amount is non-zero and canvas still exists
                        if scroll_amount != 0 and canvas.winfo_exists():
                            canvas.yview_scroll(int(scroll_amount), "units")
                            
                            # Check if we hit an edge (view didn't change despite scroll attempt)
                            new_pos = canvas.yview()
                            if new_pos == current_pos and scroll_amount != 0:
                                # At edge of scrolling - allow propagation to parent
                                return
                            else:
                                # Scrolled successfully - prevent further propagation
                                return "break"
                    except Exception as e:
                        # Handle attribute access errors silently
                        self.logger.debug(f"Scroll attribute error (non-critical): {str(e)}")
                        return
                        
                except Exception as e:
                    # Log but don't disrupt user experience
                    self.logger.debug(f"Scroll handling error (non-critical): {str(e)}")
                
                # Allow event to propagate if not handled
                return
            
            # Directly bind to relevant widgets - more targeted approach
            canvas.bind("<MouseWheel>", _on_mousewheel)
            canvas.bind("<Button-4>", _on_mousewheel)
            canvas.bind("<Button-5>", _on_mousewheel)
            
            # Bind scrolling to the home frame as well
            home_frame.bind("<MouseWheel>", _on_mousewheel)
            home_frame.bind("<Button-4>", _on_mousewheel)
            home_frame.bind("<Button-5>", _on_mousewheel)
            
            # Create a label for the home screen
            home_label = ctk.CTkLabel(
                home_frame,
                text="Numerical Analysis Calculator",
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color=self.theme.get("text", "#1E293B")
            )
            home_label.pack(pady=(10, 10))
            
            # Create the input form with the correct parameters
            from src.ui.widgets.input_form import InputForm
            self.input_form = InputForm(
                home_frame, 
                self.theme, 
                list(self.solver.methods.keys()), 
                self.solve
            )
            self.input_form.frame.pack(fill="x", padx=10, pady=10)
            
            # Create layout containers
            main_content = ctk.CTkFrame(home_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            main_content.pack(fill="both", expand=True, padx=10, pady=5)
            
            # Create a container frame for the table that fills most of the screen
            table_container = ctk.CTkFrame(main_content, fg_color=self.theme.get("bg", "#F0F4F8"), height=450)
            table_container.pack(fill="both", expand=True, side="top", padx=2, pady=2)
            table_container.pack_propagate(False)  # Prevent container from resizing
            
            # Create the results table with fixed_position=True
            self.result_table = ResultTable(table_container, self.theme, height=450, fixed_position=True)
            self.result_table.table_frame.pack(fill="both", expand=True)
            
            # Create a frame for the result to give it a distinct appearance
            result_container = ctk.CTkFrame(
                main_content,
                fg_color=self.theme.get("primary_light", "#EFF6FF"),
                corner_radius=4,  # Reduced from 6
                border_width=1,
                border_color=self.theme.get("border", "#CBD5E1")
            )
            result_container.pack(fill="x", padx=5, pady=(3, 5))  # Reduced padding
            
            # Create a label for displaying the result
            self.result_label = ctk.CTkLabel(
                result_container,
                text="",
                font=ctk.CTkFont(size=14, weight="bold"),  # Reduced font size from 16 to 14
                text_color=self.theme.get("primary", "#3B82F6")
            )
            self.result_label.pack(pady=5)  # Reduced from 8 to 5
            
            # Add a frame for the plot with sufficient height but reduced padding
            self.plot_frame = ctk.CTkFrame(main_content, fg_color=self.theme.get("bg", "#F0F4F8"), height=350)
            self.plot_frame.pack(fill="both", expand=True, padx=2, pady=5)  # Reduced padding
            self.plot_frame.pack_propagate(False)  # Prevent plot frame from shrinking
            
            # Add a placeholder label for the plot
            self.plot_label = ctk.CTkLabel(
                self.plot_frame,
                text="Function plot will appear here after solving",
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B")
            )
            self.plot_label.pack(pady=20)
            
            # Handle plot frame mouse events - prevent scrolling when mouse is over the plot
            self.plot_frame.unbind("<MouseWheel>")
            self.plot_frame.unbind("<Button-4>")
            self.plot_frame.unbind("<Button-5>")
            
            # Add buttons container
            buttons_frame = ctk.CTkFrame(home_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            buttons_frame.pack(fill="x", padx=10, pady=5)
            
            # Add an export button
            export_button = ctk.CTkButton(
                buttons_frame,
                text="Export to PDF",
                command=self.export_solution,
                fg_color=self.theme.get("button", "#3B82F6"),
                hover_color=self.theme.get("button_hover", "#2563EB"),
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            export_button.pack(side="left", padx=10, pady=10, expand=True)
            
        except Exception as e:
            self.logger.error(f"Error showing home screen: {str(e)}")
            # Create a basic error display if the home frame creation fails
            error_frame = ctk.CTkFrame(self.content_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            error_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            error_label = ctk.CTkLabel(
                error_frame,
                text=f"Error loading home screen: {str(e)}",
                text_color="red",
                font=ctk.CTkFont(size=14)
            )
            error_label.pack(pady=10)

    def _round_value(self, value, decimal_places):
        """Round a value to the specified number of decimal places"""
        if not isinstance(value, (int, float)):
            return value
        if decimal_places is None:
            return value
        return round(value, decimal_places)
        
    def solve(self, **kwargs):
        """
        Solve a problem using the selected numerical method in a separate thread.
        
        Args:
            **kwargs: Keyword arguments containing method parameters
        """
        # Check if application is shutting down
        if getattr(self, 'is_shutting_down', False):
            return
        
        # Clear previous results
        if hasattr(self, 'result_label'):
            self.result_label.configure(text="")
        
        # Clear previous plot
        if hasattr(self, 'plot_frame'):
            # Cancel any existing plot-related after callbacks
            for after_id_name in list(self.after_ids.keys()):
                if "plot" in after_id_name:
                    try:
                        self.root.after_cancel(self.after_ids[after_id_name])
                        del self.after_ids[after_id_name]
                    except Exception as e:
                        self.logger.debug(f"Error canceling {after_id_name}: {e}")
            
            # Clear the placeholder
            for widget in self.plot_frame.winfo_children():
                try:
                    widget.destroy()
                except Exception:
                    pass
        
        # Create a progress frame
        if hasattr(self, 'plot_frame'):
            progress_frame = ctk.CTkFrame(self.plot_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            progress_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Add a label
            progress_label = ctk.CTkLabel(
                progress_frame,
                text=f"Calculating...",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=self.theme.get("text", "#1E293B")
            )
            progress_label.pack(pady=(20, 10))
            
            # Add progress indicator
            progress_bar = ctk.CTkProgressBar(
                progress_frame,
                width=300,
                mode="indeterminate",
                determinate_speed=0.5,
                indeterminate_speed=1,
                progress_color=self.theme.get("accent", "#3B82F6")
            )
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Add method info
            method_name = kwargs.get('method', 'Unknown')
            method_label = ctk.CTkLabel(
                progress_frame,
                text=f"Method: {method_name}",
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B")
            )
            method_label.pack(pady=5)
            
            # Add cancel button for long calculations
            cancel_button = ctk.CTkButton(
                progress_frame,
                text="Cancel",
                command=self._cancel_calculation,
                fg_color=self.theme.get("button", "#3B82F6"),
                hover_color=self.theme.get("button_hover", "#2563EB"),
                width=100
            )
            cancel_button.pack(pady=15)
            
            # Force update to show the progress indicators
            self.root.update_idletasks()
        
        # Store parameters for thread
        self.solve_params = kwargs.copy()
        
        # Set calculation active flag
        with self.calculation_lock:
            self.calculation_active = True
            
        # Extract parameters from kwargs for info display
        f_str = kwargs.get('f_str', '')
        method = kwargs.get('method', '')
        
        # Only use threading for matrix methods or when solving large systems
        use_threading = method in [
            "Gauss Elimination", "Gauss-Jordan", "LU Decomposition", "Cramer's Rule", 
            "Gauss Elimination (Partial Pivoting)", "LU Decomposition (Partial Pivoting)",
            "Gauss-Jordan (Partial Pivoting)"
        ]
        
        # Also use threading for high iteration counts in iterative methods
        max_iter = kwargs.get('max_iter', 0)
        if isinstance(max_iter, (int, float)) and max_iter > 100:
            use_threading = True
        
        # Define function to run in thread
        def calculation_thread():
            try:
                # Check if calculation is still active
                with self.calculation_lock:
                    if not self.calculation_active or self.is_shutting_down:
                        return
                
                    # Extract parameters from kwargs
                    f_str = self.solve_params.get('f_str', '')
                    method = self.solve_params.get('method', '')
                    params = self.solve_params.get('params', {})
                    eps = self.solve_params.get('eps', None)
                    eps_operator = self.solve_params.get('eps_operator', "<=")
                    max_iter = self.solve_params.get('max_iter', None)
                    stop_by_eps = self.solve_params.get('stop_by_eps', None)
                    decimal_places = self.solve_params.get('decimal_places', None)
                
                # Store a local copy of calculation_active
                is_active = True
                
                # Solve the problem (outside the lock to avoid deadlocks)
                result, table_data = self.solver.solve(method, f_str, params, eps, eps_operator, max_iter, stop_by_eps, decimal_places)
                
                # Check if calculation is still active before updating UI
                with self.calculation_lock:
                    is_active = self.calculation_active and not self.is_shutting_down
                
                # Post the results back to the main thread if calculation is still active
                if is_active and not self.is_shutting_down and hasattr(self, 'root') and self.root.winfo_exists():
                    self.safe_after(0, lambda: self._process_solve_result(result, table_data, method, f_str, decimal_places))
                    
            except Exception as e:
                self.logger.error(f"Error in calculation thread: {str(e)}")
                
                # Check if app is still running before showing error
                if not self.is_shutting_down and hasattr(self, 'root') and self.root.winfo_exists():
                    with self.calculation_lock:
                        is_active = self.calculation_active
                    
                    if is_active:
                        self.safe_after(0, lambda: self._show_calculation_error(str(e)))
                    
            finally:
                # Release the lock and update calculation status
                with self.calculation_lock:
                    self.calculation_active = False
        
        # Run calculation in thread if needed, otherwise directly
        if use_threading:
            # Create and start a new thread
            self.calculation_thread = threading.Thread(target=calculation_thread)
            self.calculation_thread.daemon = True  # Daemon thread will be killed when main thread exits
            self.calculation_thread.start()
        else:
            # For simpler calculations, run directly
            try:
                # Extract parameters from kwargs
                f_str = kwargs.get('f_str', '')
                method = kwargs.get('method', '')
                params = kwargs.get('params', {})
                eps = kwargs.get('eps', None)
                eps_operator = kwargs.get('eps_operator', "<=")
                max_iter = kwargs.get('max_iter', None)
                stop_by_eps = kwargs.get('stop_by_eps', None)
                decimal_places = kwargs.get('decimal_places', None)
                
                # Solve the problem
                result, table_data = self.solver.solve(method, f_str, params, eps, eps_operator, max_iter, stop_by_eps, decimal_places)
                
                # Process the result directly
                self._process_solve_result(result, table_data, method, f_str, decimal_places)
            except Exception as e:
                self.logger.error(f"Error solving problem: {str(e)}")
                self._show_calculation_error(str(e))
    
    def _cancel_calculation(self):
        """Cancel the current calculation."""
        # We can't actually interrupt a thread in Python safely,
        # but we can set a flag to indicate the calculation should stop
        self.logger.info("User canceled calculation")
        
        # Mark calculation as inactive so thread knows to stop
        with self.calculation_lock:
            self.calculation_active = False
        
        # Clean up UI safely
        if not self.is_shutting_down and hasattr(self, 'root') and self.root.winfo_exists() and hasattr(self, 'plot_frame'):
            # Clear the plot frame
            for widget in self.plot_frame.winfo_children():
                try:
                    widget.destroy()
                except Exception as e:
                    self.logger.debug(f"Error destroying widget: {e}")
                    pass
            
            # Show canceled message
            try:
                canceled_label = ctk.CTkLabel(
                    self.plot_frame,
                    text="Calculation canceled",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color=self.theme.get("text", "#1E293B")
                )
                canceled_label.pack(pady=20)
            except Exception as e:
                self.logger.debug(f"Error showing canceled message: {e}")
    
    def _show_calculation_error(self, error_message):
        """Show calculation error in the UI."""
        # Display error in table and result label
        if hasattr(self, 'result_table'):
            self.result_table.display(f"Error: {error_message}")
        
        if hasattr(self, 'result_label'):
            self.result_label.configure(text=f"Error: {error_message}")
        
        # Show error in plot area
        self._show_plot_error(f"Error: {error_message}")
    
    def _process_solve_result(self, result, table_data, method, f_str, decimal_places):
        """Process and display the solution result in the UI."""
        try:
            if hasattr(self, 'result_table'):
                # Handle case where table_data might be a pandas DataFrame (from iterations_table)
                if hasattr(table_data, 'to_dict'):
                    # Convert DataFrame to list of dictionaries
                    table_data = table_data.to_dict('records')
                
                # Check if result is an object with iterations_table attribute 
                if hasattr(result, 'iterations_table') and result.iterations_table is not None:
                    # Use the iterations_table from the result object for display
                    table_data = result.iterations_table.to_dict('records')
                    
                    # Add a final highlighted result row for Newton-Raphson and Secant methods
                    # to match the format of other methods like Bisection
                    if method == "Newton-Raphson":
                        # Get the converged root or last iteration value
                        if hasattr(result, 'root') and result.root is not None:
                            # Check if there's already a Result row - the method already adds one
                            has_result_row = any(
                                isinstance(row.get("Iteration"), str) and row.get("Iteration") == "Result" 
                                for row in table_data if isinstance(row, dict)
                            )
                            
                            # Only add our own result row if there isn't one already
                            if not has_result_row:
                                root_value = result.root
                                
                                # Create a highlighted result row
                                final_result_row = OrderedDict([
                                    ("Iteration", "Result"),
                                    ("Xi", ""),
                                    ("F(Xi)", ""),
                                    ("F'(Xi)", ""),
                                    ("Error%", ""),
                                    ("Xi+1", self._round_value(root_value, decimal_places) if decimal_places is not None else root_value),
                                    ("highlight", True)  # Add highlight flag for styling
                                ])
                                
                                # Add the result row to the table data
                                table_data.append(final_result_row)
                    elif method == "Secant":
                        # Get the converged root or last iteration value
                        if hasattr(result, 'root') and result.root is not None:
                            root_value = result.root
                            last_row = table_data[-1] if table_data else {}
                            
                            # Create a highlighted result row for Secant method - removing unwanted columns
                            final_result_row = OrderedDict([
                                ("Iteration", "Result"),
                                ("Xi-1", "---"),
                                ("F(Xi-1)", "---"),
                                ("Xi", "---"),
                                ("F(Xi)", "---"),
                                ("Xi+1", self._round_value(root_value, decimal_places) if decimal_places is not None else root_value),
                                ("Error%", "---"),
                                ("highlight", True)  # Add highlight flag for styling
                            ])
                            
                            # Add the result row to the table data
                            table_data.append(final_result_row)
                    elif method == "False Position":
                        # Check if there's already a Result row
                        has_result_row = any(
                            isinstance(row.get("Iteration"), str) and row.get("Iteration") == "Result" 
                            for row in table_data if isinstance(row, dict)
                        )
                        
                        # Only add our own result row if there isn't one already
                        if not has_result_row and result is not None:
                            if isinstance(result, tuple) and len(result) > 0:
                                root_value = result[0]
                            else:
                                root_value = result
                                
                            last_row = table_data[-1] if table_data else {}
                            
                            # Create a highlighted result row for False Position method
                            final_result_row = OrderedDict([
                                ("Iteration", "Result"),
                                ("Xl", ""),
                                ("f(Xl)", ""),
                                ("Xu", ""),
                                ("f(Xu)", ""),
                                ("Xr", self._round_value(root_value, decimal_places) if decimal_places is not None else root_value),
                                ("f(Xr)", last_row.get("f(Xr)", "")),
                                ("Error %", last_row.get("Error %", "")),
                                ("highlight", True)  # Add highlight flag for styling
                            ])
                            
                            # Add the result row to the table data
                            table_data.append(final_result_row)
                
                # Display the result in the table
                self.result_table.display(table_data)
                
                # For matrix methods, add a solution summary row to both result label and result table if needed
                if method in ["Gauss Elimination", "Gauss-Jordan", "LU Decomposition", "Cramer's Rule", 
                              "Gauss Elimination (Partial Pivoting)", "LU Decomposition (Partial Pivoting)",
                              "Gauss-Jordan (Partial Pivoting)"]:
                    if isinstance(result, list) or (isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], list)):
                        # Get the solution vector
                        solution_vector = result if isinstance(result, list) else result[0]
                        
                        # Format the solution for display
                        solution_str = ""
                        for i, val in enumerate(solution_vector):
                            if decimal_places is not None and isinstance(val, (int, float)):
                                val = round(val, decimal_places)
                            solution_str += f"x{i+1}={val}, "
                        solution_str = solution_str.rstrip(", ")
                        
                        # Add a summary row to the table with the solution vector
                        if isinstance(table_data, list) and len(table_data) > 0:
                            # Create a summary row with the solution (with the same structure as other rows)
                            summary_row = {"Step": "Solution Summary", "Matrix": solution_str, "Operation": "Complete solution vector"}
                            
                            # Add the row to the table data
                            table_data.append(summary_row)
                            
                            # Refresh the table display with the updated data
                            self.result_table.display(table_data)
                
                # Display the result
                if hasattr(self, 'result_label'):
                    # Get root value (handle different result types) and display it
                    root_message = "No solution found"
                    if result is not None:
                        if hasattr(result, 'root') and result.root is not None:
                            # Object with 'root' attribute (like NewtonRaphsonResult or SecantResult)
                            # Round the root value based on decimal_places
                            root_value = result.root
                            if isinstance(root_value, (int, float)) and decimal_places is not None:
                                root_value = round(root_value, decimal_places)
                            root_message = f"Root found: {root_value}"
                            
                            # Display table data directly from the result object if available
                            if hasattr(result, 'iterations_table') and result.iterations_table is not None:
                                table_data = result.iterations_table
                        elif isinstance(result, (int, float)):
                            # Direct numeric result - round it
                            if decimal_places is not None:
                                result = round(result, decimal_places)
                            root_message = f"Root found: {result}"
                        elif isinstance(result, list):
                            # Handle matrix solution (list of values)
                            solution_str = "Solution: "
                            for i, val in enumerate(result):
                                if decimal_places is not None and isinstance(val, (int, float)):
                                    val = round(val, decimal_places)
                                solution_str += f"x{i+1}={val}, "
                            root_message = solution_str.rstrip(", ")
                        elif isinstance(result, tuple) and len(result) > 0:
                            # Tuple with root as first element (like in Bisection, False Position, etc.)
                            if result[0] is not None and isinstance(result[0], (int, float)):
                                # Round the root value
                                root_value = result[0]
                                if decimal_places is not None:
                                    root_value = round(root_value, decimal_places)
                                root_message = f"Root found: {root_value}"
                            elif result[0] is not None and isinstance(result[0], list):
                                # First element is a list (solution vector)
                                solution_str = "Solution: "
                                for i, val in enumerate(result[0]):
                                    if decimal_places is not None and isinstance(val, (int, float)):
                                        val = round(val, decimal_places)
                                    solution_str += f"x{i+1}={val}, "
                                root_message = solution_str.rstrip(", ")
                    
                    self.result_label.configure(text=root_message)
            
            # Try to create a plot if we have a function
            if hasattr(self, 'plot_frame'):
                # Clear the plot frame first
                for widget in self.plot_frame.winfo_children():
                    widget.destroy()
                
                # For matrix methods, hide the plot frame completely
                if f_str == "System of Linear Equations" or method in ["Gauss Elimination", "Gauss-Jordan", "LU Decomposition", 
                                                                      "Cramer's Rule", "Gauss Elimination (Partial Pivoting)", 
                                                                      "LU Decomposition (Partial Pivoting)", "Gauss-Jordan (Partial Pivoting)"]:
                    # Hide the plot frame
                    if hasattr(self, 'plot_frame') and self.plot_frame.winfo_exists():
                        self.plot_frame.pack_forget()  # Remove plot frame from display
                    # Exit early - don't try to create a plot
                    return result
                else:
                    # For non-matrix methods, make sure plot frame is visible
                    if hasattr(self, 'plot_frame') and self.plot_frame.winfo_exists():
                        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Continue with normal plot creation code
                # Get root value (handle different result types)
                root_value = None
                if result:
                    if hasattr(result, 'root'):
                        root_value = result.root
                    elif isinstance(result, (int, float)):
                        root_value = result
                    elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], (int, float)):
                        root_value = result[0]
                
                # Get the function string if available and we have a root
                if f_str and f_str != "System of Linear Equations" and root_value is not None:
                    try:
                        # Log before attempting to create plot
                        self.logger.info(f"Attempting to create plot for function: {f_str} with root: {root_value}")
                        
                        import matplotlib.pyplot as plt
                        import numpy as np
                        import sympy as sp
                        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                        
                        # Create and solve the function
                        x = sp.Symbol('x')
                        f = sp.sympify(f_str)
                        f_lambda = sp.lambdify(x, f, 'numpy')
                        
                        # Create a plot around the root
                        root = float(root_value)
                        plot_range = max(4, abs(root) * 2)  # Ensure reasonable plot range
                        x_range = np.linspace(root - plot_range/2, root + plot_range/2, 1000)
                        
                        # Compute function values safely
                        y_values = []
                        x_filtered = []
                        
                        for x_val in x_range:
                            try:
                                y_val = f_lambda(x_val)
                                # Check if the result is a valid number
                                if np.isfinite(y_val) and not np.isnan(y_val):
                                    y_values.append(y_val)
                                    x_filtered.append(x_val)
                            except Exception:
                                pass
                        
                        if len(x_filtered) > 0 and len(y_values) > 0:
                            # Create the plot
                            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                            ax.plot(x_filtered, y_values, 'b-', label=f'f(x) = {f_str}')
                            
                            # Plot the root
                            try:
                                root_y = f_lambda(root)
                                if np.isfinite(root_y) and not np.isnan(root_y):
                                    ax.plot(root, root_y, 'ro', label=f'Root: {root:.6f}', markersize=8)
                            except Exception:
                                pass
                            
                            # Extract iteration points from the table data if available
                            iteration_x = []
                            iteration_y = []
                            
                            try:
                                # Handle data based on the method
                                if method == "Bisection Method":
                                    # Extract data from the table
                                    for row in table_data:
                                        if isinstance(row, dict) and "Iteration" in row and "Xr" in row:
                                            if isinstance(row["Iteration"], int):  # Only plot numerical iterations
                                                x_val = float(row["Xr"])
                                                try:
                                                    y_val = f_lambda(x_val)
                                                    if np.isfinite(y_val) and not np.isnan(y_val):
                                                        iteration_x.append(x_val)
                                                        iteration_y.append(y_val)
                                                except Exception:
                                                    pass
                                
                                elif method == "False Position Method":
                                    # Extract data from the table
                                    for row in table_data:
                                        if isinstance(row, dict) and "Iteration" in row and "Xr" in row:
                                            if isinstance(row["Iteration"], int):  # Only plot numerical iterations
                                                x_val = float(row["Xr"])
                                                try:
                                                    y_val = f_lambda(x_val)
                                                    if np.isfinite(y_val) and not np.isnan(y_val):
                                                        iteration_x.append(x_val)
                                                        iteration_y.append(y_val)
                                                except Exception:
                                                    pass
                                
                                elif method == "Secant Method":
                                    # Extract data from the table
                                    for row in table_data:
                                        if isinstance(row, dict) and "Iteration" in row and "Xi+1" in row:
                                            if isinstance(row["Iteration"], int):  # Only plot numerical iterations
                                                try:
                                                    if row["Xi+1"] != "---":  # Skip initial row where Xi+1 is not calculated
                                                        x_val = float(row["Xi+1"])
                                                        y_val = f_lambda(x_val)
                                                        if np.isfinite(y_val) and not np.isnan(y_val):
                                                            iteration_x.append(x_val)
                                                            iteration_y.append(y_val)
                                                except Exception:
                                                    pass
                                
                                elif method == "Newton Raphson Method":
                                    # Extract data from the table
                                    for row in table_data:
                                        if isinstance(row, dict) and "Iteration" in row and "Xi+1" in row:
                                            if isinstance(row["Iteration"], int):  # Only plot numerical iterations
                                                try:
                                                    if row["Xi+1"] != "---":  # Skip rows where Xi+1 is not calculated
                                                        x_val = float(row["Xi+1"])
                                                        y_val = f_lambda(x_val)
                                                        if np.isfinite(y_val) and not np.isnan(y_val):
                                                            iteration_x.append(x_val)
                                                            iteration_y.append(y_val)
                                                except Exception:
                                                    pass
                                
                                # Plot iteration points if available
                                if iteration_x and iteration_y:
                                    # Plot iteration points with connecting line to show convergence path
                                    ax.plot(iteration_x, iteration_y, 'g--o', label='Iteration Points', 
                                            alpha=0.7, markersize=6, markerfacecolor='white')
                                    
                                    # Annotate the first few points with iteration numbers
                                    max_annotations = min(6, len(iteration_x))
                                    for i in range(max_annotations):
                                        ax.annotate(f"{i}", 
                                                   (iteration_x[i], iteration_y[i]),
                                                   textcoords="offset points", 
                                                   xytext=(0,10), 
                                                   ha='center')
                            
                            except Exception as e:
                                self.logger.warning(f"Error plotting iteration points: {str(e)}")
                            
                            # Add a horizontal line at y=0
                            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                            
                            # Add labels and grid
                            ax.set_xlabel('x')
                            ax.set_ylabel('f(x)')
                            ax.set_title(f'Plot of f(x) = {f_str}')
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            
                            # Adjust plot limits to show convergence more clearly
                            if iteration_x and len(iteration_x) > 1:
                                # Get the range of iteration points
                                iter_min, iter_max = min(iteration_x), max(iteration_x)
                                # Extend the range by 20% on each side for better visibility
                                range_extension = (iter_max - iter_min) * 0.2
                                # Make sure we include the root
                                plot_min = min(iter_min - range_extension, root - range_extension)
                                plot_max = max(iter_max + range_extension, root + range_extension)
                                # Set the x-axis limits
                                ax.set_xlim(plot_min, plot_max)
                            
                            # Use FigureCanvasTkAgg
                            plot_frame = ctk.CTkFrame(self.plot_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
                            plot_frame.pack(fill="both", expand=True)
                            
                            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                            canvas_widget = canvas.get_tk_widget()
                            canvas_widget.pack(fill="both", expand=True)
                            
                            # Draw the canvas
                            canvas.draw()
                            
                            # Store reference to avoid garbage collection
                            self.current_plot = {
                                'figure': fig,
                                'canvas': canvas,
                                'frame': plot_frame
                            }
                            
                            # Log that the plot was successfully created
                            self.logger.info("Plot created successfully")
                        else:
                            self._show_plot_error("Could not generate valid function values for plotting")
                    except Exception as e:
                        self.logger.error(f"Error creating plot: {str(e)}")
                        self._show_plot_error(f"Error creating plot: {str(e)}")
                elif f_str == "System of Linear Equations":
                    # For matrix methods, hide the plot frame completely
                    if hasattr(self, 'plot_frame') and self.plot_frame.winfo_exists():
                        self.plot_frame.pack_forget()  # Remove plot frame from display
                    # Exit early - don't try to create a plot
                    return result
                else:
                    reason = "No root found" if f_str else "No valid function provided"
                    self._show_plot_error(f"Cannot create plot: {reason}")
            
            # Store the result for later export
            if result is not None and f_str is not None and method is not None:
                self.last_solution = (f_str, method, result, table_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing solution result: {str(e)}")
            self._show_calculation_error(str(e))
            return None
    
    def _show_plot_error(self, message="Error generating plot"):
        """Show an error message in the plot frame."""
        try:
            # Check if this is for a matrix method before showing any error
            if hasattr(self, 'last_solution') and self.last_solution:
                method = self.last_solution[1]
                if method in ["Gauss Elimination", "Gauss-Jordan", "LU Decomposition", 
                             "Cramer's Rule", "Gauss Elimination (Partial Pivoting)", 
                             "LU Decomposition (Partial Pivoting)", "Gauss-Jordan (Partial Pivoting)"]:
                    # For matrix methods, hide the plot frame completely
                    if hasattr(self, 'plot_frame') and self.plot_frame.winfo_exists():
                        self.plot_frame.pack_forget()  # Remove plot frame from display
                    return
            
            # For non-matrix methods, show error message
            if hasattr(self, 'plot_frame'):
                for widget in self.plot_frame.winfo_children():
                    try:
                        widget.destroy()
                    except Exception:
                        pass
                
                # Make sure plot frame is visible
                if hasattr(self, 'plot_frame') and self.plot_frame.winfo_exists():
                    self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Add an error label
                error_label = ctk.CTkLabel(
                    self.plot_frame,
                    text=message,
                    font=ctk.CTkFont(size=14),
                    text_color="red"
                )
                error_label.pack(pady=20)
        except Exception as e:
            self.logger.error(f"Error showing plot error: {str(e)}")

    def export_solution(self):
        if hasattr(self, "last_solution"):
            try:
                func, method, root, table_data = self.last_solution
                from datetime import datetime
                filename = f"solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                export_to_pdf(filename, func, method, root, table_data)
                self.result_label.configure(text=f"Exported to {filename}")
            except Exception as e:
                self.logger.error(f"Error exporting solution: {str(e)}")
                self.result_label.configure(text=f"Export error: {str(e)}")

    def clear_content(self):
        """Clear all widgets from the content frame."""
        try:
            if hasattr(self, "content_frame") and self.content_frame.winfo_exists():
                # First cancel any after callbacks that might reference these widgets
                for after_id_name in list(self.after_ids.keys()):
                    if "settings" in after_id_name or "history" in after_id_name:
                        try:
                            self.root.after_cancel(self.after_ids[after_id_name])
                            del self.after_ids[after_id_name]
                        except Exception as e:
                            self.logger.debug(f"Error canceling {after_id_name}: {e}")
                
                # Destroy all widgets in the content frame
                for widget in self.content_frame.winfo_children():
                    if widget.winfo_exists():
                        widget.destroy()
                        
                # Reset references to content-specific widgets
                if hasattr(self, "input_form"):
                    delattr(self, "input_form")
                if hasattr(self, "result_table"):
                    delattr(self, "result_table")
                if hasattr(self, "result_label"):
                    delattr(self, "result_label")
                if hasattr(self, "history_table"):
                    delattr(self, "history_table")
                if hasattr(self, "current_plot"):
                    delattr(self, "current_plot")
                
        except Exception as e:
            self.logger.error(f"Error clearing content: {str(e)}")
            # Continue execution even if there's an error

    def show_history(self):
        """Display the history screen."""
        try:
            self.clear_content()
            
            # Create a frame for the history table that takes up most of the space
            history_frame = ctk.CTkFrame(self.content_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            history_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create a label for the history
            history_label = ctk.CTkLabel(
                history_frame,
                text="Calculation History",
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color=self.theme.get("text", "#1E293B")
            )
            history_label.pack(pady=(0, 10))
            
            # Create a container frame for the table with fixed height
            table_container = ctk.CTkFrame(history_frame, fg_color=self.theme.get("bg", "#F0F4F8"), height=400)
            table_container.pack(fill="both", expand=True, padx=5, pady=5)
            table_container.pack_propagate(False)  # Prevent the frame from resizing based on its children
            
            # Create the history table with fixed height
            self.history_table = ResultTable(table_container, self.theme, height=400, fixed_position=True)
            self.history_table.table_frame.pack(fill="both", expand=True)
            
            # Load and display history data
            try:
                history_data = self.history_manager.load_history()
                self.history_table.display_history(history_data)
                
                # Display last solution if available
                if hasattr(self, "last_solution") and self.last_solution:
                    try:
                        func, method, root, table_data = self.last_solution
                        
                        # Create a frame for the last solution
                        last_solution_frame = ctk.CTkFrame(history_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
                        last_solution_frame.pack(fill="x", padx=5, pady=10)
                        
                        # Add a label for the last solution
                        last_solution_label = ctk.CTkLabel(
                            last_solution_frame,
                            text="Last Solution",
                            font=ctk.CTkFont(size=18, weight="bold"),
                            text_color=self.theme.get("text", "#1E293B")
                        )
                        last_solution_label.pack(pady=(0, 5))
                        
                        # Create a frame for the solution details
                        details_frame = ctk.CTkFrame(last_solution_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
                        details_frame.pack(fill="x", padx=10, pady=5)
                        
                        # Display function and method
                        ctk.CTkLabel(
                            details_frame,
                            text=f"Function: {func}",
                            font=ctk.CTkFont(size=14),
                            text_color=self.theme.get("text", "#1E293B")
                        ).pack(anchor="w", pady=2)
                        
                        ctk.CTkLabel(
                            details_frame,
                            text=f"Method: {method}",
                            font=ctk.CTkFont(size=14),
                            text_color=self.theme.get("text", "#1E293B")
                        ).pack(anchor="w", pady=2)
                        
                        # Display root if available
                        if root is not None:
                            ctk.CTkLabel(
                                details_frame,
                                text=f"Root: {root}",
                                font=ctk.CTkFont(size=14, weight="bold"),
                                text_color=self.theme.get("accent", "#0EA5E9")
                            ).pack(anchor="w", pady=2)
                        
                        # Add a button to view the full solution
                        def view_full_solution():
                            # Create a new window for the full solution
                            solution_window = ctk.CTkToplevel(self.root)
                            solution_window.title("Solution Details")
                            solution_window.geometry("900x650")
                            solution_window.grab_set()  # Make the window modal
                            
                            # Create a frame for the solution
                            solution_frame = ctk.CTkFrame(solution_window, fg_color=self.theme.get("bg", "#F0F4F8"))
                            solution_frame.pack(fill="both", expand=True, padx=8, pady=8)  # Reduced padding
                            
                            # Create header with method and function
                            header_frame = ctk.CTkFrame(solution_frame, fg_color=self.theme.get("fg", "#E2E8F0"))
                            header_frame.pack(fill="x", padx=2, pady=(0, 5))  # Reduced padding
                            
                            # Method title on left
                            method_label = ctk.CTkLabel(
                                header_frame,
                                text=f"Method: {method}",
                                font=ctk.CTkFont(size=14, weight="bold"),  # Reduced font size
                                text_color=self.theme.get("text", "#1E293B")
                            )
                            method_label.pack(side="left", padx=8, pady=5)  # Reduced padding
                            
                            # Function on right
                            func_label = ctk.CTkLabel(
                                header_frame,
                                text=f"Function: {func}",
                                font=ctk.CTkFont(size=14),  # Reduced font size
                                text_color=self.theme.get("text", "#1E293B")
                            )
                            func_label.pack(side="right", padx=8, pady=5)  # Reduced padding
                            
                            # Create a container for the solution table with fixed height and better styling
                            solution_table_container = ctk.CTkFrame(solution_frame, 
                                                                  fg_color=self.theme.get("bg", "#F0F4F8"), 
                                                                  height=470,
                                                                  border_width=1,
                                                                  border_color=self.theme.get("border", "#CBD5E1"))
                            solution_table_container.pack(fill="both", expand=True, padx=2, pady=2)  # Reduced padding
                            solution_table_container.pack_propagate(False)  # Prevent container from resizing
                            
                            # Create the table with fixed position
                            solution_table = ResultTable(solution_table_container, self.theme, height=470, fixed_position=True)
                            solution_table.table_frame.pack(fill="both", expand=True)
                            solution_table.display(table_data)
                            
                            # Create a result frame at the bottom with reduced padding
                            result_frame = ctk.CTkFrame(solution_frame, 
                                                      fg_color=self.theme.get("primary_light", "#EFF6FF"),
                                                      corner_radius=4)  # Reduced corner radius
                            result_frame.pack(fill="x", padx=2, pady=(2, 5))  # Reduced padding
                            
                            # Add the root result in the result frame
                            if root is not None:
                                root_label = ctk.CTkLabel(
                                    result_frame,
                                    text=f"Root found: {root}",
                                    font=ctk.CTkFont(size=14, weight="bold"),  # Reduced font size
                                    text_color=self.theme.get("primary", "#3B82F6")
                                )
                                root_label.pack(pady=5)  # Reduced padding
                            
                            # Add a divider with reduced padding
                            divider = ctk.CTkFrame(solution_frame, height=1, fg_color=self.theme.get("border", "#CBD5E1"))
                            divider.pack(fill="x", padx=2, pady=(0, 5))  # Reduced padding
                            
                            # Add button frame at the bottom with reduced padding
                            button_frame = ctk.CTkFrame(solution_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
                            button_frame.pack(fill="x", padx=2, pady=(0, 2))  # Reduced padding
                            
                            # Add export button with smaller size
                            export_button = ctk.CTkButton(
                                button_frame,
                                text="Export PDF",
                                command=lambda: self.export_solution(),
                                fg_color=self.theme.get("secondary", "#64748B"),
                                hover_color=self.theme.get("secondary_hover", "#475569"),
                                text_color="white",
                                font=ctk.CTkFont(size=12, weight="bold"),  # Reduced font size
                                width=100  # Reduced width
                            )
                            export_button.pack(side="left", padx=5)  # Reduced padding
                            
                            # Add a close button with smaller size
                            close_button = ctk.CTkButton(
                                button_frame,
                                text="Close",
                                command=solution_window.destroy,
                                fg_color=self.theme.get("primary", "#3B82F6"),
                                hover_color=self.theme.get("primary_hover", "#2563EB"),
                                text_color="white",
                                font=ctk.CTkFont(size=12, weight="bold"),  # Reduced font size
                                width=100  # Reduced width
                            )
                            close_button.pack(side="right", padx=5)  # Reduced padding
                        
                        view_button = ctk.CTkButton(
                            details_frame,
                            text="View Full Solution",
                            command=view_full_solution,
                            fg_color=self.theme.get("primary", "#3B82F6"),
                            hover_color=self.theme.get("primary_hover", "#2563EB"),
                            text_color="white",
                            font=ctk.CTkFont(size=14, weight="bold")
                        )
                        view_button.pack(pady=10)
                        
                    except Exception as solution_error:
                        self.logger.error(f"Error displaying last solution: {str(solution_error)}")
            except Exception as history_error:
                self.logger.error(f"Error loading history data: {str(history_error)}")
                self.history_table.display({"Error": f"Error loading history: {str(history_error)}"})
            
            # Create button container
            button_container = ctk.CTkFrame(history_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            button_container.pack(fill="x", pady=10)
            
            # Add a back button at the bottom
            back_button = ctk.CTkButton(
                button_container,
                text="Back to Home",
                command=self.show_home,
                fg_color=self.theme.get("primary", "#3B82F6"),
                hover_color=self.theme.get("primary_hover", "#2563EB"),
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            back_button.pack(side="right", padx=10, expand=True)
            
            # Add a clear history button
            def clear_history():
                try:
                    if self.history_manager.clear_history():
                        # Reload the history data
                        history_data = self.history_manager.load_history()
                        self.history_table.display_history(history_data)
                        
                        # Show success message
                        success_label = ctk.CTkLabel(
                            history_frame, 
                            text="History cleared successfully!", 
                            text_color="green", 
                            font=ctk.CTkFont(size=14)
                        )
                        success_label.pack(pady=10)
                        
                        # Remove the success message after 3 seconds and track the after ID
                        if "clear_history_success" in self.after_ids:
                            self.root.after_cancel(self.after_ids["clear_history_success"])
                        self.safe_after(3000, success_label.destroy, "clear_history_success")
                    else:
                        raise Exception("Failed to clear history")
                except Exception as e:
                    self.logger.error(f"Error clearing history: {str(e)}")
                    error_label = ctk.CTkLabel(
                        history_frame, 
                        text=f"Error clearing history: {str(e)}", 
                        text_color="red", 
                        font=ctk.CTkFont(size=14)
                    )
                    error_label.pack(pady=10)
                    
                    # Remove the error message after 5 seconds and track the after ID
                    if "clear_history_error" in self.after_ids:
                        self.root.after_cancel(self.after_ids["clear_history_error"])
                    self.safe_after(5000, error_label.destroy, "clear_history_error")
            
            clear_button = ctk.CTkButton(
                button_container,
                text="Clear History",
                command=clear_history,
                fg_color=self.theme.get("secondary", "#64748B"),
                hover_color=self.theme.get("secondary_hover", "#475569"),
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            clear_button.pack(side="left", padx=10, expand=True)
            
        except Exception as e:
            self.logger.error(f"Error showing history screen: {str(e)}")
            # Create a basic error display if the history frame creation fails
            error_frame = ctk.CTkFrame(self.content_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            error_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            error_label = ctk.CTkLabel(
                error_frame,
                text=f"Error loading history screen: {str(e)}",
                text_color="red",
                font=ctk.CTkFont(size=14)
            )
            error_label.pack(pady=10)
            
            # Add a back button to return to home
            back_button = ctk.CTkButton(
                error_frame,
                text="Back to Home",
                command=self.show_home,
                fg_color=self.theme.get("primary", "#3B82F6"),
                hover_color=self.theme.get("primary_hover", "#2563EB"),
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            back_button.pack(pady=10)

    def show_settings(self):
        """Display the settings screen with error handling."""
        try:
            self.clear_content()
            
            # Create a frame for the settings
            settings_frame = ctk.CTkFrame(self.content_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Add a title with icon
            title_frame = ctk.CTkFrame(settings_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            title_frame.pack(fill="x", pady=(20, 10))
            
            title_label = ctk.CTkLabel(
                title_frame, 
                text="Settings & Preferences",
                font=ctk.CTkFont(size=28, weight="bold"),
                text_color=self.theme.get("text", "#1E293B")
            )
            title_label.pack(side="left", padx=20)
            
            subtitle = ctk.CTkLabel(
                settings_frame, 
                text="Customize the application to suit your workflow",
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#64748B")
            )
            subtitle.pack(pady=(0, 20), anchor="w", padx=20)
            
            # Create a scrollable frame for settings
            scrollable_frame = ctk.CTkScrollableFrame(
                settings_frame, 
                fg_color=self.theme.get("bg", "#F0F4F8"),
                width=700,
                height=450
            )
            scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # --- Calculation Settings Card ---
            calc_card = ctk.CTkFrame(scrollable_frame, fg_color=self.theme.get("fg", "#DDE4E6"), corner_radius=10)
            calc_card.pack(fill="x", pady=10, padx=5, ipady=10)
            
            calc_header = ctk.CTkLabel(
                calc_card, 
                text="Calculation Settings",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.theme.get("accent", "#4C51BF")
            )
            calc_header.pack(anchor="w", padx=15, pady=(10, 15))
            
            # Default Decimal Places
            decimal_frame = ctk.CTkFrame(calc_card, fg_color="transparent")
            decimal_frame.pack(fill="x", pady=5, padx=15)
            
            decimal_label = ctk.CTkLabel(
                decimal_frame, 
                text="Default Decimal Places:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            decimal_label.pack(side="left")
            
            decimal_var = ctk.StringVar(value=str(self.solver.decimal_places))
            decimal_entry = ctk.CTkEntry(
                decimal_frame, 
                width=120, 
                textvariable=decimal_var,
                placeholder_text="e.g., 6"
            )
            decimal_entry.pack(side="left", padx=10)
            
            decimal_info = ctk.CTkLabel(
                decimal_frame,
                text="Affects display precision",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            decimal_info.pack(side="left", padx=10)
            
            # Maximum Iterations
            iter_frame = ctk.CTkFrame(calc_card, fg_color="transparent")
            iter_frame.pack(fill="x", pady=5, padx=15)
            
            iter_label = ctk.CTkLabel(
                iter_frame, 
                text="Maximum Iterations:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            iter_label.pack(side="left")
            
            iter_var = ctk.StringVar(value=str(self.solver.max_iter))
            iter_entry = ctk.CTkEntry(
                iter_frame, 
                width=120, 
                textvariable=iter_var,
                placeholder_text="e.g., 50"
            )
            iter_entry.pack(side="left", padx=10)
            
            iter_info = ctk.CTkLabel(
                iter_frame,
                text="Higher values may increase accuracy",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            iter_info.pack(side="left", padx=10)
            
            # Error Tolerance
            eps_frame = ctk.CTkFrame(calc_card, fg_color="transparent")
            eps_frame.pack(fill="x", pady=5, padx=15)
            
            eps_label = ctk.CTkLabel(
                eps_frame, 
                text="Error Tolerance:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            eps_label.pack(side="left")
            
            eps_var = ctk.StringVar(value=str(self.solver.eps))
            eps_entry = ctk.CTkEntry(
                eps_frame, 
                width=120, 
                textvariable=eps_var,
                placeholder_text="e.g., 0.0001"
            )
            eps_entry.pack(side="left", padx=10)
            
            eps_info = ctk.CTkLabel(
                eps_frame,
                text="Lower values increase precision",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            eps_info.pack(side="left", padx=10)
            
            # Maximum Epsilon Value
            max_eps_frame = ctk.CTkFrame(calc_card, fg_color="transparent")
            max_eps_frame.pack(fill="x", pady=5, padx=15)
            
            max_eps_label = ctk.CTkLabel(
                max_eps_frame, 
                text="Maximum Epsilon Value:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            max_eps_label.pack(side="left")
            
            max_eps_var = ctk.StringVar(value=str(getattr(self.solver, "max_eps", 1.0)))
            max_eps_entry = ctk.CTkEntry(
                max_eps_frame, 
                width=120, 
                textvariable=max_eps_var,
                placeholder_text="e.g., 1.0"
            )
            max_eps_entry.pack(side="left", padx=10)
            
            max_eps_info = ctk.CTkLabel(
                max_eps_frame,
                text="Upper bound for convergence check",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            max_eps_info.pack(side="left", padx=10)
            
            # Stop Condition
            stop_frame = ctk.CTkFrame(calc_card, fg_color="transparent")
            stop_frame.pack(fill="x", pady=5, padx=15)
            
            stop_label = ctk.CTkLabel(
                stop_frame, 
                text="Stop Condition:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            stop_label.pack(side="left")
            
            stop_var = ctk.StringVar(value="Error Tolerance" if self.solver.stop_by_eps else "Maximum Iterations")
            stop_option = ctk.CTkOptionMenu(
                stop_frame, 
                values=["Error Tolerance", "Maximum Iterations", "Both"], 
                variable=stop_var, 
                width=120,
                dropdown_fg_color=self.theme.get("bg", "#F0F4F8"),
                fg_color=self.theme.get("button", "#4C51BF"), 
                button_color=self.theme.get("button_hover", "#3C41AF"),
                button_hover_color=self.theme.get("accent", "#4C51BF")
            )
            stop_option.pack(side="left", padx=10)
            
            stop_info = ctk.CTkLabel(
                stop_frame,
                text="Determines when to stop iterations",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            stop_info.pack(side="left", padx=10)

            # --- User Interface Settings Card ---
            ui_card = ctk.CTkFrame(scrollable_frame, fg_color=self.theme.get("fg", "#DDE4E6"), corner_radius=10)
            ui_card.pack(fill="x", pady=10, padx=5, ipady=10)
            
            ui_header = ctk.CTkLabel(
                ui_card, 
                text="User Interface Settings",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.theme.get("accent", "#4C51BF")
            )
            ui_header.pack(anchor="w", padx=15, pady=(10, 15))
            
            # Theme Selection
            theme_frame = ctk.CTkFrame(ui_card, fg_color="transparent")
            theme_frame.pack(fill="x", pady=5, padx=15)
            
            theme_label = ctk.CTkLabel(
                theme_frame, 
                text="Application Theme:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            theme_label.pack(side="left")
            
            # Currently only light theme is available
            theme_var = ctk.StringVar(value="Light")
            theme_option = ctk.CTkOptionMenu(
                theme_frame, 
                values=["Light"], 
                variable=theme_var,
                width=120,
                dropdown_fg_color=self.theme.get("bg", "#F0F4F8"),
                fg_color=self.theme.get("button", "#4C51BF"), 
                button_color=self.theme.get("button_hover", "#3C41AF"),
                button_hover_color=self.theme.get("accent", "#4C51BF")
            )
            theme_option.pack(side="left", padx=10)
            
            theme_info = ctk.CTkLabel(
                theme_frame,
                text="More themes coming soon",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            theme_info.pack(side="left", padx=10)
            
            # Font Size
            font_frame = ctk.CTkFrame(ui_card, fg_color="transparent")
            font_frame.pack(fill="x", pady=5, padx=15)
            
            font_label = ctk.CTkLabel(
                font_frame, 
                text="Font Size:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            font_label.pack(side="left")
            
            font_size = getattr(self, "font_size", "Medium")
            font_var = ctk.StringVar(value=font_size)
            font_option = ctk.CTkOptionMenu(
                font_frame, 
                values=["Small", "Medium", "Large"], 
                variable=font_var,
                width=120,
                dropdown_fg_color=self.theme.get("bg", "#F0F4F8"),
                fg_color=self.theme.get("button", "#4C51BF"), 
                button_color=self.theme.get("button_hover", "#3C41AF"),
                button_hover_color=self.theme.get("accent", "#4C51BF")
            )
            font_option.pack(side="left", padx=10)
            
            font_info = ctk.CTkLabel(
                font_frame,
                text="Affects UI text size",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            font_info.pack(side="left", padx=10)
            
            # Auto-save Settings
            autosave_frame = ctk.CTkFrame(ui_card, fg_color="transparent")
            autosave_frame.pack(fill="x", pady=5, padx=15)
            
            autosave_label = ctk.CTkLabel(
                autosave_frame, 
                text="Auto-save Results:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            autosave_label.pack(side="left")
            
            autosave = getattr(self, "autosave", False)
            autosave_var = ctk.BooleanVar(value=autosave)
            autosave_switch = ctk.CTkSwitch(
                autosave_frame, 
                text="", 
                variable=autosave_var,
                width=60,
                fg_color=self.theme.get("bg", "#F0F4F8"),
                progress_color=self.theme.get("accent", "#4C51BF"),
                button_color=self.theme.get("fg", "#DDE4E6"),
                button_hover_color=self.theme.get("fg", "#DDE4E6"),
            )
            autosave_switch.pack(side="left", padx=10)
            
            autosave_info = ctk.CTkLabel(
                autosave_frame,
                text="Automatically save calculation results",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            autosave_info.pack(side="left", padx=10)
            
            # --- Advanced Settings Card ---
            adv_card = ctk.CTkFrame(scrollable_frame, fg_color=self.theme.get("fg", "#DDE4E6"), corner_radius=10)
            adv_card.pack(fill="x", pady=10, padx=5, ipady=10)
            
            adv_header = ctk.CTkLabel(
                adv_card, 
                text="Advanced Settings",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.theme.get("accent", "#4C51BF")
            )
            adv_header.pack(anchor="w", padx=15, pady=(10, 15))
            
            # Export Format
            export_frame = ctk.CTkFrame(adv_card, fg_color="transparent")
            export_frame.pack(fill="x", pady=5, padx=15)
            
            export_label = ctk.CTkLabel(
                export_frame, 
                text="Default Export Format:", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            export_label.pack(side="left")
            
            export_format = getattr(self, "export_format", "CSV")
            export_var = ctk.StringVar(value=export_format)
            export_option = ctk.CTkOptionMenu(
                export_frame, 
                values=["CSV", "Excel", "PDF", "JSON"], 
                variable=export_var,
                width=120,
                dropdown_fg_color=self.theme.get("bg", "#F0F4F8"),
                fg_color=self.theme.get("button", "#4C51BF"), 
                button_color=self.theme.get("button_hover", "#3C41AF"),
                button_hover_color=self.theme.get("accent", "#4C51BF")
            )
            export_option.pack(side="left", padx=10)
            
            export_info = ctk.CTkLabel(
                export_frame,
                text="For exporting calculation results",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            export_info.pack(side="left", padx=10)
            
            # Calculation Timeout
            timeout_frame = ctk.CTkFrame(adv_card, fg_color="transparent")
            timeout_frame.pack(fill="x", pady=5, padx=15)
            
            timeout_label = ctk.CTkLabel(
                timeout_frame, 
                text="Calculation Timeout (sec):", 
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B"),
                width=200,
                anchor="w"
            )
            timeout_label.pack(side="left")
            
            timeout = getattr(self, "timeout", 30)
            timeout_var = ctk.StringVar(value=str(timeout))
            timeout_entry = ctk.CTkEntry(
                timeout_frame, 
                width=120, 
                textvariable=timeout_var,
                placeholder_text="e.g., 30"
            )
            timeout_entry.pack(side="left", padx=10)
            
            timeout_info = ctk.CTkLabel(
                timeout_frame,
                text="Maximum time before cancellation",
                font=ctk.CTkFont(size=12),
                text_color=self.theme.get("text", "#64748B")
            )
            timeout_info.pack(side="left", padx=10)
            
            # Create button container
            button_container = ctk.CTkFrame(settings_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            button_container.pack(fill="x", pady=20)
            
            # Save Button
            def save_settings():
                try:
                    # Validate and save decimal places
                    try:
                        decimal_places = int(decimal_var.get())
                        if decimal_places < 0:
                            raise ValueError("Decimal places must be non-negative")
                        self.solver.decimal_places = decimal_places
                    except ValueError as e:
                        raise ValueError(f"Invalid decimal places: {str(e)}")
                    
                    # Validate and save maximum iterations
                    try:
                        max_iter = int(iter_var.get())
                        if max_iter <= 0:
                            raise ValueError("Maximum iterations must be positive")
                        self.solver.max_iter = max_iter
                    except ValueError as e:
                        raise ValueError(f"Invalid maximum iterations: {str(e)}")
                    
                    # Validate and save error tolerance
                    try:
                        eps = float(eps_var.get())
                        if eps <= 0:
                            raise ValueError("Error tolerance must be positive")
                        self.solver.eps = eps
                    except ValueError as e:
                        raise ValueError(f"Invalid error tolerance: {str(e)}")
                    
                    # Validate and save maximum epsilon
                    try:
                        max_eps = float(max_eps_var.get())
                        if max_eps <= 0:
                            raise ValueError("Maximum epsilon must be positive")
                        self.solver.max_eps = max_eps
                    except ValueError as e:
                        raise ValueError(f"Invalid maximum epsilon: {str(e)}")
                    
                    # Save stop condition
                    self.solver.stop_by_eps = stop_var.get() in ["Error Tolerance", "Both"]
                    
                    # Save UI settings
                    self.font_size = font_var.get()
                    self.autosave = autosave_var.get()
                    self.export_format = export_var.get()
                    
                    # Save timeout setting
                    try:
                        timeout = int(timeout_var.get())
                        if timeout <= 0:
                            raise ValueError("Timeout must be positive")
                        self.timeout = timeout
                    except ValueError as e:
                        raise ValueError(f"Invalid timeout: {str(e)}")
                    
                    # Show success message with animation
                    success_frame = ctk.CTkFrame(
                        settings_frame, 
                        fg_color=self.theme.get("accent", "#4C51BF"),
                        corner_radius=10
                    )
                    success_frame.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.5, height=40)
                    
                    success_label = ctk.CTkLabel(
                        success_frame, 
                        text="✓ Settings saved successfully!", 
                        text_color="white", 
                        font=ctk.CTkFont(size=14, weight="bold")
                    )
                    success_label.pack(pady=10, padx=10, fill="both", expand=True)
                    
                    # Remove the success message after 3 seconds and track the after ID
                    if "save_settings_success" in self.after_ids:
                        self.root.after_cancel(self.after_ids["save_settings_success"])
                    
                    def hide_success():
                        success_frame.place_forget()
                        
                    self.safe_after(3000, hide_success, "save_settings_success")
                    
                except Exception as e:
                    self.logger.error(f"Error saving settings: {str(e)}")
                    
                    # Show error message with animation
                    error_frame = ctk.CTkFrame(
                        settings_frame, 
                        fg_color="#E53E3E",  # Error red color
                        corner_radius=10
                    )
                    error_frame.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.7, height=40)
                    
                    error_label = ctk.CTkLabel(
                        error_frame, 
                        text=f"✗ Error: {str(e)}", 
                        text_color="white", 
                        font=ctk.CTkFont(size=14, weight="bold")
                    )
                    error_label.pack(pady=10, padx=10, fill="both", expand=True)
                    
                    # Remove the error message after 5 seconds and track the after ID
                    if "save_settings_error" in self.after_ids:
                        self.root.after_cancel(self.after_ids["save_settings_error"])
                    
                    def hide_error():
                        error_frame.place_forget()
                        
                    self.safe_after(5000, hide_error, "save_settings_error")
            
            save_button = ctk.CTkButton(
                button_container, 
                text="Save Changes",
                command=save_settings,
                fg_color=self.theme.get("primary", "#4C51BF"), 
                hover_color=self.theme.get("primary_hover", "#3C41AF"),
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold"),
                height=38,
                corner_radius=8,
                width=170
            )
            save_button.pack(side="left", padx=10, expand=True)
            
            # Reset Button
            def reset_settings():
                try:
                    # Reset calculation settings
                    self.solver.decimal_places = 6
                    self.solver.max_iter = 50
                    self.solver.eps = 0.0001
                    self.solver.max_eps = 1.0
                    self.solver.stop_by_eps = True
                    
                    # Reset UI variables
                    decimal_var.set("6")
                    iter_var.set("50")
                    eps_var.set("0.0001")
                    max_eps_var.set("1.0")
                    stop_var.set("Error Tolerance")
                    font_var.set("Medium")
                    autosave_var.set(False)
                    export_var.set("CSV")
                    timeout_var.set("30")
                    
                    # Reset UI settings
                    self.font_size = "Medium"
                    self.autosave = False
                    self.export_format = "CSV"
                    self.timeout = 30
                    
                    # Show success message with animation
                    reset_frame = ctk.CTkFrame(
                        settings_frame, 
                        fg_color=self.theme.get("accent", "#4C51BF"),
                        corner_radius=10
                    )
                    reset_frame.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.5, height=40)
                    
                    reset_label = ctk.CTkLabel(
                        reset_frame, 
                        text="✓ Settings reset to defaults!", 
                        text_color="white", 
                        font=ctk.CTkFont(size=14, weight="bold")
                    )
                    reset_label.pack(pady=10, padx=10, fill="both", expand=True)
                    
                    # Remove the reset message after 3 seconds and track the after ID
                    if "reset_settings_success" in self.after_ids:
                        self.root.after_cancel(self.after_ids["reset_settings_success"])
                    
                    def hide_reset():
                        reset_frame.place_forget()
                        
                    self.safe_after(3000, hide_reset, "reset_settings_success")
                    
                except Exception as e:
                    self.logger.error(f"Error resetting settings: {str(e)}")
                    
                    # Show error message with animation
                    error_frame = ctk.CTkFrame(
                        settings_frame, 
                        fg_color="#E53E3E",  # Error red color
                        corner_radius=10
                    )
                    error_frame.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.7, height=40)
                    
                    error_label = ctk.CTkLabel(
                        error_frame, 
                        text=f"✗ Error: {str(e)}", 
                        text_color="white", 
                        font=ctk.CTkFont(size=14, weight="bold")
                    )
                    error_label.pack(pady=10, padx=10, fill="both", expand=True)
                    
                    # Remove the error message after 5 seconds and track the after ID
                    if "reset_settings_error" in self.after_ids:
                        self.root.after_cancel(self.after_ids["reset_settings_error"])
                    
                    def hide_error():
                        error_frame.place_forget()
                        
                    self.safe_after(5000, hide_error, "reset_settings_error")
            
            reset_button = ctk.CTkButton(
                button_container, 
                text="Reset to Defaults",
                command=reset_settings,
                fg_color="transparent", 
                hover_color=self.theme.get("fg", "#DDE4E6"),
                text_color=self.theme.get("text", "#1E293B"),
                font=ctk.CTkFont(size=14),
                border_width=1,
                border_color=self.theme.get("text", "#1E293B"),
                height=38,
                corner_radius=8,
                width=170
            )
            reset_button.pack(side="right", padx=10, expand=True)
            
        except Exception as e:
            self.logger.error(f"Error showing settings: {str(e)}")
            # Create a basic error display if the settings frame creation fails
            error_frame = ctk.CTkFrame(self.content_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            error_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            error_label = ctk.CTkLabel(
                error_frame,
                text=f"Error loading settings: {str(e)}",
                text_color="red",
                font=ctk.CTkFont(size=14)
            )
            error_label.pack(pady=10)
            
            # Add a back button to return to home
            back_button = ctk.CTkButton(
                error_frame,
                text="Back to Home",
                command=self.show_home,
                fg_color=self.theme.get("primary", "#4C51BF"),
                hover_color=self.theme.get("primary_hover", "#3C41AF"),
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            back_button.pack(pady=10)

    def show_about(self):
        """Display the about screen with application information."""
        try:
            self.clear_content()
            
            # Create a frame for the about screen
            about_frame = ctk.CTkFrame(self.content_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            about_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create top section with logo and app info
            top_section = ctk.CTkFrame(about_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            top_section.pack(fill="x", padx=20, pady=(20, 10))
            
            # Application Title with decorative element
            title_frame = ctk.CTkFrame(top_section, fg_color=self.theme.get("accent", "#4C51BF"), corner_radius=10)
            title_frame.pack(side="left", padx=(0, 20))
            
            # Add a mathematical symbol as decorative element
            math_symbol = ctk.CTkLabel(
                title_frame,
                text="∫ ∑ ∂",
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color="white"
            )
            math_symbol.pack(pady=20, padx=20)
            
            # App info container
            app_info = ctk.CTkFrame(top_section, fg_color=self.theme.get("bg", "#F0F4F8"))
            app_info.pack(side="left", fill="both", expand=True)
            
            title_label = ctk.CTkLabel(
                app_info,
                text="Numerical Analysis Application",
                font=ctk.CTkFont(size=28, weight="bold"),
                text_color=self.theme.get("text", "#1E293B"),
                anchor="w"
            )
            title_label.pack(fill="x", pady=(0, 5))
            
            # Version - safely handle if version is not defined
            version = getattr(self, "version", "1.0.0")
            version_label = ctk.CTkLabel(
                app_info,
                text=f"Version: {version}",
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#64748B"),
                anchor="w"
            )
            version_label.pack(fill="x", pady=(0, 5))
            
            # Release date
            release_date = ctk.CTkLabel(
                app_info,
                text=f"Release Date: May 2024",
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#64748B"),
                anchor="w"
            )
            release_date.pack(fill="x", pady=(0, 10))
            
            # Divider
            divider = ctk.CTkFrame(about_frame, height=2, fg_color=self.theme.get("fg", "#DDE4E6"))
            divider.pack(fill="x", padx=20, pady=10)
            
            # Create a scrollable frame for content
            scrollable_frame = ctk.CTkScrollableFrame(
                about_frame, 
                fg_color=self.theme.get("bg", "#F0F4F8"),
                scrollbar_fg_color=self.theme.get("bg", "#F0F4F8"),
                scrollbar_button_color=self.theme.get("accent", "#4C51BF")
            )
            scrollable_frame.pack(fill="both", expand=True, padx=20, pady=5)
            
            # Features Section
            features_frame = ctk.CTkFrame(scrollable_frame, fg_color=self.theme.get("fg", "#DDE4E6"), corner_radius=10)
            features_frame.pack(fill="x", pady=10, ipady=10)
            
            features_title = ctk.CTkLabel(
                features_frame,
                text="Features",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.theme.get("accent", "#4C51BF")
            )
            features_title.pack(anchor="w", padx=15, pady=(10, 15))
            
            # Feature list
            features = [
                "📊 Multiple numerical methods for root finding and equation solving",
                "📈 Interactive function plotting with error analysis",
                "📋 Detailed step-by-step solution visualization",
                "📁 Export results to various formats (CSV, Excel, PDF)",
                "🔧 Customizable calculation settings and precision control",
                "📱 Responsive interface with modern design",
                "💾 Solution history for reviewing past calculations"
            ]
            
            for feature in features:
                feature_label = ctk.CTkLabel(
                    features_frame,
                    text=feature,
                    font=ctk.CTkFont(size=14),
                    text_color=self.theme.get("text", "#1E293B"),
                    anchor="w"
                )
                feature_label.pack(fill="x", padx=15, pady=3)
            
            # Implemented Methods Section
            methods_frame = ctk.CTkFrame(scrollable_frame, fg_color=self.theme.get("fg", "#DDE4E6"), corner_radius=10)
            methods_frame.pack(fill="x", pady=10, ipady=10)
            
            methods_title = ctk.CTkLabel(
                methods_frame,
                text="Implemented Methods",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.theme.get("accent", "#4C51BF")
            )
            methods_title.pack(anchor="w", padx=15, pady=(10, 15))
            
            # Methods grid - using two columns
            methods_container = ctk.CTkFrame(methods_frame, fg_color="transparent")
            methods_container.pack(fill="x", padx=15, pady=5)
            
            # Column 1
            methods_col1 = ctk.CTkFrame(methods_container, fg_color="transparent")
            methods_col1.pack(side="left", fill="both", expand=True)
            
            methods1 = [
                "• Bisection Method",
                "• Newton-Raphson Method",
                "• Secant Method",
                "• Fixed-Point Iteration"
            ]
            
            for method in methods1:
                method_label = ctk.CTkLabel(
                    methods_col1,
                    text=method,
                    font=ctk.CTkFont(size=14),
                    text_color=self.theme.get("text", "#1E293B"),
                    anchor="w"
                )
                method_label.pack(fill="x", pady=3)
            
            # Column 2
            methods_col2 = ctk.CTkFrame(methods_container, fg_color="transparent")
            methods_col2.pack(side="left", fill="both", expand=True)
            
            methods2 = [
                "• Gauss Elimination",
                "• Gauss-Jordan Method",
                "• LU Decomposition",
                "• Cramer's Rule"
            ]
            
            for method in methods2:
                method_label = ctk.CTkLabel(
                    methods_col2,
                    text=method,
                    font=ctk.CTkFont(size=14),
                    text_color=self.theme.get("text", "#1E293B"),
                    anchor="w"
                )
                method_label.pack(fill="x", pady=3)
            
            # Technology Section
            tech_frame = ctk.CTkFrame(scrollable_frame, fg_color=self.theme.get("fg", "#DDE4E6"), corner_radius=10)
            tech_frame.pack(fill="x", pady=10, ipady=10)
            
            tech_title = ctk.CTkLabel(
                tech_frame,
                text="Technology Stack",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.theme.get("accent", "#4C51BF")
            )
            tech_title.pack(anchor="w", padx=15, pady=(10, 15))
            
            tech_items = [
                "• Python - Core programming language",
                "• CustomTkinter - Modern UI framework",
                "• NumPy - Numerical computations and array operations",
                "• Matplotlib - Data visualization and plotting",
                "• SymPy - Symbolic mathematics for equation parsing"
            ]
            
            for item in tech_items:
                tech_label = ctk.CTkLabel(
                    tech_frame,
                    text=item,
                    font=ctk.CTkFont(size=14),
                    text_color=self.theme.get("text", "#1E293B"),
                    anchor="w"
                )
                tech_label.pack(fill="x", padx=15, pady=3)
            
            # Credits section 
            credits_frame = ctk.CTkFrame(scrollable_frame, fg_color=self.theme.get("fg", "#DDE4E6"), corner_radius=10)
            credits_frame.pack(fill="x", pady=10, ipady=10)
            
            credits_title = ctk.CTkLabel(
                credits_frame,
                text="Credits & Contributors",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.theme.get("accent", "#4C51BF")
            )
            credits_title.pack(anchor="w", padx=15, pady=(10, 15))
            
            credits_info = ctk.CTkLabel(
                credits_frame,
                text="Developed by Hosam Dyab and Hazem Mohamed\nSpecial thanks to all numerical analysis and scientific computing communities",
                font=ctk.CTkFont(size=14),
                text_color=self.theme.get("text", "#1E293B")
            )
            credits_info.pack(padx=15, pady=5)
            
            # Create button container
            button_container = ctk.CTkFrame(about_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            button_container.pack(fill="x", pady=15)
            
            # Back Button
            back_button = ctk.CTkButton(
                button_container,
                text="Back to Home",
                command=self.show_home,
                fg_color=self.theme.get("primary", "#4C51BF"),
                hover_color=self.theme.get("primary_hover", "#3C41AF"),
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold"),
                height=38,
                corner_radius=8
            )
            back_button.pack(padx=20, pady=10)
            
        except Exception as e:
            self.logger.error(f"Error showing about screen: {str(e)}")
            # Create a basic error display if the about frame creation fails
            error_frame = ctk.CTkFrame(self.content_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
            error_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            error_label = ctk.CTkLabel(
                error_frame,
                text=f"Error loading about screen: {str(e)}",
                text_color="red",
                font=ctk.CTkFont(size=14)
            )
            error_label.pack(pady=10)
            
            # Add a back button to return to home
            back_button = ctk.CTkButton(
                error_frame,
                text="Back to Home",
                command=self.show_home,
                fg_color=self.theme.get("primary", "#4C51BF"),
                hover_color=self.theme.get("primary_hover", "#3C41AF"),
                text_color="white",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            back_button.pack(pady=10)

    def check_dpi_scaling(self):
        """Check and adjust for high DPI displays."""
        try:
            # If app is shutting down, don't schedule any new tasks
            if getattr(self, 'is_shutting_down', False):
                return
                
            # Cancel any existing check_dpi_scaling callbacks
            if "check_dpi_scaling" in self.after_ids:
                try:
                    self.root.after_cancel(self.after_ids["check_dpi_scaling"])
                    del self.after_ids["check_dpi_scaling"]
                except Exception as e:
                    self.logger.debug(f"Error canceling check_dpi_scaling callback: {e}")
            
            # Check if the window still exists before proceeding
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            # Get screen width and height
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # Determine if high DPI scaling is needed
            if screen_width > 2560 or screen_height > 1440:  # 2K+ resolution
                ctk.set_widget_scaling(1.2)  # Increase widget size by 20%
                ctk.set_window_scaling(1.1)  # Increase window size by 10%
                self.logger.info(f"Adjusted scaling for high DPI display: {screen_width}x{screen_height}")
            elif screen_width > 1920 or screen_height > 1080:  # Full HD+
                ctk.set_widget_scaling(1.1)  # Increase widget size by 10%
                self.logger.info(f"Adjusted scaling for Full HD+ display: {screen_width}x{screen_height}")
                
        except Exception as e:
            self.logger.warning(f"Failed to adjust DPI scaling: {e}")
            
    def cleanup(self):
        """Clean up resources and cancel scheduled events before closing."""
        try:
            # Flag to indicate the application is shutting down
            self.is_shutting_down = True
            
            # Cancel all scheduled after events
            for after_id_name, after_id in list(self.after_ids.items()):
                try:
                    self.root.after_cancel(after_id)
                    self.logger.debug(f"Cancelled after event: {after_id_name}")
                except Exception as e:
                    self.logger.debug(f"Error cancelling after event {after_id_name}: {e}")
            
            # Clear after IDs dictionary
            self.after_ids.clear()
            
            # Cancel all other after events
            try:
                for widget_id in self.root.tk.call('after', 'info'):
                    try:
                        # Ensure widget_id is a valid integer before trying to cancel
                        id_to_cancel = int(widget_id)
                        if id_to_cancel > 0:
                            self.root.after_cancel(id_to_cancel)
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f"Invalid after ID {widget_id}: {e}")
                    except Exception:
                        pass
            except Exception as e:
                self.logger.debug(f"Error during after events cleanup: {e}")
            
            # Additional cleanup for threads and resources
            if hasattr(self, 'calculation_thread') and getattr(self, 'calculation_thread', None) is not None:
                if self.calculation_thread.is_alive():
                    self._cancel_calculation()
            
            # Wait a brief moment to ensure all cancellations take effect
            time.sleep(0.1)
                
            self.logger.info("Application cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during application cleanup: {e}")
            
    def run(self):
        """Run the application main loop."""
        try:
            # Set shutdown flag to False during application startup
            self.is_shutting_down = False
            
            # Register cleanup function to handle window close
            # Use a safer shutdown sequence
            def on_close():
                try:
                    self.cleanup()
                    self.root.quit()  # Signal to exit mainloop
                    self.root.destroy()  # Destroy the window
                except Exception as e:
                    self.logger.error(f"Error during window close: {e}")
                    # Try one more time to destroy the window
                    try:
                        self.root.destroy()
                    except:
                        pass
            
            self.root.protocol("WM_DELETE_WINDOW", on_close)
            
            # Start the main event loop
            self.root.mainloop()
        except Exception as e:
            # Attempt cleanup
            self.cleanup()
            self.logger.error(f"Error running application: {str(e)}")
            raise

    def safe_after(self, delay, callback, after_id_name=None):
        """Safely schedule an 'after' callback with shutdown check.
        
        Args:
            delay: Delay in milliseconds
            callback: Function to call after delay
            after_id_name: Name to use as key in self.after_ids dictionary
            
        Returns:
            The after ID if scheduled, or None if app is shutting down
        """
        # Check for shutdown flag
        if getattr(self, 'is_shutting_down', False):
            return None
            
        # Check if window still exists
        if not hasattr(self, 'root') or not self.root.winfo_exists():
            return None
            
        # Create a wrapper to check if app is still running before executing callback
        def safe_callback_wrapper():
            if not getattr(self, 'is_shutting_down', False) and hasattr(self, 'root') and self.root.winfo_exists():
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Error in delayed callback: {str(e)}")
        
        # Schedule the callback
        after_id = self.root.after(delay, safe_callback_wrapper)
        
        # Track the ID if name provided
        if after_id_name:
            self.after_ids[after_id_name] = after_id
            
        return after_id