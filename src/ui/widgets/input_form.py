import customtkinter as ctk
from tkinter import messagebox
import logging
import math

class InputForm:
    def __init__(self, parent, theme: dict, methods: list, solve_callback):
        self.parent = parent
        self.theme = theme
        self.solve_callback = solve_callback
        self.method_var = ctk.StringVar(value="Bisection")
        self.function_var = ctk.StringVar()
        self.example_var = ctk.StringVar(value="Select Example")
        self.decimal_var = ctk.BooleanVar(value=False)
        self.matrix_size = 3
        self.matrix_entries = []
        self.logger = logging.getLogger(__name__)
        self.MAX_EPS = 100.0  # Increased maximum allowed epsilon value to 100
        
        # Create the main frame
        self.frame = ctk.CTkFrame(parent, fg_color=theme["bg"])
        
        # Method-specific example functions dictionary
        self.example_functions = {
            "Bisection": {
                "Select Example": "",
                "Function: f(x) = -2+7x-5x^2+6x^3": "-2+7*x-5*x**2+6*x**3",
                "Function: f(x) = 4x^3-6x^2+7x-2.3": "4*x**3-6*x**2+7*x-2.3",
                "Function: f(x) = x^2 - 4": "x**2 - 4",
                "Function: f(x) = sin(x)": "sin(x)",
                "Function: f(x) = cos(x)": "cos(x)",
                "Function: f(x) = x^3 - 2x + 1": "x**3 - 2*x + 1",
                "Function: f(x) = e^x - 2": "exp(x) - 2",
                "Function: f(x) = ln(x) - 1": "log(x) - 1",
                "Function: f(x) = x^4 - 3x^2 + 2": "x**4 - 3*x**2 + 2",
                "Function: f(x) = x^5 - 5x + 3": "x**5 - 5*x + 3",
                "Function: f(x) = tan(x) - x": "tan(x) - x",
                "Function: f(x) = 4x^3 - 6x^2 + 7x - 2.3": "4*x**3 - 6*x**2 + 7*x - 2.3"
            },
            "False Position": {
                "Select Example": "",
                "Function: f(x) = -26+82.3x-88x^2+45.4x^3-9x^4+0.65x^5": "-26+82.3*x-88*x**2+45.4*x**3-9*x**4+0.65*x**5",
                "Function: f(x) = -13-20x+19x^2-3x^3": "-13-20*x+19*x**2-3*x**3",
                "Function: f(x) = x^2 - 4": "x**2 - 4",
                "Function: f(x) = sin(x)": "sin(x)",
                "Function: f(x) = cos(x)": "cos(x)",
                "Function: f(x) = x^3 - 2x + 1": "x**3 - 2*x + 1",
                "Function: f(x) = e^x - 3x": "exp(x) - 3*x",
                "Function: f(x) = ln(x) + x - 2": "log(x) + x - 2",
                "Function: f(x) = x^4 - 10x^2 + 9": "x**4 - 10*x**2 + 9",
                "Function: f(x) = x^3 + 4x^2 - 10": "x**3 + 4*x**2 - 10",
                "Function: f(x) = tan(x) - 2x": "tan(x) - 2*x",
                "Function: f(x) = x ln(x) - 1.2": "x * log(x) - 1.2"
            },
            "Fixed Point": {
                "Select Example": "",
                "Function: g(x) = -0.9x^2+1.7x+2.5": "-0.9*x**2 + 1.7*x + 2.5",
                "Function: g(x) = -x^2+1.8x+2.5": "-x**2 + 1.8*x + 2.5",
                "Function: g(x) = sin(√x)": "sin(sqrt(x))",
                "Function: g(x) = √(2 - x)": "sqrt(2 - x)",
                "Function: g(x) = e^(-x)": "exp(-x)",
                "Function: g(x) = 1/x": "1/x",
                "Function: g(x) = x^2/3": "x**2/3",
                "Function: g(x) = cos(x)/2": "cos(x)/2",
                "Function: g(x) = sin(x)": "sin(x)",
                "Function: g(x) = cos(x)": "cos(x)",
                "Function: g(x) = x^2 - 4": "x**2 - 4",
                "Function: g(x) = x^3 - 2x + 1": "x**3 - 2*x + 1"
            },
            "Newton-Raphson": {
                "Select Example": "",
                "Function: f(x) = -0.9x^2+1.7x+2.5": "-0.9*x**2 + 1.7*x + 2.5",
                "Function: f(x) = -x^2+1.8x+2.5": "-x**2 + 1.8*x + 2.5",
                "Function: f(x) = x^2 - 4": "x**2 - 4",
                "Function: f(x) = sin(x)": "sin(x)",
                "Function: f(x) = cos(x)": "cos(x)",
                "Function: f(x) = x^3 - 2x + 1": "x**3 - 2*x + 1",
                "Function: f(x) = e^x - 5": "exp(x) - 5",
                "Function: f(x) = ln(x) - 1": "log(x) - 1",
                "Function: f(x) = x^4 - 16": "x**4 - 16",
                "Function: f(x) = x^5 - 32": "x**5 - 32",
                "Function: f(x) = tan(x) - 1": "tan(x) - 1",
                "Function: f(x) = e^(-x) - sin(x)": "exp(-x) - sin(x)",
                "Function: f(x) = 2sin(√x) - x": "2*sin(sqrt(x)) - x",
                "Function: f(x) = x^3 + x^2 - 3x - 3": "x**3 + x**2 - 3*x - 3",
                "Function: f(x) = 2 + 6x - 4x^2 + 0.5x^3": "2 + 6*x - 4*x**2 + 0.5*x**3"
            },
            "Secant": {
                "Select Example": "",
                "Function: f(x) = 0.95x^3-5.9x^2+10.9x-6": "0.95*x**3 - 5.9*x**2 + 10.9*x - 6",
                "Function: f(x) = 2x^3-11.7x^2+17.7x-5": "2*x**3 - 11.7*x**2 + 17.7*x - 5",
                "Function: f(x) = x^2 - 4": "x**2 - 4",
                "Function: f(x) = sin(x)": "sin(x)",
                "Function: f(x) = cos(x)": "cos(x)",
                "Function: f(x) = x^3 - 2x + 1": "x**3 - 2*x + 1",
                "Function: f(x) = e^x - 10": "exp(x) - 10",
                "Function: f(x) = ln(x) - 2": "log(x) - 2",
                "Function: f(x) = x^4 - 81": "x**4 - 81",
                "Function: f(x) = x^5 - 243": "x**5 - 243",
                "Function: f(x) = tan(x) - 3": "tan(x) - 3",
                "Function: f(x) = -x^3 + 7.89x + 11": "-x**3 + 7.89*x + 11"
            },
            "Gauss Elimination": {
                "Select Example": "",
                "Example 1 (Book Example2.1):\n2x₁ + x₂ + x₃ = 8\n3x₁ + 4x₂ + 0x₃ = 11\n-2x₁ + 2x₂ + x₃ = 43": "2x₁ + x₂ + x₃ = 8\n3x₁ + 4x₂ + 0x₃ = 11\n-2x₁ + 2x₂ + x₃ = 43",
                "Example 2 (Simple):\n2x + y - z = 8\n-3x - y + 2z = -11\n-2x + y + 2z = -3": "2x + y - z = 8\n-3x - y + 2z = -11\n-2x + y + 2z = -3",
                "Example 3 (Book Example2.2):\n2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 4:\nx + 2y + 3z = 6\n2x + 5y + 2z = 4\n6x - 3y + z = 2": "x + 2y + 3z = 6\n2x + 5y + 2z = 4\n6x - 3y + z = 2",
                "Example 5:\n3x + 2y - z = 1\n2x - 2y + 4z = -2\n-x + 0.5y - z = 0": "3x + 2y - z = 1\n2x - 2y + 4z = -2\n-x + 0.5y - z = 0"
            },
            "Gauss Elimination (Partial Pivoting)": {
                "Select Example": "",
                "Example 1 (Book Example2.7):\n4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6": "\n4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6",
                "Example 2:\n0.003x₁ + 59.14x₂ = 59.17\n5.291x₁ - 6.13x₂ = 46.78": "0.003x₁ + 59.14x₂ = 59.17\n5.291x₁ - 6.13x₂ = 46.78",
                "Example 3 (Book Example2.8):\n2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 4 (Small Coefficients):\n0.0001x₁ + x₂ + x₃ = 1\nx₁ + x₂ - x₃ = 2\nx₁ - x₂ + x₃ = 0": "0.0001x₁ + x₂ + x₃ = 1\nx₁ + x₂ - x₃ = 2\nx₁ - x₂ + x₃ = 0"
            },
            "LU Decomposition": {
                "Select Example": "",
                "Example 1 (Book Example2.3):\n2x₁ + x₂ + x₃ = 8\n3x₁ + 4x₂ + 0x₃ = 11\n-2x₁ + 2x₂ + x₃ = 43": "2x₁ + x₂ + x₃ = 8\n3x₁ + 4x₂ + 0x₃ = 11\n-2x₁ + 2x₂ + x₃ = 43",
                "Example 2 (Simple):\n2x + y - z = 8\n-3x - y + 2z = -11\n-2x + y + 2z = -3": "2x + y - z = 8\n-3x - y + 2z = -11\n-2x + y + 2z = -3",
                "Example 3 (Book Example2.4):\n2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 4:\nx + 2y + 3z = 6\n2x + 5y + 2z = 4\n6x - 3y + z = 2": "x + 2y + 3z = 6\n2x + 5y + 2z = 4\n6x - 3y + z = 2",
                "Example 5:\n3x + 2y - z = 1\n2x - 2y + 4z = -2\n-x + 0.5y - z = 0": "3x + 2y - z = 1\n2x - 2y + 4z = -2\n-x + 0.5y - z = 0"
            },
            "LU Decomposition (Partial Pivoting)": {
                "Select Example": "",
                "Example 1 (Book Example2.9):\n4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6": "\n4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6",
                "Example 2:\n0.003x₁ + 59.14x₂ = 59.17\n5.291x₁ - 6.13x₂ = 46.78": "0.003x₁ + 59.14x₂ = 59.17\n5.291x₁ - 6.13x₂ = 46.78",
                "Example 3 (Book Example 2.10):\n2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 4 (Small Coefficients):\n0.0001x₁ + x₂ + x₃ = 1\nx₁ + x₂ - x₃ = 2\nx₁ - x₂ + x₃ = 0": "0.0001x₁ + x₂ + x₃ = 1\nx₁ + x₂ - x₃ = 2\nx₁ - x₂ + x₃ = 0"
            },
            "Gauss-Jordan": {
                "Select Example": "",
                "Example 1(Book Example2.11):\n4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6": "4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6",
                "Example 2(Book Example2.12):\n2x₁ + x₂ - x₃ = -2\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "\n2x₁ + x₂ - x₃ = -2\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 3:\n2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 4:\nx₁ + 2x₂ + 3x₃ = 14\n2x₁ - x₂ + x₃ = 4\n3x₁ - x₂ - x₃ = 2": "x₁ + 2x₂ + 3x₃ = 14\n2x₁ - x₂ + x₃ = 4\n3x₁ - x₂ - x₃ = 2"
            },
            "Gauss-Jordan (Partial Pivoting)": {
                "Select Example": "",
                "Example 1 (Book Example2.13):\n4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6": "4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6",
                "Example 2 (Book Example2.14):\n2x₁ + x₂ - x₃ = -2\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "2x₁ + x₂ - x₃ = -2\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 3:\n2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 4:\n0.003x₁ + 59.14x₂ = 59.17\n5.291x₁ - 6.13x₂ = 46.78": "0.003x₁ + 59.14x₂ = 59.17\n5.291x₁ - 6.13x₂ = 46.78"
            },
            "Cramer's Rule": {
                "Select Example": "",
                "Example 1 (Book Example 2.13):\n4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6": "4x₁ + x₂ - x₃ = -2\n5x₁ + x₂ + 2x₃ = 4\n6x₁ + x₂ + x₃ = 6",
                "Example 2 (Book Example 2.14):\n2x₁ + x₂ - x₃ = -2\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "\n2x₁ + x₂ - x₃ = -2\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 2:\n2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5": "2x₁ + x₂ - x₃ = 1\n5x₁ + 2x₂ + 2x₃ = -4\n3x₁ + x₂ + x₃ = 5",
                "Example 3:\nx₁ + 2x₂ + 3x₃ = 14\n2x₁ - x₂ + x₃ = 4\n3x₁ - x₂ - x₃ = 2": "x₁ + 2x₂ + 3x₃ = 14\n2x₁ - x₂ + x₃ = 4\n3x₁ - x₂ - x₃ = 2"
            }
        }
        
        # Default example functions (used when method changes)
        self.default_examples = list(self.example_functions["Bisection"].keys())
        
        self.setup_widgets(methods)

    def setup_widgets(self, methods):
        # Function input row with example dropdown
        func_frame = ctk.CTkFrame(self.frame, fg_color=self.theme["bg"])
        func_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")
        
        # Function label and entry
        self.func_label = ctk.CTkLabel(func_frame, text="Function f(x):", text_color=self.theme["text"], 
                    font=("Helvetica", 12, "bold"))
        self.func_label.pack(side="left", padx=10)
        self.func_entry = ctk.CTkEntry(func_frame, width=250, placeholder_text="Enter function (e.g., x**2 - 4)")
        self.func_entry.pack(side="left", padx=5)
        
        # Example functions dropdown
        self.example_var = ctk.StringVar(value="Select Example")
        self.example_menu = ctk.CTkOptionMenu(
            func_frame,
            values=self.default_examples,
            variable=self.example_var,
            command=self.load_example,
            fg_color=self.theme["button"],
            button_color=self.theme["button_hover"],
            button_hover_color=self.theme["accent"],
            width=200
        )
        self.example_menu.pack(side="left", padx=10)

        # Method selection
        method_frame = ctk.CTkFrame(self.frame, fg_color=self.theme["bg"])
        method_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
        
        ctk.CTkLabel(method_frame, text="Method:", text_color=self.theme["text"], 
                    font=("Helvetica", 12, "bold")).pack(side="left", padx=10)
        self.method_menu = ctk.CTkOptionMenu(
            method_frame, 
            variable=self.method_var, 
            values=methods, 
            command=self.update_fields,
            fg_color=self.theme["button"], 
            button_color=self.theme["button_hover"],
            button_hover_color=self.theme["accent"],
            font=("Helvetica", 12, "bold")
        )
        self.method_menu.pack(side="left", padx=5)

        # Parameters frame
        self.input_frame = ctk.CTkFrame(self.frame, fg_color=self.theme["bg"])
        self.input_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # Settings frame
        self.settings_frame = ctk.CTkFrame(self.frame, fg_color=self.theme["bg"])
        self.settings_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Epsilon and Max Iterations
        eps_frame = ctk.CTkFrame(self.settings_frame, fg_color=self.theme["bg"])
        eps_frame.grid(row=0, column=0, columnspan=4, pady=5, sticky="ew")
        
        # Create a more visible epsilon section with label, operator, and value
        ctk.CTkLabel(eps_frame, text="Epsilon:", text_color=self.theme["text"], 
                    font=("Helvetica", 12, "bold")).pack(side="left", padx=10)
        
        # Epsilon operator dropdown with better visibility
        self.eps_operator = ctk.CTkOptionMenu(
            eps_frame,
            values=["<=", ">=", "<", ">", "="],
            width=60,
            fg_color=self.theme["button"],
            button_color=self.theme["button_hover"],
            button_hover_color=self.theme["accent"],
            font=("Helvetica", 12, "bold")
        )
        self.eps_operator.pack(side="left", padx=5)
        self.eps_operator.set("<=")  # Default operator
        
        # Epsilon value entry
        self.eps_entry = ctk.CTkEntry(eps_frame, width=100, placeholder_text="0.0001")
        self.eps_entry.pack(side="left", padx=5)
        
        # Max iterations in a separate row for clarity
        iter_frame = ctk.CTkFrame(self.settings_frame, fg_color=self.theme["bg"])
        iter_frame.grid(row=1, column=0, columnspan=4, pady=5, sticky="ew")
        
        ctk.CTkLabel(iter_frame, text="Max Iterations:", text_color=self.theme["text"],
                    font=("Helvetica", 12, "bold")).pack(side="left", padx=10)
        self.iter_entry = ctk.CTkEntry(iter_frame, width=150, placeholder_text="e.g., 50")
        self.iter_entry.pack(side="left", padx=5)
        
        # Round Result checkbox and decimal places
        self.round_var = ctk.BooleanVar(value=True)
        self.round_checkbox = ctk.CTkCheckBox(
            self.settings_frame, 
            text="Round Result", 
            variable=self.round_var,
            command=self.toggle_decimal_entry,
            text_color=self.theme["text"],
            fg_color=self.theme["button"],
            hover_color=self.theme["button_hover"],
            checkmark_color="#FFFFFF"
        )
        self.round_checkbox.grid(row=2, column=0, columnspan=2, pady=5, padx=10, sticky="w")
        
        ctk.CTkLabel(self.settings_frame, text="Decimal Places:", text_color=self.theme["text"]).grid(row=2, column=2, pady=5, padx=10, sticky="e")
        self.decimal_entry = ctk.CTkEntry(self.settings_frame, width=150, placeholder_text="e.g., 6")
        self.decimal_entry.grid(row=2, column=3, pady=5)
        
        # Stop condition - Make it more prominent
        self.stop_frame = ctk.CTkFrame(self.frame, fg_color=self.theme["bg"], border_width=2, border_color=self.theme["accent"])
        self.stop_frame.grid(row=4, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # Add a title for the stop condition with larger font
        ctk.CTkLabel(
            self.stop_frame, 
            text="Stop Condition:", 
            text_color=self.theme["text"],
            font=("Helvetica", 14, "bold")
        ).pack(side="left", padx=10)
        
        self.stop_var = ctk.StringVar(value="Epsilon")
        self.epsilon_radio = ctk.CTkRadioButton(
            self.stop_frame, 
            text="Stop by Epsilon", 
            variable=self.stop_var, 
            value="Epsilon", 
            text_color=self.theme["text"],
            fg_color=self.theme.get("primary", self.theme["button"]),
            hover_color=self.theme.get("primary_hover", self.theme["button_hover"]),
            border_color=self.theme.get("primary_hover", self.theme["button_hover"]),
            font=("Helvetica", 12)
        )
        self.epsilon_radio.pack(side="left", padx=20)
        
        self.iterations_radio = ctk.CTkRadioButton(
            self.stop_frame, 
            text="Stop by Iterations", 
            variable=self.stop_var, 
            value="Iterations", 
            text_color=self.theme["text"],
            fg_color=self.theme.get("primary", self.theme["button"]),
            hover_color=self.theme.get("primary_hover", self.theme["button_hover"]),
            border_color=self.theme.get("primary_hover", self.theme["button_hover"]),
            font=("Helvetica", 12)
        )
        self.iterations_radio.pack(side="left", padx=20)

        # Solve button
        self.solve_button = ctk.CTkButton(
            self.frame, 
            text="Solve", 
            command=self.on_solve, 
            fg_color=self.theme["button"], 
            hover_color=self.theme["button_hover"], 
            font=("Helvetica", 14, "bold"),
            height=40
        )
        self.solve_button.grid(row=5, column=0, columnspan=2, pady=15)
        
        self.update_fields("Bisection")

    def update_fields(self, method: str):
        """Update the input fields based on the selected method."""
        try:
            # Clear previous fields
            for widget in self.input_frame.winfo_children():
                widget.destroy()
            
            # Create method-specific input fields
            self.entries = {}
            
            # Add a title for the parameters
            ctk.CTkLabel(
                self.input_frame, 
                text=f"{method} Parameters:", 
                text_color=self.theme["text"],
                font=("Helvetica", 12, "bold")
            ).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
            
            if method in ["Gauss Elimination", "Gauss Elimination (Partial Pivoting)", 
                         "LU Decomposition", "LU Decomposition (Partial Pivoting)",
                         "Gauss-Jordan", "Gauss-Jordan (Partial Pivoting)", "Cramer's Rule"]:
                # Hide function input for linear system methods
                self.func_label.pack_forget()
                self.func_entry.pack_forget()
                
                # Matrix size and example selection row
                controls_frame = ctk.CTkFrame(self.input_frame, fg_color=self.theme["bg"])
                controls_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
                
                # Matrix size selection
                ctk.CTkLabel(controls_frame, text="Matrix Size:", text_color=self.theme["text"]).pack(side="left", padx=10)
                size_var = ctk.StringVar(value="3")
                size_menu = ctk.CTkOptionMenu(
                    controls_frame,
                    values=["2", "3", "4", "5"],
                    variable=size_var,
                    command=self.update_matrix_size,
                    fg_color=self.theme["button"],
                    button_color=self.theme["button_hover"],
                    button_hover_color=self.theme["accent"]
                )
                size_menu.pack(side="left", padx=5)
                
                # Example dropdown for matrix methods
                ctk.CTkLabel(controls_frame, text="Examples:", text_color=self.theme["text"]).pack(side="left", padx=10)
                self.matrix_example_var = ctk.StringVar(value="Select Example")
                self.matrix_example_menu = ctk.CTkOptionMenu(
                    controls_frame,
                    values=["Select Example"] + list(self.example_functions.get(method, {}).keys())[1:],
                    variable=self.matrix_example_var,
                    command=self.load_example,
                    fg_color=self.theme["button"],
                    button_color=self.theme["button_hover"],
                    button_hover_color=self.theme["accent"]
                )
                self.matrix_example_menu.pack(side="left", padx=5)
                
                # Matrix input frame with [A|b] format
                matrix_frame = ctk.CTkFrame(self.input_frame, fg_color=self.theme["bg"])
                matrix_frame.grid(row=2, column=0, columnspan=2, pady=5)
                
                # Add [A|b] label on the left side
                ctk.CTkLabel(
                    matrix_frame,
                    text="[A|b] =",
                    text_color=self.theme["text"],
                    font=("Helvetica", 12, "bold")
                ).grid(row=0, column=0, padx=(0, 10), sticky="e", rowspan=self.matrix_size)
                
                # Create matrix entries frame with no padding
                entries_frame = ctk.CTkFrame(matrix_frame, fg_color=self.theme["bg"])
                entries_frame.grid(row=0, column=1, padx=0, pady=0, rowspan=self.matrix_size)
                
                # Create matrix entries
                self.matrix_entries = []
                for i in range(self.matrix_size):
                    row_entries = []
                    for j in range(self.matrix_size + 1):  # +1 for the b vector
                        # Add a vertical separator before the b column
                        if j == self.matrix_size:
                            separator = ctk.CTkFrame(entries_frame, width=4, fg_color=self.theme["accent"])
                            separator.grid(row=0, column=j, sticky="ns", padx=3, pady=0, rowspan=self.matrix_size)
                        
                        # Calculate the actual column position considering the separator
                        col_pos = j if j < self.matrix_size else j + 1
                        
                        # Create an entry with better size and styling
                        entry = ctk.CTkEntry(
                            entries_frame, 
                            width=65, 
                            height=25, 
                            placeholder_text="0",
                            text_color=self.theme["text"],
                            fg_color=self.theme["table_bg"],
                            border_color=self.theme["accent"],
                            border_width=1
                        )
                        # Use better padding for improved layout
                        entry.grid(row=i, column=col_pos, padx=3, pady=2, ipady=2)
                        
                        # Set font size for better readability
                        entry.configure(font=("Helvetica", 10))
                        
                        row_entries.append(entry)
                    self.matrix_entries.append(row_entries)
                
                # Configure row spacing to be minimal
                for i in range(self.matrix_size):
                    entries_frame.grid_rowconfigure(i, minsize=10, pad=0)
                    
                # Set row weights to 0 to prevent expansion
                for i in range(self.matrix_size):
                    entries_frame.grid_rowconfigure(i, weight=0)
                    
                # Remove any internal padding in the frame
                entries_frame.configure(corner_radius=0)
                
                # Remove the existing separator that was added after creating all entries
                if hasattr(self, 'matrix_separator'):
                    self.matrix_separator.grid_forget()
                
                # Hide settings and stop condition frames for matrix methods
                self.settings_frame.grid_remove()
                self.stop_frame.grid_remove()
                
                # Hide decimal places frame for matrix methods
                if hasattr(self, 'decimal_frame'):
                    self.decimal_frame.grid_remove()
            else:
                # For non-matrix methods, update the example menu and show function input
                if method in self.example_functions:
                    self.example_menu.configure(values=["Select Example"] + list(self.example_functions.get(method, {}).keys())[1:])
                    self.example_var.set("Select Example")
                
                # Show function input for other methods
                self.func_label.pack(side="left", padx=10)
                self.func_entry.pack(side="left", padx=5)
                
                if method in ["Bisection", "False Position"]:
                    ctk.CTkLabel(self.input_frame, text="Lower Bound (Xl):", text_color=self.theme["text"]).grid(row=1, column=0, pady=5, padx=10, sticky="e")
                    self.entries["xl"] = ctk.CTkEntry(self.input_frame, width=150, placeholder_text="e.g., 0")
                    self.entries["xl"].grid(row=1, column=1, pady=5)
                    
                    ctk.CTkLabel(self.input_frame, text="Upper Bound (Xu):", text_color=self.theme["text"]).grid(row=2, column=0, pady=5, padx=10, sticky="e")
                    self.entries["xu"] = ctk.CTkEntry(self.input_frame, width=150, placeholder_text="e.g., 2")
                    self.entries["xu"].grid(row=2, column=1, pady=5)
                    
                elif method in ["Fixed Point", "Newton-Raphson"]:
                    ctk.CTkLabel(self.input_frame, text="Initial Value (X0):", text_color=self.theme["text"]).grid(row=1, column=0, pady=5, padx=10, sticky="e")
                    self.entries["xi"] = ctk.CTkEntry(self.input_frame, width=150, placeholder_text="e.g., 1")
                    self.entries["xi"].grid(row=1, column=1, pady=5)
                    
                    # Add auto-generate g(x) checkbox for Fixed Point method only
                    if method == "Fixed Point":
                        self.auto_g_var = ctk.BooleanVar(value=False)
                        auto_g_frame = ctk.CTkFrame(self.input_frame, fg_color=self.theme["bg"])
                        auto_g_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="w")
                        
                        auto_g_check = ctk.CTkCheckBox(
                            auto_g_frame, 
                            text="Auto-generate optimal g(x) function", 
                            variable=self.auto_g_var,
                            text_color=self.theme["text"],
                            fg_color=self.theme["accent"],
                            hover_color=self.theme.get("accent_hover", self.theme["accent"])
                        )
                        auto_g_check.pack(side="left", padx=20)
                        
                        # Add an info label with tooltip-like explanation
                        info_label = ctk.CTkLabel(
                            auto_g_frame,
                            text="ⓘ",
                            text_color=self.theme["accent"],
                            font=("Helvetica", 16, "bold")
                        )
                        info_label.pack(side="left", padx=5)
                        
                        # Add tooltip functionality (hover text)
                        tooltip_text = "Automatically generates multiple g(x) candidates and selects the best one for convergence.\nThis will rearrange f(x) = 0 into x = g(x) using algebraic manipulation."
                        
                        def show_tooltip(event):
                            tooltip = ctk.CTkToplevel(self.frame)
                            tooltip.wm_overrideredirect(True)
                            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                            tooltip.attributes('-topmost', True)
                            
                            # Create tooltip content
                            tooltip_label = ctk.CTkLabel(
                                tooltip,
                                text=tooltip_text,
                                text_color=self.theme["text"],
                                fg_color=self.theme["fg"],
                                corner_radius=5,
                                font=("Helvetica", 10),
                                wraplength=300,
                                justify="left"
                            )
                            tooltip_label.pack(padx=10, pady=5)
                            
                            # Store the tooltip reference for later destruction
                            info_label.tooltip = tooltip
                            
                        def hide_tooltip(event):
                            if hasattr(info_label, "tooltip"):
                                info_label.tooltip.destroy()
                                
                        info_label.bind("<Enter>", show_tooltip)
                        info_label.bind("<Leave>", hide_tooltip)
                
                elif method == "Secant":
                    ctk.CTkLabel(self.input_frame, text="First Initial Value (X-1):", text_color=self.theme["text"]).grid(row=1, column=0, pady=5, padx=10, sticky="e")
                    self.entries["xi_minus_1"] = ctk.CTkEntry(self.input_frame, width=150, placeholder_text="e.g., 0")
                    self.entries["xi_minus_1"].grid(row=1, column=1, pady=5)
                    
                    ctk.CTkLabel(self.input_frame, text="Second Initial Value (X0):", text_color=self.theme["text"]).grid(row=2, column=0, pady=5, padx=10, sticky="e")
                    self.entries["xi"] = ctk.CTkEntry(self.input_frame, width=150, placeholder_text="e.g., 1")
                    self.entries["xi"].grid(row=2, column=1, pady=5)
                
                # Show settings and stop condition frames for other methods
                self.settings_frame.grid()
                self.stop_frame.grid()
                
                # Show decimal places frame for other methods if it exists
                if hasattr(self, 'decimal_frame'):
                    self.decimal_frame.grid()

        except Exception as e:
            self.logger.error(f"Error updating fields: {str(e)}")
            messagebox.showerror("Error", f"An error occurred while updating fields: {str(e)}")

    def update_matrix_size(self, size: str):
        """Update the matrix size and recreate the input fields."""
        self.matrix_size = int(size)
        self.update_fields("Gauss Elimination")

    def validate_input(self) -> tuple[bool, str]:
        """Validate all input fields and return (is_valid, error_message)."""
        try:
            method = self.method_var.get()
            
            # Validate matrix and vector for linear system methods
            if method in ["Gauss Elimination", "Gauss Elimination (Partial Pivoting)", 
                         "LU Decomposition", "LU Decomposition (Partial Pivoting)",
                         "Gauss-Jordan", "Gauss-Jordan (Partial Pivoting)", "Cramer's Rule"]:
                # Check if matrix entries are valid numbers
                matrix = []
                vector = []
                for i in range(self.matrix_size):
                    row = []
                    for j in range(self.matrix_size + 1):
                        try:
                            value = float(self.matrix_entries[i][j].get() or "0")
                            if j < self.matrix_size:
                                row.append(value)
                            else:
                                vector.append(value)
                        except ValueError:
                            return False, f"Invalid value for {'matrix entry A' + str(i+1) + str(j+1) if j < self.matrix_size else 'vector entry b' + str(i+1)}"
                    matrix.append(row)
                
                return True, ""
            
            # Validate function for other methods
            func = self.func_entry.get().strip()
            if not func:
                return False, "Function cannot be empty"
                
            # Validate method-specific parameters
            if method in ["Bisection", "False Position"]:
                # Validate xl
                try:
                    xl = float(self.entries["xl"].get() or "0")
                except ValueError:
                    return False, "Lower bound (Xl) must be a number"
                    
                # Validate xu
                try:
                    xu = float(self.entries["xu"].get() or "0")
                except ValueError:
                    return False, "Upper bound (Xu) must be a number"
                    
                # Check that xl < xu
                if xl >= xu:
                    return False, "Lower bound (Xl) must be less than upper bound (Xu)"
                    
            elif method in ["Fixed Point", "Newton-Raphson"]:
                # Validate xi
                try:
                    xi = float(self.entries["xi"].get() or "0")
                except ValueError:
                    return False, "Initial value (X0) must be a number"
                    
            elif method == "Secant":
                # Validate xi_minus_1
                try:
                    xi_minus_1 = float(self.entries["xi_minus_1"].get() or "0")
                except ValueError:
                    return False, "First initial value (X-1) must be a number"
                    
                # Validate xi
                try:
                    xi = float(self.entries["xi"].get() or "0")
                except ValueError:
                    return False, "Second initial value (X0) must be a number"
                    
                # Check that xi_minus_1 != xi
                if xi_minus_1 == xi:
                    return False, "First and second initial values must be different"
            
            # Validate epsilon
            try:
                eps = float(self.eps_entry.get() or "0.0001")
                if eps <= 0:
                    return False, "Error tolerance must be positive"
                if eps > self.MAX_EPS:
                    return False, f"Error tolerance must be less than {self.MAX_EPS}"
            except ValueError:
                return False, "Error tolerance must be a number"
                
            # Validate max iterations
            try:
                max_iter = int(self.iter_entry.get() or "50")
                if max_iter <= 0:
                    return False, "Maximum iterations must be positive"
            except ValueError:
                return False, "Maximum iterations must be an integer"
                
            # Validate decimal places
            if self.round_var.get():
                try:
                    decimal_places = int(self.decimal_entry.get() or "6")
                    if decimal_places < 0:
                        return False, "Decimal places must be non-negative"
                except ValueError:
                    return False, "Decimal places must be an integer"
            
            return True, ""
        except Exception as e:
            self.logger.error(f"Input validation error: {str(e)}")
            return False, f"Input validation error: {str(e)}"

    def on_solve(self):
        is_valid, error_msg = self.validate_input()
        if not is_valid:
            messagebox.showerror("Input Error", error_msg)
            return

        try:
            method = self.method_var.get()
            if method in ["Gauss Elimination", "Gauss Elimination (Partial Pivoting)", 
                         "LU Decomposition", "LU Decomposition (Partial Pivoting)",
                         "Gauss-Jordan", "Gauss-Jordan (Partial Pivoting)", "Cramer's Rule"]:
                # For linear system methods, only pass matrix and vector
                matrix = []
                vector = []
                for i in range(self.matrix_size):
                    row = []
                    # Get the matrix entries (A)
                    for j in range(self.matrix_size):
                        try:
                            value = float(self.matrix_entries[i][j].get() or "0")
                            row.append(value)
                        except (ValueError, IndexError) as e:
                            self.logger.error(f"Error getting matrix value at [{i}][{j}]: {str(e)}")
                            messagebox.showerror("Input Error", f"Invalid value in matrix at position [{i+1}][{j+1}]")
                            return
                    matrix.append(row)
                    
                    # Get the vector entry (b)
                    try:
                        b_value = float(self.matrix_entries[i][self.matrix_size].get() or "0")
                        vector.append(b_value)
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Error getting vector value at [{i}]: {str(e)}")
                        messagebox.showerror("Input Error", f"Invalid value in vector at position [{i+1}]")
                        return
                
                # Call solve_callback with matrix and vector as params
                # Use fixed decimal places (6) for matrix methods
                self.solve_callback(
                    f_str="System of Linear Equations",
                    method=method,
                    params={"matrix": str(matrix), "vector": str(vector)},
                    eps=None,
                    eps_operator=None,
                    max_iter=None,
                    stop_by_eps=None,
                    decimal_places=6
                )
            else:
                # For other methods, use the existing logic with improved epsilon handling
                func = self.func_entry.get().strip()
                eps = float(self.eps_entry.get() or "0.0001")
                eps_operator = self.eps_operator.get()
                max_iter = int(self.iter_entry.get() or "50")
                decimal_places = int(self.decimal_entry.get() or "6") if self.round_var.get() else 10
                stop_by_eps = self.stop_var.get() == "Epsilon"
                
                # Log the stop condition and epsilon operator for debugging
                self.logger.info(f"Stop by Epsilon: {stop_by_eps}, Epsilon: {eps}, Operator: {eps_operator}")
                
                # Pass parameters based on the method
                params = {key: float(entry.get() or "0") for key, entry in self.entries.items()}
                
                # Add auto_generate_g parameter for Fixed Point method
                if method == "Fixed Point" and hasattr(self, 'auto_g_var'):
                    params["auto_generate_g"] = self.auto_g_var.get()
                
                # Call the solve callback with all parameters using keyword arguments
                self.solve_callback(
                    f_str=func,
                    method=method,
                    params=params,
                    eps=eps,
                    eps_operator=eps_operator,
                    max_iter=max_iter,
                    stop_by_eps=stop_by_eps,
                    decimal_places=decimal_places
                )
        except Exception as e:
            self.logger.error(f"Error in solve callback: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def update_theme(self, theme: dict):
        """Update the theme for all elements in the input form."""
        self.theme = theme
        
        # Update main frame
        self.frame.configure(fg_color=theme["bg"])
        self.input_frame.configure(fg_color=theme["bg"])
        
        # Update labels
        for widget in self.frame.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and hasattr(widget, "winfo_exists") and widget.winfo_exists():
                widget.configure(text_color=theme["text"])
        
        # Update method dropdown
        if hasattr(self, "method_menu") and self.method_menu.winfo_exists():
            self.method_menu.configure(
                fg_color=theme["button"],
                button_color=theme["button"],
                button_hover_color=theme["button_hover"],
                dropdown_fg_color=theme["bg"],
                dropdown_hover_color=theme["button_hover"],
                dropdown_text_color=theme["text"]
            )
        
        # Update example dropdown
        if hasattr(self, "example_menu") and self.example_menu.winfo_exists():
            self.example_menu.configure(
                fg_color=theme["button"],
                button_color=theme["button"],
                button_hover_color=theme["button_hover"],
                dropdown_fg_color=theme["bg"],
                dropdown_hover_color=theme["button_hover"],
                dropdown_text_color=theme["text"]
            )
        
        # Update matrix example dropdown if it exists
        if hasattr(self, "matrix_example_menu") and self.matrix_example_menu.winfo_exists():
            self.matrix_example_menu.configure(
                fg_color=theme["button"],
                button_color=theme["button"],
                button_hover_color=theme["button_hover"],
                dropdown_fg_color=theme["bg"],
                dropdown_hover_color=theme["button_hover"],
                dropdown_text_color=theme["text"]
            )
        
        # Update matrix entries if they exist
        if hasattr(self, "matrix_entries") and self.matrix_entries:
            for row in self.matrix_entries:
                for entry in row:
                    if hasattr(entry, "winfo_exists") and entry.winfo_exists():
                        entry.configure(
                            text_color=theme["text"],
                            fg_color=theme["table_bg"],
                            border_color=theme["accent"]
                        )
        
        # Update function entry
        if hasattr(self, "func_entry") and self.func_entry.winfo_exists():
            self.func_entry.configure(
                text_color=theme["text"],
                fg_color=theme["table_bg"],
                border_color=theme["accent"]
            )
        
        # Update other entries
        for entry in self.entries.values():
            if hasattr(entry, "winfo_exists") and entry.winfo_exists():
                entry.configure(
                    text_color=theme["text"],
                    fg_color=theme["table_bg"],
                    border_color=theme["accent"]
                )
        
        # Update epsilon entry
        if hasattr(self, "eps_entry") and self.eps_entry.winfo_exists():
            self.eps_entry.configure(
                text_color=theme["text"],
                fg_color=theme["table_bg"],
                border_color=theme["accent"]
            )
        
        # Update iterations entry
        if hasattr(self, "iter_entry") and self.iter_entry.winfo_exists():
            self.iter_entry.configure(
                text_color=theme["text"],
                fg_color=theme["table_bg"],
                border_color=theme["accent"]
            )
        
        # Update decimal places entry
        if hasattr(self, "decimal_entry") and self.decimal_entry.winfo_exists():
            self.decimal_entry.configure(
                text_color=theme["text"],
                fg_color=theme["table_bg"],
                border_color=theme["accent"]
            )
        
        # Update solve button
        if hasattr(self, "solve_button") and self.solve_button.winfo_exists():
            self.solve_button.configure(
                fg_color=theme["button"],
                hover_color=theme["button_hover"],
                text_color=theme["table_bg"]
            )
        
        # Update matrix separator if it exists
        if hasattr(self, "matrix_separator") and self.matrix_separator.winfo_exists():
            self.matrix_separator.configure(fg_color=theme["accent"])

    def toggle_decimal_entry(self):
        """Enable or disable the decimal places entry based on the round checkbox."""
        if self.round_var.get():
            self.decimal_entry.configure(state="normal")
        else:
            self.decimal_entry.configure(state="disabled")

    def load_example(self, example_name: str):
        """Load the selected example function into the input field."""
        method = self.method_var.get()
        if method in self.example_functions and example_name in self.example_functions[method]:
            if method in ["Gauss Elimination", "Gauss Elimination (Partial Pivoting)", "LU Decomposition", "LU Decomposition (Partial Pivoting)", "Gauss-Jordan", "Gauss-Jordan (Partial Pivoting)", "Cramer's Rule"]:
                # Parse the system of equations into matrix form
                equations = self.example_functions[method][example_name].split('\n')
                if not equations or example_name == "Select Example":
                    return
                
                # Clear current matrix entries
                for i in range(self.matrix_size):
                    for j in range(self.matrix_size + 1):
                        if j < self.matrix_size:
                            # Matrix entries are at their original indices
                            self.matrix_entries[i][j].delete(0, "end")
                        else:
                            # Vector entries are at the last position in each row
                            self.matrix_entries[i][j].delete(0, "end")
                
                # Parse equations and fill matrix
                matrix = []
                vector = []
                for eq in equations:
                    if not eq.strip() or ":" in eq:  # Skip empty lines or titles
                        continue
                    # Split at '=' and get left and right sides
                    left, right = eq.split('=')
                    # Convert right side to float
                    b = float(right.strip())
                    vector.append(b)
                    
                    # Parse coefficients
                    coeffs = [0] * self.matrix_size
                    # Replace subscripts with regular x for parsing
                    left = left.replace("x₁", "x").replace("x₂", "y").replace("x₃", "z")
                    terms = left.replace('-', '+-').split('+')
                    for term in terms:
                        term = term.strip()
                        if not term:
                            continue
                        
                        # Handle special cases
                        if term.startswith('-'):
                            term = term[1:]
                            coef = -1
                        else:
                            coef = 1
                            
                        if 'x' in term:
                            if term == 'x':
                                coeffs[0] = coef
                            elif term == '-x':
                                coeffs[0] = -coef
                            else:
                                val = term.split('x')[0].strip()
                                if val:
                                    coeffs[0] = float(val) * coef
                        elif 'y' in term:
                            if term == 'y':
                                coeffs[1] = coef
                            elif term == '-y':
                                coeffs[1] = -coef
                            else:
                                val = term.split('y')[0].strip()
                                if val:
                                    coeffs[1] = float(val) * coef
                        elif 'z' in term:
                            if term == 'z':
                                coeffs[2] = coef
                            elif term == '-z':
                                coeffs[2] = -coef
                            else:
                                val = term.split('z')[0].strip()
                                if val:
                                    coeffs[2] = float(val) * coef
                    
                    matrix.append(coeffs)
                
                # Fill the matrix entries
                for i in range(min(len(matrix), self.matrix_size)):
                    # Fill matrix entries (A)
                    for j in range(self.matrix_size):
                        self.matrix_entries[i][j].insert(0, str(matrix[i][j]))
                    
                    # Fill vector entry (b)
                    if i < len(vector):
                        self.matrix_entries[i][self.matrix_size].insert(0, str(vector[i]))
            else:
                self.func_entry.delete(0, "end")
                self.func_entry.insert(0, self.example_functions[method][example_name])