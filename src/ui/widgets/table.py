import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np
import pandas as pd
import math
from collections import OrderedDict

class ResultTable:
    def __init__(self, parent, theme=None, height=None, width=None, fixed_position=False):
        """
        Initialize the result table.
        
        Args:
            parent: Parent widget
            theme: Theme dictionary
            height: Optional height constraint
            width: Optional width constraint
            fixed_position: Whether to use a fixed position that doesn't expand
        """
        self.theme = theme or {}
        self.logger = logging.getLogger(__name__)
        self.fixed_position = fixed_position
        
        # Create the main frame with a fixed height if provided
        self.table_frame = ctk.CTkFrame(parent, fg_color=self.theme.get("bg", "#F0F4F8"))
        
        # Set up the frame with fixed height/width and proper packing
        if height:
            self.table_frame.configure(height=height)
        if width:
            self.table_frame.configure(width=width)
            
        # Always use pack with fill="both" but don't expand to keep fixed size
        self.table_frame.pack(fill="both", padx=2, pady=2)
        
        # Prevent the frame from resizing based on its contents
        if fixed_position or height:
            self.table_frame.pack_propagate(False)
        
        # Create a frame for the table header
        self.header_frame = ctk.CTkFrame(self.table_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
        self.header_frame.pack(fill="x", side="top")
        
        # Create a separate frame for table content with scrollbars
        self.content_frame = ctk.CTkFrame(self.table_frame, fg_color=self.theme.get("bg", "#F0F4F8"))
        self.content_frame.pack(fill="both", expand=True)
        
        # Create vertical scrollbar for the content
        self.scrollbar_y = ttk.Scrollbar(self.content_frame, orient="vertical")
        self.scrollbar_y.pack(side="right", fill="y")
        
        # Create horizontal scrollbar if needed
        self.scrollbar_x = ttk.Scrollbar(self.table_frame, orient="horizontal")
        self.scrollbar_x.pack(side="bottom", fill="x", expand=False)
        
        # Create the table (treeview)
        self.table = ttk.Treeview(
            self.content_frame, 
            style="Custom.Treeview", 
            show="headings", 
            selectmode="browse",
            yscrollcommand=self.scrollbar_y.set,
            xscrollcommand=self.scrollbar_x.set
        )
        self.table.pack(fill="both", expand=True)
        
        # Configure scrollbars
        self.scrollbar_y.config(command=self.table.yview)
        self.scrollbar_x.config(command=self.table.xview)
        
        # Configure hover effect
        hover_color = self.theme.get("table_hover", "#E2E8F0")
        self.table.tag_configure('hover', background=hover_color)
        
        # Configure alternating row colors
        odd_row_color = self.theme.get("table_odd_row", "#F8FAFC")
        even_row_color = self.theme.get("table_even_row", "#FFFFFF")
        self.table.tag_configure('oddrow', background=odd_row_color)
        self.table.tag_configure('evenrow', background=even_row_color)
        
        # Configure special row types
        self.table.tag_configure('info', background="#E6F2FF", foreground="#1E40AF")
        self.table.tag_configure('success', background="#E6F9E6", foreground="#166534")
        self.table.tag_configure('warning', background="#FFF5E6", foreground="#9A3412")
        self.table.tag_configure('error', background="#FFE6E6", foreground="#BE123C")
        self.table.tag_configure('result', background="#F0F9FF", foreground="#0369A1", font=('Helvetica', 10, 'bold'))
        
        # Configure row height and padding for all rows - reduced height
        style = ttk.Style()
        style.configure("Custom.Treeview", rowheight=28)  # Reduced from 35 to 28
        
        # Bind hover events
        self.table.bind('<Motion>', self._on_motion)
        self.table.bind('<Leave>', self._on_leave)
        
        # Set up mousewheel scrolling
        self._setup_mousewheel_scrolling()
        
        # Initialize column sort state
        self.sort_columns = {}  # {column_id: ascending}
        
        # Bind header click for sorting
        self.table.bind("<Button-1>", self._on_header_click)
        
    def _setup_mousewheel_scrolling(self):
        """Set up mousewheel scrolling with improved cross-platform support and better focus handling."""
        # Keep track of the active bindings to manage them properly
        self.active_bindings = []
        
        # Define a more efficient mousewheel handler with better platform detection
        def _on_mousewheel(event):
            """Handle mousewheel scrolling with improved cross-platform support."""
            try:
                # Ensure the table still exists and has focus
                if not self.table.winfo_exists():
                    return
                
                # Get current scroll position to detect edges
                current_pos = self.table.yview()
                
                # Calculate appropriate scroll amount based on platform detection
                scroll_amount = 0
                
                # Windows OS (event.delta is usually multiples of 120)
                if hasattr(event, "delta") and abs(event.delta) >= 120:
                    # Scale for smoother scrolling (delta of 120 = 1 unit)
                    scroll_amount = -1 * (event.delta // 120)
                    
                # macOS (event.delta with smaller values)
                elif hasattr(event, "delta") and abs(event.delta) < 20:
                    scroll_amount = -1 * event.delta
                    
                # Linux/Unix (Button-4/Button-5)
                elif hasattr(event, "num"):
                    if event.num == 4:
                        scroll_amount = -1
                    elif event.num == 5:
                        scroll_amount = 1
                
                # Apply scroll if we have a non-zero amount
                if scroll_amount != 0:
                    # Apply scroll
                    self.table.yview_scroll(int(scroll_amount), "units")
                    
                    # Check if we hit an edge (view didn't change despite scroll attempt)
                    if self.table.yview() == current_pos and scroll_amount != 0:
                        # We're at the edge - let event propagate to parent
                        return
                    else:
                        # Scrolled successfully - prevent further propagation
                        return "break"
                        
            except Exception as e:
                # Log and continue - non-critical errors shouldn't disrupt the user experience
                self.logger.debug(f"Scroll handling error (non-critical): {str(e)}")
            
            # Allow event to propagate if we didn't handle it
            return
            
        # More efficiently manage mousewheel bindings
        def _bind_to_widget(widget):
            """Bind mousewheel events to a specific widget."""
            if not widget.winfo_exists():
                return
                
            try:
                # Only bind when the widget has focus
                # Windows and macOS
                widget.bind("<MouseWheel>", _on_mousewheel, add="+")
                # Linux
                widget.bind("<Button-4>", _on_mousewheel, add="+")
                widget.bind("<Button-5>", _on_mousewheel, add="+")
            except Exception as e:
                self.logger.error(f"Error binding mousewheel to widget: {str(e)}")
        
        # Bind to both the table and scrollbar for a more intuitive experience
        _bind_to_widget(self.table)
        _bind_to_widget(self.scrollbar_y)
        
    def _on_motion(self, event):
        """Handle mouse motion over the table."""
        try:
            item = self.table.identify_row(event.y)
            if item:
                # Get current tags
                tags = list(self.table.item(item, "tags"))
                
                # Skip special rows
                if any(tag in tags for tag in ['info', 'success', 'warning', 'error', 'result']):
                    return
                    
                # Add hover tag if not already present
                if 'hover' not in tags:
                    self.table.item(item, tags=tags + ['hover'])
                    
                # Remove hover from all other items
                for other_item in self.table.get_children():
                    if other_item != item and 'hover' in self.table.item(other_item, "tags"):
                        other_tags = list(self.table.item(other_item, "tags"))
                        other_tags.remove('hover')
                        self.table.item(other_item, tags=other_tags)
        except Exception as e:
            self.logger.error(f"Error in hover effect: {str(e)}")
            
    def _on_leave(self, event):
        """Handle mouse leaving the table."""
        try:
            for item in self.table.get_children():
                tags = list(self.table.item(item, "tags"))
                if 'hover' in tags:
                    tags.remove('hover')
                    self.table.item(item, tags=tags)
        except Exception as e:
            self.logger.error(f"Error in hover leave: {str(e)}")
            
    def _on_header_click(self, event):
        """Handle header click for sorting."""
        try:
            region = self.table.identify_region(event.x, event.y)
            if region == "heading":
                column = self.table.identify_column(event.x)
                column_id = self.table["columns"][int(column.replace('#', '')) - 1]
                
                # Toggle sort order for this column
                ascending = not self.sort_columns.get(column_id, True)
                self.sort_columns = {column_id: ascending}  # Reset other columns
                
                # Get all items
                data = []
                for item in self.table.get_children():
                    values = self.table.item(item, "values")
                    tags = self.table.item(item, "tags")
                    data.append((values, tags))
                
                # Remove all items
                for item in self.table.get_children():
                    self.table.delete(item)
                
                # Sort data by the selected column
                col_idx = self.table["columns"].index(column_id)
                
                # Sort preserving special rows
                special_rows = []
                numeric_rows = []
                text_rows = []
                
                for values, tags in data:
                    if any(tag in tags for tag in ['info', 'success', 'warning', 'error', 'result']):
                        special_rows.append((values, tags))
                    else:
                        # Try to determine if value is numeric
                        value = values[col_idx] if col_idx < len(values) else ""
                        try:
                            # Remove any % or other non-numeric characters
                            clean_value = value.replace('%', '').strip() if isinstance(value, str) else value
                            num_value = float(clean_value)
                            numeric_rows.append((values, tags, num_value))
                        except (ValueError, TypeError):
                            text_rows.append((values, tags, str(value).lower()))
                
                # Sort numeric and text rows separately
                numeric_rows.sort(key=lambda x: x[2], reverse=not ascending)
                text_rows.sort(key=lambda x: x[2], reverse=not ascending)
                
                # Add back in sorted order: numeric, text, special
                for values, tags, _ in numeric_rows:
                    item = self.table.insert("", "end", values=values)
                    self.table.item(item, tags=tags)
                    
                for values, tags, _ in text_rows:
                    item = self.table.insert("", "end", values=values)
                    self.table.item(item, tags=tags)
                    
                for values, tags in special_rows:
                    item = self.table.insert("", "end", values=values)
                    self.table.item(item, tags=tags)
                
                # Update column headers to show sort direction
                for col in self.table["columns"]:
                    if col == column_id:
                        direction = "↑" if ascending else "↓"
                        self.table.heading(col, text=f"{col} {direction}")
                    else:
                        # Remove sort indicator from other columns
                        current_text = self.table.heading(col, "text")
                        clean_text = current_text.split(" ")[0] if " " in current_text else current_text
                        self.table.heading(col, text=clean_text)
        except Exception as e:
            self.logger.error(f"Error sorting table: {str(e)}")

    def update_theme(self, theme):
        """Update the table theme colors."""
        try:
            # Check if widgets still exist
            if not hasattr(self, "table_frame") or not self.table_frame.winfo_exists():
                return
                
            self.theme = theme
            
            # Update frame colors
            self.table_frame.configure(fg_color=self.theme.get("bg", "#F0F4F8"))
            self.header_frame.configure(fg_color=self.theme.get("bg", "#F0F4F8"))
            self.content_frame.configure(fg_color=self.theme.get("bg", "#F0F4F8"))
            
            if hasattr(self, "table") and self.table.winfo_exists():
                # Update row colors - use white for all rows
                self.table.tag_configure('hover', background=self.theme.get("table_hover", "#E2E8F0"))
                self.table.tag_configure('oddrow', background="#FFFFFF")  # Plain white for all rows
                self.table.tag_configure('evenrow', background="#FFFFFF") # Plain white for all rows
                
                # Add tags for special message rows
                self.table.tag_configure('info', background="#E6F2FF", foreground="#1E40AF")
                self.table.tag_configure('success', background="#E6F9E6", foreground="#166534")
                self.table.tag_configure('warning', background="#FFF5E6", foreground="#9A3412")
                self.table.tag_configure('error', background="#FFE6E6", foreground="#BE123C")
                # Make result rows more prominent with a highlighted background
                self.table.tag_configure('result', background="#6247AA", foreground="#FFFFFF", font=('Helvetica', 10, 'bold'))
                
                # Update table style while preserving row height
                style = ttk.Style()
                current_rowheight = style.lookup("Custom.Treeview", "rowheight")
                
                style.configure("Custom.Treeview",
                              background=self.theme.get("table_bg", "#FFFFFF"),  # Use plain white background
                              foreground=self.theme.get("table_fg", "#1E293B"),
                              fieldbackground=self.theme.get("table_bg", "#FFFFFF"),  # Use plain white background
                              rowheight=current_rowheight or 28)  # Maintain current row height or use default
                
                style.configure("Custom.Treeview.Heading",
                              background=self.theme.get("table_heading_bg", "#E2E8F0"),
                              foreground=self.theme.get("table_heading_fg", "#1E293B"))
                
                # Reapply row colors to existing rows
                for idx, item in enumerate(self.table.get_children()):
                    # Get current tags
                    tags = list(self.table.item(item, "tags"))
                    
                    # Filter out row coloring tags
                    tags = [tag for tag in tags if tag not in ('oddrow', 'evenrow')]
                    
                    # Add appropriate row tag
                    if "info" in tags or "warning" in tags or "error" in tags or "success" in tags or "result" in tags:
                        # Preserve special row tags
                        pass
                    else:
                        # Add row coloring - plain white for all
                        tags.append('evenrow')
                    
                    # Update tags
                    self.table.item(item, tags=tags)
            
        except Exception as e:
            self.logger.error(f"Error updating table theme: {str(e)}")

    def _format_value(self, value, decimal_places=4):
        """Format a value for display."""
        try:
            # Check for special types
            if isinstance(value, np.ndarray):
                return self._format_matrix(value, decimal_places)
                
            if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                return self._format_vector(value, decimal_places)
                
            # Check for numeric types
            if isinstance(value, (int, float)):
                if math.isnan(value):
                    return "NaN"
                elif math.isinf(value):
                    return "Inf" if value > 0 else "-Inf"
                elif value == int(value):
                    return str(int(value))
                else:
                    return f"{value:.{decimal_places}f}"
                    
            # Default: convert to string
            return str(value)
        except Exception as e:
            self.logger.error(f"Error formatting value: {str(e)}")
            return str(value)

    def _format_matrix(self, matrix: np.ndarray, decimal_places: int) -> str:
        """Format a matrix for display in a cell."""
        try:
            if matrix.ndim == 2:
                rows = []
                for row in matrix:
                    formatted_row = '[ '  # Added space after bracket
                    row_values = []
                    for x in row:
                        if isinstance(x, (int, float)):
                            if math.isnan(x):
                                row_values.append("NaN")
                            elif math.isinf(x):
                                row_values.append("Inf" if x > 0 else "-Inf")
                            else:
                                row_values.append(f"{x:.{decimal_places}f}")
                        else:
                            row_values.append(str(x))
                    formatted_row += ' ,  '.join(row_values) + ' ]'  # Added more spacing
                    rows.append(formatted_row)
                return '[\n  ' + '\n  '.join(rows) + '\n]'  # Return with newlines for better readability
            else:
                return self._format_vector(matrix, decimal_places)
        except Exception as e:
            self.logger.error(f"Error formatting matrix: {str(e)}")
            return str(matrix)

    def _format_vector(self, vector: Union[np.ndarray, List], decimal_places: int) -> str:
        """Format a vector for display in a cell."""
        try:
            formatted_values = []
            for x in vector:
                if isinstance(x, (int, float)):
                    if math.isnan(x):
                        formatted_values.append("NaN")
                    elif math.isinf(x):
                        formatted_values.append("Inf" if x > 0 else "-Inf")
                    else:
                        formatted_values.append(f"{x:.{decimal_places}f}")
                else:
                    formatted_values.append(str(x))
            return '[ ' + ' ,  '.join(formatted_values) + ' ]'  # Added more spacing
        except Exception as e:
            self.logger.error(f"Error formatting vector: {str(e)}")
            return str(vector)

    def display(self, data):
        """
        Display data in the table.
        
        Args:
            data: List of dictionaries or Pandas DataFrame
        """
        try:
            # Clear existing data
            self.clear()
            
            # Handle different data types
            if isinstance(data, pd.DataFrame):
                df_data = data
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # For OrderedDict or regular dict, preserve the order of columns
                # from the first row, which is crucial for Fixed Point and other methods
                df_data = pd.DataFrame(data)
                # Preserve the original order of columns if it was an OrderedDict
                if isinstance(data[0], OrderedDict):
                    df_data = df_data[list(data[0].keys())]
            elif isinstance(data, dict):
                df_data = pd.DataFrame([data])
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                # Display error message
                self.table["columns"] = ["Error"]
                self.table.column("Error", width=400, anchor="center")
                self.table.heading("Error", text="Error")
                self.table.insert("", "end", values=["Unsupported data format"], tags=("error",))
                return

            # Filter out rows that have NaN values in important columns
            if isinstance(df_data, pd.DataFrame) and len(df_data) > 0:
                # Don't filter special message rows (like errors, warnings, etc.)
                message_rows = df_data[df_data["Iteration"].astype(str).isin(["Error", "Warning", "Info", "Result", "Success"])] if "Iteration" in df_data.columns else pd.DataFrame()
                
                # Filter data rows to remove NaN rows
                data_rows = df_data[~df_data["Iteration"].astype(str).isin(["Error", "Warning", "Info", "Result", "Success"])] if "Iteration" in df_data.columns else df_data
                data_rows = data_rows.dropna(how='all')
                
                # Remove any row that has "NaN" as string in the Iteration column
                if "Iteration" in data_rows.columns:
                    data_rows = data_rows[~data_rows["Iteration"].astype(str).str.contains("NaN", na=False)]
                
                # Combine filtered rows with message rows
                df_data = pd.concat([data_rows, message_rows]).reset_index(drop=True)
            
            # Configure columns
            column_ids = list(df_data.columns)
            
            # Remove any special styling columns that shouldn't be displayed
            if 'highlight' in column_ids:
                column_ids.remove('highlight')
                
            self.table["columns"] = column_ids
            
            # Set column widths and headings - use more compact widths
            for col in column_ids:
                # Default width - reduced from 150 to 120
                width = 120
                
                # Adjust width based on column type
                col_str = str(col)
                header_width = len(col_str) * 8  # Reduced multiplier from 10 to 8
                
                # Use more compact widths for specific columns
                if col in ["xi", "g(xi)"]:
                    width = 180  # Reduced from 250
                elif col in ["Error %"]:
                    width = 90   # Reduced from 120
                elif col in ["Iteration"]:
                    width = 70   # Reduced from 100
                elif col in ["f(Xl)", "f(Xu)", "f(Xr)"]:
                    width = 80   # More compact for function values
                elif col in ["Xl", "Xu", "Xr"]:
                    width = 80   # More compact for x values
                elif col in ["Status"]:
                    width = 100  # Reduced for status
                
                # Adjust width based on content length, but use tighter constraints
                col_values = df_data[col].astype(str)
                if not col_values.empty:
                    # Check for content length with a more compact multiplier
                    max_content_width = col_values.str.len().max() * 6  # Reduced from 8 to 6
                    width = max(width, header_width, min(max_content_width, 300))  # Reduced max from 400 to 300
                else:
                    width = max(width, header_width)
                
                self.table.column(col, width=width, minwidth=70, anchor="center")  # Reduced minwidth from 100 to 70
                self.table.heading(col, text=col_str, anchor="center")
            
            # Check if we need to use different row heights for matrix displays
            contains_matrix = False
            for col in column_ids:
                # Check a sample of values from the column
                if len(df_data) > 0:
                    sample_values = df_data[col].astype(str).sample(min(5, len(df_data)))
                    for val in sample_values:
                        if isinstance(val, str) and '\n' in val:
                            contains_matrix = True
                            break
                    if contains_matrix:
                        break
            
            # If matrix data is detected, configure a larger row height
            if contains_matrix:
                style = ttk.Style()
                style.configure("Custom.Treeview", rowheight=90)  # Increased from 70 to 90 for matrices
            
            # Iterate through each row
            for idx, row in df_data.iterrows():
                values = []
                
                # Format each value
                for col in column_ids:
                    value = row[col]
                    values.append(self._format_value(value))
                
                # Determine row tags
                tags = []
                
                # Apply alternating row colors
                if idx % 2 == 0:
                    tags.append('oddrow')
                else:
                    tags.append('evenrow')
                
                # Check for special row types
                if 'Step' in row:
                    if row['Step'] == 'Warning' or str(row['Step']).startswith('Warning'):
                        tags.append('warning')
                    elif row['Step'] == 'Error' or str(row['Step']).startswith('Error'):
                        tags.append('error')
                    elif row['Step'] == 'Solution' or str(row['Step']).startswith('Solution'):
                        tags.append('result')
                    elif row['Step'] == 'Solution Summary':
                        tags.append('result')
                    elif row['Step'] == 'Info' or str(row['Step']).startswith('Info'):
                        tags.append('info')
                    elif row['Step'] == 'Success' or str(row['Step']).startswith('Success'):
                        tags.append('success')
                elif 'Iteration' in row:
                    if row['Iteration'] == 'Warning' or str(row['Iteration']).startswith('Warning'):
                        tags.append('warning')
                    elif row['Iteration'] == 'Error' or str(row['Iteration']).startswith('Error'):
                        tags.append('error')
                    elif row['Iteration'] == 'Solution' or str(row['Iteration']).startswith('Solution'):
                        tags.append('result')
                    elif row['Iteration'] == 'Solution Summary':
                        tags.append('result')
                    elif row['Iteration'] == 'Info' or str(row['Iteration']).startswith('Info'):
                        tags.append('info')
                    elif row['Iteration'] == 'Success' or str(row['Iteration']).startswith('Success'):
                        tags.append('success')
                        
                # Apply any custom highlight from the data
                if 'highlight' in row:
                    if row['highlight'] == 'warning':
                        tags.append('warning')
                    elif row['highlight'] == 'error':
                        tags.append('error')
                    elif row['highlight'] == 'result':
                        tags.append('result')
                    elif row['highlight'] == 'info':
                        tags.append('info')
                    elif row['highlight'] == 'success':
                        tags.append('success')
                    elif row['highlight'] == 'alternate':
                        tags = ['evenrow' if 'oddrow' in tags else 'oddrow']
                    elif row['highlight'] is True:  # Check for Boolean True value
                        tags.append('result')
                
                # Special check for Result rows to ensure they're always highlighted
                if 'Iteration' in row and row['Iteration'] == 'Result':
                    if 'result' not in tags:
                        tags.append('result')
                
                # Insert the row with values and tags
                self.table.insert("", "end", values=values, tags=tags)
                
            # Show horizontal scrollbar if needed
            table_width = sum(int(self.table.column(col, "width")) for col in column_ids)
            frame_width = self.table_frame.winfo_width()
            
            if table_width > frame_width:
                self.scrollbar_x.pack(side="bottom", fill="x")
            else:
                self.scrollbar_x.pack_forget()
                
        except Exception as e:
            self.logger.error(f"Error displaying data: {str(e)}")
            # Display error message
            self.clear()
            self.table["columns"] = ["Error"]
            self.table.column("Error", width=400, anchor="center")
            self.table.heading("Error", text="Error")
            self.table.insert("", "end", values=[f"Error displaying data: {str(e)}"], tags=("error",))

    def display_history(self, history):
        """
        Display history entries in the table.
        
        Args:
            history: List of history entries
        """
        try:
            # Clear existing data
            self.clear()
            
            if not history or len(history) == 0:
                # Display message for empty history
                self.table["columns"] = ["Info"]
                self.table.column("Info", width=400, anchor="center")
                self.table.heading("Info", text="Information")
                self.table.insert("", "end", values=["No history entries found"], tags=("info",))
                return
            
            # Configure columns
            columns = ["Index", "Date", "Time", "Method", "Function", "Root", "Tags"]
            self.table["columns"] = columns
            
            # Set column widths and headings
            col_widths = {
                "Index": 60,
                "Date": 100,
                "Time": 100,
                "Method": 150,
                "Function": 250,
                "Root": 150,
                "Tags": 150
            }
            
            for col in columns:
                width = col_widths.get(col, 150)
                self.table.column(col, width=width, minwidth=60, anchor="center")
                self.table.heading(col, text=col, anchor="center")
            
            # Insert history entries
            for idx, entry in enumerate(history):
                # Format root value(s)
                root = entry.get("root", "")
                if isinstance(root, list):
                    root_str = ", ".join([str(r) for r in root if r is not None])
                else:
                    root_str = str(root)
                
                # Format tags
                tags = entry.get("tags", [])
                tags_str = ", ".join(tags) if tags else ""
                
                # Prepare row values
                values = [
                    idx,
                    entry.get("date", ""),
                    entry.get("time", ""),
                    entry.get("method", ""),
                    entry.get("function", ""),
                    root_str,
                    tags_str
                ]
                
                # Insert with alternating row colors
                tag = "evenrow" if idx % 2 == 0 else "oddrow"
                self.table.insert("", "end", values=values, tags=(tag,))
                
            # Show horizontal scrollbar if needed
            table_width = sum(int(self.table.column(col, "width")) for col in columns)
            frame_width = self.table_frame.winfo_width()
            
            if table_width > frame_width:
                self.scrollbar_x.pack(side="bottom", fill="x")
            else:
                self.scrollbar_x.pack_forget()
                
        except Exception as e:
            self.logger.error(f"Error displaying history: {str(e)}")
            # Display error message
            self.clear()
            self.table["columns"] = ["Error"]
            self.table.column("Error", width=400, anchor="center")
            self.table.heading("Error", text="Error")
            self.table.insert("", "end", values=[f"Error displaying history: {str(e)}"], tags=("error",))

    def clear(self):
        """Clear the table."""
        try:
            # Check if the table exists
            if hasattr(self, "table") and self.table.winfo_exists():
                # Delete all items
                for item in self.table.get_children():
                    self.table.delete(item)
                    
                # Reset columns
                self.table["columns"] = []
                
                # Reset sort state
                self.sort_columns = {}
        except Exception as e:
            self.logger.error(f"Error clearing table: {str(e)}")
            
    def get_selected_row(self):
        """
        Get the selected row data.
        
        Returns:
            Dict with column names and values
        """
        try:
            selected_items = self.table.selection()
            if not selected_items:
                return None
                
            # Get the first selected item
            item = selected_items[0]
            
            # Get values and column names
            values = self.table.item(item, "values")
            columns = self.table["columns"]
            
            # Create a dictionary
            return {columns[i]: values[i] for i in range(min(len(columns), len(values)))}
            
        except Exception as e:
            self.logger.error(f"Error getting selected row: {str(e)}")
            return None
            
    def export_to_csv(self, filename: str):
        """
        Export table data to CSV.
        
        Args:
            filename: CSV file path
            
        Returns:
            bool: True if successful
        """
        try:
            # Get column names
            columns = self.table["columns"]
            if not columns:
                return False
                
            # Get all rows
            rows = []
            for item in self.table.get_children():
                values = self.table.item(item, "values")
                row_dict = {columns[i]: values[i] for i in range(min(len(columns), len(values)))}
                rows.append(row_dict)
                
            # Create DataFrame and save to CSV
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            return False
