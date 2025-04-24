from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from typing import List, Dict, Any, Optional
import os
import re
from datetime import datetime

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to make it safe for all operating systems.
    """
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized

def format_table_data(table_data: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Format table data for PDF export with proper handling of different data types,
    especially for matrix data.
    """
    if not table_data:
        return [["Message"], ["No data available"]]
        
    if "Error" in table_data[0]:
        return [["Message"], [table_data[0].get("Error", "Unknown error")]]
        
    # Get headers from the first row
    headers = list(table_data[0].keys())
    
    # Format each row
    formatted_data = [headers]
    for row in table_data:
        formatted_row = []
        for header in headers:
            value = row.get(header, "")
            
            # Special handling for matrix data
            if header == "Matrix" and isinstance(value, str):
                # Preserve line breaks in matrix representations
                # Replace multiple spaces with non-breaking spaces to maintain alignment
                formatted_value = value.replace("  ", " \xa0")
                formatted_row.append(formatted_value)
            # Convert various types to string representation
            elif isinstance(value, float):
                # Format floats with consistent decimal places
                formatted_row.append(f"{value:.6f}")
            elif isinstance(value, int):
                formatted_row.append(str(value))
            elif isinstance(value, dict):
                formatted_row.append(str(value))
            else:
                formatted_row.append(str(value))
                
        formatted_data.append(formatted_row)
    
    return formatted_data

def export_to_pdf(filename: str, func: str, method: str, root: Any, table_data: List[Dict[str, Any]]) -> bool:
    """
    Export solution to a PDF file.
    """
    try:
        # Sanitize filename
        safe_filename = sanitize_filename(filename)
        if not safe_filename.endswith(".pdf"):
            safe_filename += ".pdf"
            
        # Create PDF document
        doc = SimpleDocTemplate(safe_filename, pagesize=letter)
        elements = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading_style = styles["Heading2"]
        normal_style = styles["Normal"]
        
        # Add title
        elements.append(Paragraph(f"Numerical Analysis Report: {method}", title_style))
        elements.append(Spacer(1, 0.25 * inch))
        
        # Add date
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated on: {date_str}", normal_style))
        elements.append(Spacer(1, 0.25 * inch))
        
        # Add function information
        elements.append(Paragraph("Function Information", heading_style))
        elements.append(Paragraph(f"Function: {func}", normal_style))
        elements.append(Paragraph(f"Method: {method}", normal_style))
        
        # Add root information
        elements.append(Spacer(1, 0.25 * inch))
        elements.append(Paragraph("Solution", heading_style))
        
        # Check if this is a matrix method by examining the function name or method name
        is_matrix_method = func == "System of Linear Equations" or any(matrix_method in method for matrix_method in [
            "Gauss", "Jordan", "LU", "Cramer", "Linear System"
        ])
        
        if isinstance(root, list):
            # For linear system methods
            if is_matrix_method:
                # Format matrix solution with proper spacing and alignment
                elements.append(Paragraph("Solution Vector:", normal_style))
                solution_items = []
                for i, val in enumerate(root):
                    # Format each variable with proper precision
                    if isinstance(val, (int, float)):
                        formatted_val = f"{val:.6f}" if isinstance(val, float) else str(val)
                        solution_items.append(f"x{i+1} = {formatted_val}")
                    else:
                        solution_items.append(f"x{i+1} = {val}")
                
                # Add each solution variable on its own line for better readability
                for item in solution_items:
                    elements.append(Paragraph(f"• {item}", normal_style))
            else:
                # For other list-type solutions, use the original comma-separated format
                root_text = ", ".join([f"x{i+1} = {val}" for i, val in enumerate(root)])
                elements.append(Paragraph(f"Solution: {root_text}", normal_style))
        elif root is not None:
            # For root-finding methods
            elements.append(Paragraph(f"Root: {root}", normal_style))
        else:
            # No solution found
            elements.append(Paragraph("No solution found", normal_style))
        
        # Add iteration table
        elements.append(Spacer(1, 0.25 * inch))
        elements.append(Paragraph("Iteration Details", heading_style))
        
        # Format table data
        formatted_data = format_table_data(table_data)
        
        # Create table
        if formatted_data:
            # Adjust column widths for better display of matrix data
            col_widths = None
            if is_matrix_method and len(formatted_data) > 0 and len(formatted_data[0]) > 0:
                # Make the Matrix column wider for matrix methods
                col_widths = [1.0 * inch]  # Step column
                for i in range(1, len(formatted_data[0])):
                    if i == 1 and formatted_data[0][i] == "Matrix":  # Matrix column
                        col_widths.append(3.5 * inch)  # Make Matrix column wider
                    else:
                        col_widths.append(2.0 * inch)  # Other columns
            
            # Create table with adjusted column widths if needed
            table = Table(formatted_data, colWidths=col_widths)
            
            # Style the table
            style = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")
            ])
            
            # Add special handling for matrix column
            if is_matrix_method:
                style.add("ALIGN", (1, 1), (1, -1), "LEFT")  # Left-align matrix column content
                style.add("LEFTPADDING", (1, 1), (1, -1), 10)  # Add left padding to matrix column
            
            table.setStyle(style)
            elements.append(table)
        
        # Build PDF
        doc.build(elements)
        return True
        
    except Exception as e:
        print(f"Error exporting to PDF: {str(e)}")
        return False