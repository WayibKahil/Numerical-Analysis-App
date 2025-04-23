import json
import os
import datetime
from typing import List, Dict, Any, Optional, Union
import logging
from collections import defaultdict

class HistoryManager:
    def __init__(self, file_path: str = "history.json"):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
        
        # Create history file if it doesn't exist
        if not os.path.exists(self.file_path):
            self._save_empty_history()

    def _save_empty_history(self) -> None:
        """Create an empty history file."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
        except IOError as e:
            self.logger.error(f"Failed to create empty history file: {str(e)}")
            raise

    def _validate_solution_data(self, func: str, method: str, root: Union[float, List[float]], table: List[Dict[str, Any]]) -> bool:
        """Validate the solution data before saving."""
        if not isinstance(func, str) or not func:
            return False
            
        if not isinstance(method, str) or not method:
            return False
            
        if not isinstance(table, list):
            return False
            
        return True

    def save_solution(self, func: str, method: str, root: Union[float, List[float]], table: List[Dict[str, Any]], 
                     params: Dict[str, Any] = None, tags: List[str] = None) -> bool:
        """
        Save a solution to the history file.
        
        Args:
            func: Function string
            method: Numerical method name
            root: Root value(s)
            table: Iteration table data
            params: Optional parameters
            tags: Optional list of tags
            
        Returns:
            bool: True if saving was successful
        """
        if not self._validate_solution_data(func, method, root, table):
            self.logger.error(f"Failed to save solution: Invalid solution data")
            return False
            
        try:
            # Load existing history
            history = self.load_history()
            
            # Ensure all table entries are dictionaries
            validated_table = []
            for row in table:
                if isinstance(row, dict):
                    validated_table.append(row)
                else:
                    # Skip non-dictionary rows or convert to a simple message dict
                    self.logger.warning(f"Skipping non-dictionary row: {row}")
            
            # Create solution entry
            current_time = datetime.datetime.now()
            solution = {
                "function": func,
                "method": method,
                "root": root,
                "iterations": validated_table,
                "parameters": params or {},
                "timestamp": current_time.isoformat(),
                "date": current_time.strftime("%Y-%m-%d"),
                "time": current_time.strftime("%H:%M:%S"),
                "tags": tags or []
            }
            
            # Add to history
            history.append(solution)
            
            # Save updated history
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to save solution: {str(e)}")
            return False

    def load_history(self) -> List[Dict[str, Any]]:
        """Load the solution history."""
        try:
            if not os.path.exists(self.file_path):
                return []
                
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load history: {str(e)}")
            return []

    def clear_history(self) -> bool:
        """Clear the solution history."""
        try:
            self._save_empty_history()
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear history: {str(e)}")
            return False

    def get_solution(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific solution by index."""
        history = self.load_history()
        
        if 0 <= index < len(history):
            return history[index]
        else:
            return None

    def delete_solution(self, index: int) -> bool:
        """Delete a specific solution by index."""
        try:
            history = self.load_history()
            
            if 0 <= index < len(history):
                del history[index]
                
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                    
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Failed to delete solution: {str(e)}")
            return False
            
    def search_history(self, query: str = None, method: str = None, 
                      date_from: str = None, date_to: str = None,
                      tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search history with various filters.
        
        Args:
            query: Text to search in function or tag
            method: Specific method name
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            tags: List of tags to match
            
        Returns:
            Filtered list of history entries
        """
        try:
            history = self.load_history()
            results = []
            
            # Parse dates if provided
            if date_from:
                try:
                    date_from_obj = datetime.datetime.strptime(date_from, "%Y-%m-%d").date()
                except ValueError:
                    self.logger.warning(f"Invalid date_from format: {date_from}")
                    date_from_obj = None
            else:
                date_from_obj = None
                
            if date_to:
                try:
                    date_to_obj = datetime.datetime.strptime(date_to, "%Y-%m-%d").date()
                except ValueError:
                    self.logger.warning(f"Invalid date_to format: {date_to}")
                    date_to_obj = None
            else:
                date_to_obj = None
            
            for entry in history:
                # Check if entry matches all criteria
                matches = True
                
                # Match query in function or tags
                if query:
                    query_lower = query.lower()
                    function_match = query_lower in entry.get("function", "").lower()
                    tag_match = any(query_lower in tag.lower() for tag in entry.get("tags", []))
                    
                    if not (function_match or tag_match):
                        matches = False
                
                # Match method
                if method and method != entry.get("method", ""):
                    matches = False
                
                # Match date range
                if date_from_obj or date_to_obj:
                    try:
                        entry_date = datetime.datetime.strptime(entry.get("date", "1970-01-01"), "%Y-%m-%d").date()
                        
                        if date_from_obj and entry_date < date_from_obj:
                            matches = False
                            
                        if date_to_obj and entry_date > date_to_obj:
                            matches = False
                    except ValueError:
                        # Skip entries with invalid dates
                        matches = False
                
                # Match tags
                if tags:
                    entry_tags = set(entry.get("tags", []))
                    if not all(tag in entry_tags for tag in tags):
                        matches = False
                
                # Add to results if matches
                if matches:
                    results.append(entry)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching history: {str(e)}")
            return []
            
    def get_history_by_date(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group history entries by date.
        
        Returns:
            Dictionary with dates as keys and lists of entries as values
        """
        try:
            history = self.load_history()
            grouped = defaultdict(list)
            
            for entry in history:
                date = entry.get("date", "Unknown")
                grouped[date].append(entry)
            
            return dict(grouped)
            
        except Exception as e:
            self.logger.error(f"Error grouping history by date: {str(e)}")
            return {}
            
    def get_history_by_method(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group history entries by method.
        
        Returns:
            Dictionary with methods as keys and lists of entries as values
        """
        try:
            history = self.load_history()
            grouped = defaultdict(list)
            
            for entry in history:
                method = entry.get("method", "Unknown")
                grouped[method].append(entry)
            
            return dict(grouped)
            
        except Exception as e:
            self.logger.error(f"Error grouping history by method: {str(e)}")
            return {}
            
    def add_tag_to_solution(self, index: int, tag: str) -> bool:
        """
        Add a tag to a specific solution.
        
        Args:
            index: Solution index
            tag: Tag to add
            
        Returns:
            bool: True if successful
        """
        try:
            history = self.load_history()
            
            if 0 <= index < len(history):
                # Get current tags or initialize empty list
                tags = history[index].get("tags", [])
                
                # Add tag if not already present
                if tag not in tags:
                    tags.append(tag)
                    history[index]["tags"] = tags
                    
                    # Save updated history
                    with open(self.file_path, 'w', encoding='utf-8') as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                        
                    return True
                return True  # Tag already exists, still successful
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding tag to solution: {str(e)}")
            return False
            
    def remove_tag_from_solution(self, index: int, tag: str) -> bool:
        """
        Remove a tag from a specific solution.
        
        Args:
            index: Solution index
            tag: Tag to remove
            
        Returns:
            bool: True if successful
        """
        try:
            history = self.load_history()
            
            if 0 <= index < len(history):
                # Get current tags
                tags = history[index].get("tags", [])
                
                # Remove tag if present
                if tag in tags:
                    tags.remove(tag)
                    history[index]["tags"] = tags
                    
                    # Save updated history
                    with open(self.file_path, 'w', encoding='utf-8') as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                        
                    return True
                return True  # Tag doesn't exist, still successful
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing tag from solution: {str(e)}")
            return False
            
    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags used in the history.
        
        Returns:
            List of unique tags
        """
        try:
            history = self.load_history()
            all_tags = set()
            
            for entry in history:
                tags = entry.get("tags", [])
                all_tags.update(tags)
            
            return sorted(list(all_tags))
            
        except Exception as e:
            self.logger.error(f"Error getting all tags: {str(e)}")
            return []