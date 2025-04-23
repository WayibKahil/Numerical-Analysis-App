import json
import os
from typing import Dict, Any, Optional

class PreferencesManager:
    """Manages user preferences."""
    
    # Default preferences
    DEFAULT_PREFERENCES = {
        "theme": "light",
        "decimal_places": 6,
        "max_iterations": 50,
        "epsilon": 0.0001,
        "stop_by_epsilon": True,
        "window_size": {
            "width": 1000,
            "height": 700
        }
    }
    
    def __init__(self, file_path: str = "user_preferences.json"):
        """Initialize the preferences manager."""
        self.file_path = file_path
        self.preferences = self.load_preferences()
        
    def load_preferences(self) -> Dict[str, Any]:
        """Load preferences from file or create with defaults."""
        try:
            if not os.path.exists(self.file_path):
                self.save_preferences(self.DEFAULT_PREFERENCES)
                return self.DEFAULT_PREFERENCES.copy()
                
            with open(self.file_path, 'r', encoding='utf-8') as f:
                loaded_prefs = json.load(f)
            
            # Update with any missing defaults
            for key, value in self.DEFAULT_PREFERENCES.items():
                if key not in loaded_prefs:
                    loaded_prefs[key] = value
                    
            return loaded_prefs
        except Exception as e:
            print(f"Error loading preferences: {str(e)}")
            return self.DEFAULT_PREFERENCES.copy()
    
    def save_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Save preferences to file."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(preferences, f, ensure_ascii=False, indent=2)
            self.preferences = preferences
            return True
        except Exception as e:
            print(f"Error saving preferences: {str(e)}")
            return False
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a specific preference value."""
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value: Any) -> bool:
        """Set a specific preference value."""
        try:
            self.preferences[key] = value
            return self.save_preferences(self.preferences)
        except Exception as e:
            print(f"Error setting preference: {str(e)}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset all preferences to default values."""
        return self.save_preferences(self.DEFAULT_PREFERENCES.copy())