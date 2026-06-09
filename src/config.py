import os
import yaml
from pathlib import Path
from typing import Any, Dict, List

class Config:
    """Config loader for TerraSense AI platform."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default location: project_root/config/config.yaml
            project_root = Path(__file__).resolve().parent.parent
            config_path = str(project_root / "config" / "config.yaml")
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(self.config_path, "r") as f:
            self._config_dict = yaml.safe_load(f)
            
        self._project_root = self.config_path.parent.parent
        self._initialize_paths()
        
    def _initialize_paths(self) -> None:
        """Resolves relative paths and creates necessary directories."""
        self.paths = {}
        for key, val in self._config_dict.get("paths", {}).items():
            # If the path is relative, make it absolute relative to project root
            p = Path(val)
            if not p.is_absolute():
                p = self._project_root / p
            self.paths[key] = str(p)
            
            # Create directories for files if needed (except files themselves)
            if key == "log_file":
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)
                
    @property
    def spatial(self) -> Dict[str, Any]:
        return self._config_dict.get("spatial", {})
        
    @property
    def targets(self) -> List[Dict[str, Any]]:
        return self._config_dict.get("targets", [])
        
    @property
    def features(self) -> Dict[str, List[str]]:
        return self._config_dict.get("features", {})
        
    @property
    def training(self) -> Dict[str, Any]:
        return self._config_dict.get("training", {})
        
    @property
    def agronomy(self) -> Dict[str, Any]:
        return self._config_dict.get("agronomy", {})
        
    @property
    def fertilizers(self) -> Dict[str, Any]:
        return self._config_dict.get("fertilizers", {})
        
    @property
    def soil_conditioners(self) -> Dict[str, Any]:
        return self._config_dict.get("soil_conditioners", {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._config_dict.get(key, default)
