"""
Workaround for PyTorch/Streamlit compatibility issue
Prevents Streamlit from inspecting torch.classes which triggers a RuntimeError
"""

def apply_pytorch_fix():
    """
    Apply a workaround to prevent Streamlit from inspecting torch.classes
    This fixes the error: RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
    """
    try:
        import torch
        # Prevent Streamlit from inspecting torch.classes by creating a mock __path__
        if hasattr(torch, 'classes'):
            if not hasattr(torch.classes, '__path__'):
                class MockPath:
                    """Mock __path__ for torch.classes"""
                    def __iter__(self):
                        return iter([])
                    
                    def __contains__(self, item):
                        return False
                    
                    def __getitem__(self, item):
                        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
                
                torch.classes.__path__ = MockPath()
    except (ImportError, AttributeError, TypeError):
        # PyTorch not installed or no issue detected
        pass

# Appliquer le fix automatiquement lors de l'import
apply_pytorch_fix()










