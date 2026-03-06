"""
Workaround pour le problème de compatibilité PyTorch/Streamlit
Empêche Streamlit d'inspecter torch.classes qui cause une RuntimeError
"""

def apply_pytorch_fix():
    """
    Applique un workaround pour empêcher Streamlit d'inspecter torch.classes
    Cela résout l'erreur: RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
    """
    try:
        import torch
        # Empêcher Streamlit d'inspecter torch.classes en créant un mock __path__
        if hasattr(torch, 'classes'):
            if not hasattr(torch.classes, '__path__'):
                class MockPath:
                    """Mock __path__ pour torch.classes"""
                    def __iter__(self):
                        return iter([])
                    
                    def __contains__(self, item):
                        return False
                    
                    def __getitem__(self, item):
                        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
                
                torch.classes.__path__ = MockPath()
    except (ImportError, AttributeError, TypeError):
        # PyTorch n'est pas installé ou pas de problème
        pass

# Appliquer le fix automatiquement lors de l'import
apply_pytorch_fix()










