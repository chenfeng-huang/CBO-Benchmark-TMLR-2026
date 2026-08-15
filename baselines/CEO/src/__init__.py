import numpy as np

# Global NumPy compatibility shims for older dependencies (e.g., GPy uses np.bool)
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]

# Optional: uncomment if other deprecated aliases are ever needed
# if not hasattr(np, "int"):
#     np.int = int  # type: ignore[attr-defined]
# if not hasattr(np, "float"):
#     np.float = float  # type: ignore[attr-defined]
# if not hasattr(np, "object"):
#     np.object = object  # type: ignore[attr-defined]
# if not hasattr(np, "complex"):
#     np.complex = complex  # type: ignore[attr-defined]

