"""
Vector Lane Map & Polynomial Splines
Converts discrete lane perception points into continuous, differentiable parametric splines:
  y(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3

Provides:
  - Analytical lateral offset: y(x)
  - Analytical heading tangent: psi(x) = arctan(y'(x))
  - Analytical road curvature: kappa(x) = |y''(x)| / (1 + y'(x)^2)^(3/2)
"""

import numpy as np

class VectorLane:
    def __init__(self, lane_id: str, coeffs: np.ndarray):
        """
        coeffs: [c0, c1, c2, c3] for y(x) = c0 + c1*x + c2*x^2 + c3*x^3
        where x is forward distance in meters, y is lateral distance.
        """
        self.id = lane_id
        self.coeffs = np.asarray(coeffs, dtype=np.float64)

    @classmethod
    def fit_from_points(cls, lane_id: str, x_points: np.ndarray, y_points: np.ndarray, order: int = 3):
        """Fits polynomial curve from discrete observed perception points."""
        poly = np.polyfit(x_points, y_points, deg=order)
        # polyfit returns highest degree first: [c3, c2, c1, c0] -> reverse to [c0, c1, c2, c3]
        return cls(lane_id, poly[::-1])

    def lateral_offset(self, x: float) -> float:
        """Evaluates lateral position y at forward distance x."""
        c0, c1, c2, c3 = self.coeffs
        return c0 + c1 * x + c2 * (x**2) + c3 * (x**3)

    def heading_tangent(self, x: float) -> float:
        """Computes tangent road heading angle (radians) at forward distance x."""
        _, c1, c2, c3 = self.coeffs
        # Derivative: y'(x) = c1 + 2*c2*x + 3*c3*x^2
        dy = c1 + 2.0 * c2 * x + 3.0 * c3 * (x**2)
        return np.arctan(dy)

    def curvature(self, x: float) -> float:
        """
        Computes analytical curvature kappa (1/meters) at forward distance x.
        kappa = |y''| / (1 + y'^2)^(3/2)
        """
        _, c1, c2, c3 = self.coeffs
        dy = c1 + 2.0 * c2 * x + 3.0 * c3 * (x**2)
        d2y = 2.0 * c2 + 6.0 * c3 * x
        
        kappa = abs(d2y) / np.power(1.0 + dy**2, 1.5)
        return float(kappa)

    def sample_polyline(self, x_start: float = 0.0, x_end: float = 40.0, step: float = 1.0) -> np.ndarray:
        """Generates (N, 2) polyline points [x, y] for plotting or trajectory search."""
        x_vals = np.arange(x_start, x_end + step, step)
        y_vals = np.array([self.lateral_offset(x) for x in x_vals])
        return np.stack([x_vals, y_vals], axis=-1)
