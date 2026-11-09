
import numpy as np


class RingAttractor:
    """Simplified, biologically inspired neural compass."""

    def __init__(self, n=16):
        self.n = n
        self.angles = np.linspace(
            0, 2 * np.pi, n, endpoint=False
        )

        # Nearby neurons excite; distant neurons inhibit.
        distance = (
            self.angles[:, None] - self.angles[None, :]
        )

        self.weights = (
            1.6 * np.exp(2 * (np.cos(distance) - 1))
            - 0.45
        ) / n

        self.activity = np.exp(
            3 * (np.cos(self.angles) - 1)
        )

    def step(self, turn_degrees=0, visual_heading=None):
        """Update the network by one simulation step."""

        # Turning-related input shifts the heading pattern.
        shifted_angles = (
            self.angles - np.deg2rad(turn_degrees)
        )

        shifted_activity = np.interp(
            shifted_angles % (2 * np.pi),
            np.append(self.angles, 2 * np.pi),
            np.append(self.activity, self.activity[0])
        )

        recurrent_input = self.weights @ shifted_activity

        # Optional visual landmark correction.
        visual_input = np.zeros(self.n)

        if visual_heading is not None:
            target = np.deg2rad(visual_heading)
            visual_input = 0.8 * np.exp(
                3 * (np.cos(self.angles - target) - 1)
            )

        # Simplified rate-based neural dynamics.
        drive = (
            0.6 * shifted_activity
            + recurrent_input
            + visual_input
        )

        new_activity = np.maximum(0, drive)

        # Keep activity within a stable numerical range.
        if new_activity.max() > 0:
            new_activity /= new_activity.max()

        self.activity = (
            0.7 * shifted_activity
            + 0.3 * new_activity
        )

        return self.activity.copy()

    def estimated_heading(self):
        """Decode heading from the neural activity."""
        vector = np.sum(
            self.activity * np.exp(1j * self.angles)
        )
        return np.rad2deg(np.angle(vector)) % 360
