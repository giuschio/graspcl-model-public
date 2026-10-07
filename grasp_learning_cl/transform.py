import numpy as np
import scipy.spatial.transform


class Rotation(scipy.spatial.transform.Rotation):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])


class Transform:
    """Rigid spatial transform between coordinate systems in 3D space."""

    def __init__(self, rotation, translation):
        assert isinstance(rotation, scipy.spatial.transform.Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (
                np.c_[self.rotation.as_matrix(), self.translation],
                [0.0, 0.0, 0.0, 1.0],
            )
        )

    def to_dict(self):
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_points(self, points):
        return points @ self.rotation.as_matrix().T + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, matrix):
        rotation = Rotation.from_matrix(matrix[:3, :3])
        translation = matrix[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_6dofs(cls, pose_6d, degrees=True):
        translation = pose_6d[:3]
        rotation = Rotation.from_euler("xyz", pose_6d[3:], degrees=degrees)
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, values):
        rotation = Rotation.from_quat(values[:4])
        translation = values[4:]
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        return cls(Rotation.identity(), np.zeros(3))

    @classmethod
    def look_at(cls, eye, center, up):
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        matrix = np.eye(4, 4)
        matrix[:3, 0] = right
        matrix[:3, 1] = -up
        matrix[:3, 2] = forward
        matrix[:3, 3] = eye

        return cls.from_matrix(matrix).inverse()

    def mirror(self, axis, origin=None):
        indices = {"x": 0, "y": 1, "z": 2}
        if axis not in indices:
            raise ValueError("Axis must be one of {'x', 'y', 'z'}")

        diagonal = np.ones(3, dtype=np.double)
        diagonal[indices[axis]] = -1.0
        reflection = np.diag(diagonal)
        rotation_matrix = reflection @ self.rotation.as_matrix() @ reflection
        origin = np.zeros(3, dtype=np.double) if origin is None else np.asarray(origin, np.double)
        translation = reflection @ (self.translation - origin) + origin

        return self.__class__(Rotation.from_matrix(rotation_matrix), translation)
