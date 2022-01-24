from . import feature

# Deze zocht ik niet meer te commenten.
class Timeline:
    t0: float
    id_seed: int
    features: list[tuple[int, float, feature.Feature]]

    def __init__(self):
        self.t0 = 0
        self.id_seed = 1
        self.features = []

    def add_feature(self, t: float, feat: feature.Feature):
        if len(self.features) == 0:
            self.t0 = t

        fid = self.id_seed
        self.id_seed += 1

        self.features.append((fid, t-self.t0, feat))

    def find_features(self, t0: float, t1: float) -> list[tuple[int, float, feature.Feature]]:
        res = []

        for fid, ft, f in self.features:
            if ft < t0:
                continue
            if ft > t1:
                break

            res.append((fid, ft, f))

        return res

    def remove_features(self, ids: set[int]):
        self.features = [(fid, t, f) for fid, t, f in self.features if fid not in ids]
