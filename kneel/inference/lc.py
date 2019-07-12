from ._utils import LandmarkAnnotator


class GlobalSearcher(LandmarkAnnotator):
    def __init__(self, snapshot_path, mean_std_path):
        super(GlobalSearcher, self).__init__(snapshot_path, mean_std_path)
        print(self.models)