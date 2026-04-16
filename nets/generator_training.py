"""Compatibility generators for old training notebooks.

The new implementation lives in `typhoon_rainfall.data.dataset`. These classes
preserve the old names `Generator_merge` and `Generator_merge_val` so older
notebook cells can still call `.generate()`.
"""

from typhoon_rainfall.data.dataset import MergedDataGenerator


class Generator_merge:
    """Legacy training split generator wrapper."""

    def __init__(self, batch_size, train_lines, image_size, num_classes, RD, IR, GI, RA, interval):
        from typhoon_rainfall.config import DataConfig
        from typhoon_rainfall.data.contracts import DatasetPaths, InputModes, SamplePair

        del image_size
        pairs = []
        for line in train_lines:
            source_name, target_name = line.strip().split(";")
            pairs.append(SamplePair(source_name=source_name, target_name=target_name))
        self.generator = MergedDataGenerator(
            batch_size=batch_size,
            pairs=pairs,
            dataset_paths=DatasetPaths(root=DataConfig().dataset_root, interval=interval),
            split="train",
            input_modes=InputModes(use_rd=RD, use_ir=IR, use_ra=RA, use_gi=GI),
            num_classes=num_classes,
            shuffle_each_epoch=True,
        )

    def generate(self):
        """Return an infinite Keras-compatible batch iterator."""
        return self.generator.iter_batches()


class Generator_merge_val:
    """Legacy validation split generator wrapper."""

    def __init__(self, batch_size, train_lines, image_size, num_classes, RD, IR, GI, RA, interval):
        from typhoon_rainfall.config import DataConfig
        from typhoon_rainfall.data.contracts import DatasetPaths, InputModes, SamplePair

        del image_size
        pairs = []
        for line in train_lines:
            source_name, target_name = line.strip().split(";")
            pairs.append(SamplePair(source_name=source_name, target_name=target_name))
        self.generator = MergedDataGenerator(
            batch_size=batch_size,
            pairs=pairs,
            dataset_paths=DatasetPaths(root=DataConfig().dataset_root, interval=interval),
            split="val",
            input_modes=InputModes(use_rd=RD, use_ir=IR, use_ra=RA, use_gi=GI),
            num_classes=num_classes,
            shuffle_each_epoch=False,
        )

    def generate(self):
        """Return an infinite validation batch iterator."""
        return self.generator.iter_batches()
