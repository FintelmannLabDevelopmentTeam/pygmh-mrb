
import numpy as np

from pygmh_mrb.adapter import Adapter
from pygmh_mrb_test.assets import asset_path


def test_read():

    adapter = Adapter()

    image = adapter.read(
        asset_path("simple/126145_CTChest.mrb")
    )

    assert np.array_equal(
        image.get_image_data()[60:70],
        np.load(asset_path("simple/image_data.part.npy"))
    )

    for identifier in ["lymph node", "Segment_1"]:

        segmentation = image.get_segment(identifier)

        assert np.array_equal(
            segmentation.get_mask()[60:70],
            np.load(asset_path(f"simple/{identifier}.part.npy"))
        )
