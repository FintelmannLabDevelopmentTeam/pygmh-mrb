
import io
import os
import zipfile
from typing import List, Optional, Tuple

import nrrd
import numpy as np


class MrbHandle:
    """Flat and read-only representation of a mrb-file."""

    def __init__(self, zip_file_handle: zipfile.ZipFile):

        self._zip_file_handle = zip_file_handle

    def read_mrml_file(self) -> str:

        # find mrml-file in archive
        mrml_member: Optional[zipfile.ZipInfo] = None
        for zip_info in self._zip_file_handle.infolist():

            zip_info: zipfile.ZipInfo = zip_info

            if zip_info.filename.endswith(".mrml"):
                assert mrml_member is None, "There are multiple *.mrml files in the mrb container."
                mrml_member = zip_info

        if mrml_member is None:
            raise Exception("Could not find mrml file in archive.")

        return self._zip_file_handle.read(mrml_member)

    def read_nrrd(self, relative_path: str) -> Tuple[np.ndarray, dict]:

        file_handle = io.BytesIO(
            self._zip_file_handle.read(
                self._resolve_relative_path(relative_path)
            )
        )

        nrrd_header = nrrd.read_header(file_handle)
        nrrd_data = nrrd.read_data(nrrd_header, file_handle)

        return nrrd_data, nrrd_header

    def find_image_data_member(self) -> str:

        candidates: List[str] = [
            member.filename
            for member in self._zip_file_handle.infolist()
            if member.filename.endswith(".nrrd") and not member.filename.endswith(".seg.nrrd")
        ]

        assert len(candidates) == 1, "Could not find image volume nrrd."

        return self._trim_absolute_path(candidates[0])

    def find_segmentation_data_member(self) -> str:

        candidates: List[str] = [
            member.filename
            for member in self._zip_file_handle.infolist()
            if member.filename.endswith(".seg.nrrd")
        ]

        assert len(candidates) == 1, "Could not find segmentation mask nrrd."

        return self._trim_absolute_path(candidates[0])

    def _trim_absolute_path(self, absolute_path: str) -> str:
        return absolute_path[len(self._get_common_member_path_prefix()):]

    def _resolve_relative_path(self, relative_path: str) -> str:
        return self._get_common_member_path_prefix() + relative_path

    def _get_common_member_path_prefix(self) -> str:
        """Returns the image-name, that is, the name of the top-most directory within the zip-archive."""
        return os.path.commonprefix(self._zip_file_handle.namelist())
