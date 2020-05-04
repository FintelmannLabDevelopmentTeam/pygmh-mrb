
import io
import logging
import os
import re
import urllib.parse
from typing import Tuple, Optional, cast, List
from xml.etree import ElementTree
import zipfile

import nrrd
import numpy as np

from pygmh.model import Coordinates3, Color, Image, Vector3
from pygmh.persistence.interface import IAdapter


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


class Adapter(IAdapter):

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def read(self, path: str) -> Image:

        self._logger.info("Reading in MRB file: " + path)

        assert os.path.isfile(path), "Given path is not a file: " + path

        with zipfile.ZipFile(path, "r") as zip_file_handle:

            mrb_handle = MrbHandle(zip_file_handle)

            mrml = ElementTree.fromstring(
                mrb_handle.read_mrml_file()
            )

            image = self._read_image(mrb_handle, mrml)

            self._read_segmentations(mrb_handle, mrml, image)

        return image

    def write(self, image: Image, path: str) -> None:

        raise NotImplementedError()

    def _read_image(self, mrb_handle: MrbHandle, mrml: ElementTree) -> Image:

        volume_node = mrml.find("./Volume")
        volume_storage_node = mrml.find("./VolumeArchetypeStorage[@id='{}']".format(
            volume_node.attrib["storageNodeRef"]
        ))

        nrrd_data, nrrd_header = mrb_handle.read_nrrd(
            urllib.parse.unquote(volume_storage_node.attrib["fileName"])
        )

        assert nrrd_header["space"] == "left-posterior-superior",\
            "Unsupported nrrd image volume space: " + nrrd_header["space"]
        assert [np.sign(x) for x in self._get_space_directions(nrrd_header["space directions"])] == [1., 1., 1.],\
            "Unsupported nrrd image volume directions."

        # type-cast
        nrrd_data = nrrd_data.astype(np.int32)

        # switch sagittal-coronal-axial axis-order to axial-coronal-sagittal
        nrrd_data = np.swapaxes(nrrd_data, 0, 2)

        # posterior towards anterior
        nrrd_data = np.flip(nrrd_data, 1)

        image = Image(
            nrrd_data,
            identifier=self._derive_identifier(volume_node.attrib["name"]),
            voxel_spacing=self._get_voxel_spacing(nrrd_header["space directions"])
        )

        return image

    def _get_voxel_spacing(self, space_directions: np.ndarray) -> Vector3:

        space_directions_diagonal = self._get_space_directions(space_directions)

        return cast(
            Vector3,
            [
                abs(x)
                for x in space_directions_diagonal
            ]
        )

    def _get_space_directions(self, space_directions: np.ndarray) -> Vector3:

        assert isinstance(space_directions, np.ndarray)
        assert space_directions.shape == (3, 3)

        return (
            space_directions[2][2],
            space_directions[1][1],
            space_directions[0][0]
        )

    def _read_segmentations(self, mrb_handle: MrbHandle, mrml: ElementTree, image: Image) -> None:
        """Reads in all segmentations and attaches them to the given image instance."""

        segmentation_nodes = mrml.findall("./Segmentation")
        assert len(segmentation_nodes) == 1,\
            "Expected a single <Segmentation>-node. Actual count: {}".format(len(segmentation_nodes))

        segmentation_node = segmentation_nodes[0]

        # Read 4D segmentation mask
        segmentations_volume, nrrd_header = self._get_segmentation_volume(mrb_handle, mrml, segmentation_node)

        # Read segmentation volume offset w.r.t. image volume origin
        offset = self._get_segmentation_mask_offset(nrrd_header, segmentations_volume, image)
        assert all([
            segmentations_volume.shape[1 + i] + offset[i] <= image.get_image_data().shape[i]
            for i in range(len(offset))
        ]), "Segmentation mask offset is inconsistent with mask size and image volume size."

        # Find segments
        subject_hierarchy_item_node = mrml.find("./SubjectHierarchy//SubjectHierarchyItem[@dataNode='{}']".format(
            segmentation_node.attrib["id"]
        ))
        segment_nodes = subject_hierarchy_item_node.findall(".//SubjectHierarchyItem[@type='Segments']")

        # Read in all mrb "segments" as segmentations
        for segment_index, segment_node in enumerate(segment_nodes):

            mask_partition = segmentations_volume[segment_index]

            # Build 3D mask from partition
            mask = np.zeros(image.get_image_data().shape, dtype=np.bool)
            mask[
                offset[0] : offset[0] + mask_partition.shape[0],
                offset[1] : offset[1] + mask_partition.shape[1],
                offset[2] : offset[2] + mask_partition.shape[2],
            ] = mask_partition

            header_prefix = "Segment{}_".format(segment_index)

            identifier = self._derive_identifier(nrrd_header[header_prefix + "Name"])
            if image.has_segmentation(identifier):
                identifier = self._handle_segmentation_identifier_collision(image, identifier)

            color = tuple([
                round(float(x) * 255)
                for x in
                nrrd_header[header_prefix + "Color"].split(" ")
            ])
            assert len(color) == 3, "Invalid segment color: " + nrrd_header[header_prefix + "Color"]
            color = cast(Color, color)

            image.add_segmentation(identifier, mask, color)

        assert [np.sum(segmentations_volume[x]) for x in range(segmentations_volume.shape[0])] ==\
               [np.sum(seg.get_mask()) for seg in image.get_ordered_segmentations()],\
            "Error during reconstruction of segmentation masks!"

    def _handle_segmentation_identifier_collision(self, image: Image, identifier: str) -> str:

        suffix = 0

        while True:

            candidate_identifier = "{}_{}".format(identifier, suffix)

            if not image.has_segmentation(candidate_identifier):
                return candidate_identifier

            suffix += 1

    def _get_segmentation_mask_offset(self, nrrd_header: dict, segmentations_volume: np.ndarray, image: Image) -> Coordinates3:

        offset = [
            int(s)
            for s in nrrd_header["Segmentation_ReferenceImageExtentOffset"].split()
        ]

        assert len(offset) == 3, "Invalid value in Segmentation_ReferenceImageExtentOffset."

        # ensure component order matches image coordinate system
        offset.reverse()
        offset[1] = image.get_image_data().shape[1] - (offset[1] + segmentations_volume.shape[2])

        return cast(Coordinates3, tuple(offset))

    def _get_segmentation_volume(self, mrb_handle: MrbHandle, mrml: ElementTree, segmentation_node: ElementTree) -> Tuple[np.ndarray, dict]:

        segmentation_storage_node = mrml.find("./SegmentationStorage[@id='{}']".format(
            segmentation_node.attrib["storageNodeRef"]
        ))

        nrrd_data, nrrd_header = mrb_handle.read_nrrd(
            segmentation_storage_node.attrib["fileName"]
        )

        assert len(np.shape(nrrd_data)) == 4, "Unexpected segmentation mask volume shape: {}".format(np.shape(nrrd_data))
        assert nrrd_header["space"] == "right-anterior-superior", "Unsupported segmentation mask nrrd space: " + nrrd_header["space"]
        assert [np.sign(x) for x in self._get_space_directions(nrrd_header["space directions"][1:])] == [1., -1., -1.]

        # switch sagittal-coronal-axial axis-order to axial-coronal-sagittal
        nrrd_data = np.swapaxes(nrrd_data, 1, 3)

        # posterior towards anterior
        nrrd_data = np.flip(nrrd_data, 2)

        return nrrd_data, nrrd_header

    def _derive_identifier(self, name: str) -> str:
        """Removes invalid characters from given name so that it can be used as an identifier."""
        return re.sub(r'[^A-Za-z0-9-_ ]', r'_', name)
