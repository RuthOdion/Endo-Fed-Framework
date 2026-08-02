"""
Dataset loading classes for endometriosis imaging data.

Supports GLENDA v1.5 laparoscopic images and UT Endo MRI datasets,
with combined loading and synthetic data generation for testing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from src.data.preprocessing import preprocess_image, extract_mri_slices


class GLENDADataset:
    """
    GLENDA v1.5 Dataset Loader.

    Loads laparoscopic images from pathological and non-pathological
    directories for binary endometriosis classification.

    Directory structure expected:
        GLENDA_v1.5/
            pathological/
                img_001.png
                ...
            non_pathological/
                img_001.png
                ...
            masks/  (optional)
                img_001.png
                ...
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        pathological_dir: str = "pathological",
        non_pathological_dir: str = "non_pathological",
        masks_dir: str = "masks",
    ):
        """
        Initialise GLENDA dataset loader.

        Args:
            data_dir: Root directory of the GLENDA dataset.
            image_size: Target image dimensions (height, width).
            pathological_dir: Subdirectory name for pathological images.
            non_pathological_dir: Subdirectory name for non-pathological images.
            masks_dir: Subdirectory name for segmentation masks.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.pathological_dir = pathological_dir
        self.non_pathological_dir = non_pathological_dir
        self.masks_dir = masks_dir

    def _load_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.

        Args:
            image_path: Full path to the image file.

        Returns:
            Preprocessed image array or None if loading fails.
        """
        try:
            if cv2 is not None:
                image = cv2.imread(image_path)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif Image is not None:
                pil_img = Image.open(image_path).convert("RGB")
                image = np.array(pil_img)
            else:
                return None

            image = preprocess_image(image, self.image_size)
            return image

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the complete GLENDA dataset.

        Returns:
            Tuple of (images, labels):
                - images: numpy array of shape (N, H, W, 3)
                - labels: numpy array of shape (N,) with 0=non_pathological, 1=pathological
        """
        images = []
        labels = []

        # Load non-pathological images (label 0)
        non_path_dir = self.data_dir / self.non_pathological_dir
        if non_path_dir.exists():
            for img_file in sorted(non_path_dir.iterdir()):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                    img = self._load_single_image(str(img_file))
                    if img is not None:
                        images.append(img)
                        labels.append(0)

        # Load pathological images (label 1)
        path_dir = self.data_dir / self.pathological_dir
        if path_dir.exists():
            for img_file in sorted(path_dir.iterdir()):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                    img = self._load_single_image(str(img_file))
                    if img is not None:
                        images.append(img)
                        labels.append(1)

        if len(images) == 0:
            return np.array([]), np.array([])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    def load_masks(self) -> Dict[str, np.ndarray]:
        """
        Load segmentation masks for explainability evaluation.

        Returns:
            Dictionary mapping filename to mask array.
        """
        masks = {}
        masks_dir = self.data_dir / self.masks_dir

        if not masks_dir.exists():
            return masks

        for mask_file in sorted(masks_dir.iterdir()):
            if mask_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff"]:
                try:
                    if cv2 is not None:
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    elif Image is not None:
                        mask = np.array(Image.open(str(mask_file)).convert("L"))
                    else:
                        continue

                    if mask is not None:
                        # Binarise mask
                        mask = (mask > 127).astype(np.float32)
                        masks[mask_file.stem] = mask
                except Exception as e:
                    print(f"Error loading mask {mask_file}: {e}")

        return masks


class UTEndoMRIDataset:
    """
    UT Endo MRI Dataset Loader.

    Loads MRI volumes (NIfTI or DICOM format) for endometriosis
    classification from pelvic MRI scans.

    Directory structure expected:
        UT_Endo_MRI/
            volumes/
                subject_001.nii.gz
                ...
            labels.csv
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        volumes_dir: str = "volumes",
        labels_file: str = "labels.csv",
        slice_axis: int = 2,
    ):
        """
        Initialise UT Endo MRI dataset loader.

        Args:
            data_dir: Root directory of the UT Endo MRI dataset.
            image_size: Target image dimensions (height, width).
            volumes_dir: Subdirectory name for MRI volumes.
            labels_file: Filename for subject labels CSV.
            slice_axis: Axis along which to extract slices (default: axial=2).
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.volumes_dir = volumes_dir
        self.labels_file = labels_file
        self.slice_axis = slice_axis

    def _load_volume(self, volume_path: str) -> Optional[np.ndarray]:
        """
        Load a 3D MRI volume from NIfTI or DICOM format.

        Args:
            volume_path: Path to the volume file.

        Returns:
            3D numpy array or None if loading fails.
        """
        try:
            path = Path(volume_path)

            if path.suffix in [".nii", ".gz"] or path.name.endswith(".nii.gz"):
                if nib is not None:
                    img = nib.load(str(path))
                    volume = img.get_fdata()
                elif sitk is not None:
                    img = sitk.ReadImage(str(path))
                    volume = sitk.GetArrayFromImage(img)
                else:
                    return None
            elif path.suffix == ".dcm":
                if pydicom is not None:
                    ds = pydicom.dcmread(str(path))
                    volume = ds.pixel_array.astype(np.float32)
                elif sitk is not None:
                    img = sitk.ReadImage(str(path))
                    volume = sitk.GetArrayFromImage(img)
                else:
                    return None
            else:
                if sitk is not None:
                    img = sitk.ReadImage(str(path))
                    volume = sitk.GetArrayFromImage(img)
                else:
                    return None

            return volume.astype(np.float32)

        except Exception as e:
            print(f"Error loading volume {volume_path}: {e}")
            return None

    def _get_subject_label(self, subject_id: str, labels_df) -> Optional[int]:
        """
        Get the label for a subject from the labels DataFrame.

        Args:
            subject_id: Subject identifier.
            labels_df: Pandas DataFrame with subject labels.

        Returns:
            Integer label (0 or 1) or None if not found.
        """
        try:
            row = labels_df[labels_df["subject_id"] == subject_id]
            if len(row) > 0:
                return int(row.iloc[0]["label"])
            return None
        except (KeyError, IndexError):
            return None

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the complete UT Endo MRI dataset.

        Extracts 2D slices from 3D volumes and assigns subject-level labels.

        Returns:
            Tuple of (images, labels):
                - images: numpy array of shape (N, H, W, 3)
                - labels: numpy array of shape (N,) with 0 or 1
        """
        try:
            import pandas as pd
        except ImportError:
            print("pandas required for loading UT Endo MRI labels.")
            return np.array([]), np.array([])

        images = []
        labels = []

        labels_path = self.data_dir / self.labels_file
        if not labels_path.exists():
            print(f"Labels file not found: {labels_path}")
            return np.array([]), np.array([])

        labels_df = pd.read_csv(labels_path)
        volumes_dir = self.data_dir / self.volumes_dir

        if not volumes_dir.exists():
            print(f"Volumes directory not found: {volumes_dir}")
            return np.array([]), np.array([])

        for vol_file in sorted(volumes_dir.iterdir()):
            subject_id = vol_file.stem.replace(".nii", "")
            label = self._get_subject_label(subject_id, labels_df)

            if label is None:
                continue

            volume = self._load_volume(str(vol_file))
            if volume is None:
                continue

            # Extract 2D slices from the volume
            slices = extract_mri_slices(volume, axis=self.slice_axis)

            for s in slices:
                # Convert single-channel to 3-channel
                if len(s.shape) == 2:
                    s = np.stack([s] * 3, axis=-1)

                img = preprocess_image(s, self.image_size)
                images.append(img)
                labels.append(label)

        if len(images) == 0:
            return np.array([]), np.array([])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


class CombinedDatasetLoader:
    """
    Combined dataset loader that merges GLENDA and UT Endo MRI datasets.

    Provides a unified interface for loading all available endometriosis
    imaging data for the federated learning experiments.
    """

    def __init__(self, config: Dict):
        """
        Initialise combined dataset loader.

        Args:
            config: Configuration dictionary with dataset paths and parameters.
        """
        self.config = config
        self.image_size = (
            config.get("data", {}).get("image_size", 224),
            config.get("data", {}).get("image_size", 224),
        )

    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and combine all available datasets.

        Returns:
            Tuple of (images, labels) combining all datasets.
        """
        all_images = []
        all_labels = []

        # Load GLENDA dataset
        glenda_config = self.config.get("data", {}).get("datasets", {}).get("glenda", {})
        if glenda_config.get("path"):
            glenda = GLENDADataset(
                data_dir=glenda_config["path"],
                image_size=self.image_size,
                pathological_dir=glenda_config.get("pathological_dir", "pathological"),
                non_pathological_dir=glenda_config.get("non_pathological_dir", "non_pathological"),
            )
            images, labels = glenda.load_dataset()
            if len(images) > 0:
                all_images.append(images)
                all_labels.append(labels)

        # Load UT Endo MRI dataset
        mri_config = self.config.get("data", {}).get("datasets", {}).get("ut_endo_mri", {})
        if mri_config.get("path"):
            mri_dataset = UTEndoMRIDataset(
                data_dir=mri_config["path"],
                image_size=self.image_size,
                volumes_dir=mri_config.get("volumes_dir", "volumes"),
                labels_file=mri_config.get("labels_file", "labels.csv"),
            )
            images, labels = mri_dataset.load_dataset()
            if len(images) > 0:
                all_images.append(images)
                all_labels.append(labels)

        if len(all_images) == 0:
            return np.array([]), np.array([])

        combined_images = np.concatenate(all_images, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        return combined_images, combined_labels

    @staticmethod
    def create_synthetic_dataset(
        num_samples: int = 100,
        image_size: Tuple[int, int] = (224, 224),
        num_channels: int = 3,
        class_ratio: float = 0.5,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a synthetic dataset for integration testing.

        Generates random images with controlled properties to verify
        the training pipeline without requiring real medical data.

        Args:
            num_samples: Total number of samples to generate.
            image_size: Image dimensions (height, width).
            num_channels: Number of colour channels.
            class_ratio: Proportion of positive (pathological) samples.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (images, labels) with synthetic data.
        """
        rng = np.random.RandomState(seed)

        num_positive = int(num_samples * class_ratio)
        num_negative = num_samples - num_positive

        # Generate synthetic images with different statistical properties per class
        # Negative class: lower mean intensity
        neg_images = rng.normal(
            loc=0.3, scale=0.15, size=(num_negative, image_size[0], image_size[1], num_channels)
        ).astype(np.float32)

        # Positive class: higher mean intensity with texture patterns
        pos_images = rng.normal(
            loc=0.6, scale=0.2, size=(num_positive, image_size[0], image_size[1], num_channels)
        ).astype(np.float32)

        # Clip to valid range
        neg_images = np.clip(neg_images, 0.0, 1.0)
        pos_images = np.clip(pos_images, 0.0, 1.0)

        images = np.concatenate([neg_images, pos_images], axis=0)
        labels = np.array([0] * num_negative + [1] * num_positive, dtype=np.int32)

        # Shuffle
        indices = rng.permutation(num_samples)
        images = images[indices]
        labels = labels[indices]

        return images, labels
