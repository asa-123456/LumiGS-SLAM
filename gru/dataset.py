import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(
        (torch.ByteTensor(bytearray(img.tobytes())).numpy()  # type: ignore[attr-defined]
    )).view(img.size[1], img.size[0], len(img.getbands()))  # H, W, C
    tensor = arr.float().permute(2, 0, 1) / 255.0  # C, H, W
    return tensor


class GRUDataset(Dataset):
    """
    Dataset that reads images and their target parameters from specified image and label directories.
    Supports low-light, normal-light, and over-exposed images.

    Expected file structure:
      - image_dir/: Contains image files (supports .JPG, .jpg, .png formats, etc.)
        Example: low_light_bike_0000.JPG, NeRF_360_bicycle_0003.JPG, over_exp_bike_0002.JPG
      - label_dir/: Contains corresponding JSON label files, supporting two formats:
        1. Combined format: low_light_bike_0000_affine_tone.json (contains affine_3x4 and tone_params)
        2. Separate format: NeRF_360_stump_0000_affine.json + NeRF_360_stump_0000_tone.json
    """


    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        max_samples: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        all_files = os.listdir(image_dir)
        image_extensions = {".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"}
        image_files = sorted([
            f for f in all_files 
            if os.path.splitext(f)[1].lower() in {ext.lower() for ext in image_extensions}
        ])

        if max_samples is not None:
            image_files = image_files[: max_samples]

        self.samples: List[Tuple[str, Optional[str], Optional[str]]] = []
        for img_file in image_files:
            stem = os.path.splitext(img_file)[0]
            
            label_file_combined = f"{stem}_affine_tone.json"
            label_path_combined = os.path.join(label_dir, label_file_combined)
            
            label_file_affine = f"{stem}_affine.json"
            label_file_tone = f"{stem}_tone.json"
            label_path_affine = os.path.join(label_dir, label_file_affine)
            label_path_tone = os.path.join(label_dir, label_file_tone)
            
            if os.path.exists(label_path_combined):
                self.samples.append((img_file, label_file_combined, None))
            elif os.path.exists(label_path_affine) and os.path.exists(label_path_tone):
                self.samples.append((img_file, label_file_affine, label_file_tone))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No samples found. Expected images in {image_dir} with matching label files in {label_dir}.\n"
                f"Supported label formats:\n"
                f"  1. Combined: *_affine_tone.json\n"
                f"  2. Separate: *_affine.json + *_tone.json"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _rgb_16x8_and_gray_16x1(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # rgb: C,H,W ∈ [0,1]
        rgb_small = torch.nn.functional.interpolate(
            rgb.unsqueeze(0), size=(16, 8), mode="bilinear", align_corners=False
        ).squeeze(0)
        gray_small = (
            0.299 * rgb_small[0] + 0.587 * rgb_small[1] + 0.114 * rgb_small[2]
        )  # (16,8)
        gray_16x1 = gray_small.mean(dim=1, keepdim=True)  # (16,1)
        return rgb_small, gray_16x1

    def _read_targets(
        self, 
        label_path_combined: Optional[str] = None,
        label_path_affine: Optional[str] = None,
        label_path_tone: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read label files, supporting both combined and separate formats.
        
        Args:
            label_path_combined: Combined format label file path (contains both affine and tone)
            label_path_affine: Separate format affine parameters file path
            label_path_tone: Separate format tone parameters file path
        """
        affine = None
        tone = None
        
        if label_path_combined is not None:
            # Combined format: read from single file
            json_data = _load_json(label_path_combined)
            
            # Process affine parameters
            if "affine_3x4" in json_data:
                matrix = json_data["affine_3x4"]
                affine_list = [item for row in matrix for item in row]  # Flatten row-wise
                affine = torch.tensor(affine_list, dtype=torch.float32)
            elif "affine_params" in json_data:
                affine = torch.tensor(json_data["affine_params"], dtype=torch.float32)
            else:
                raise ValueError(f"Could not find 'affine_3x4' or 'affine_params' key in {label_path_combined}")
            
            # Process tone parameters
            if "tone_params" in json_data:
                tone = torch.tensor(json_data["tone_params"], dtype=torch.float32)
            else:
                raise ValueError(f"Could not find 'tone_params' key in {label_path_combined}")
        
        elif label_path_affine is not None and label_path_tone is not None:
            # Separate format: read from two files
            json_affine = _load_json(label_path_affine)
            json_tone = _load_json(label_path_tone)
            
            # Process affine parameters
            if "affine_3x4" in json_affine:
                matrix = json_affine["affine_3x4"]
                affine_list = [item for row in matrix for item in row]  # Flatten row-wise
                affine = torch.tensor(affine_list, dtype=torch.float32)
            elif "affine_params" in json_affine:
                affine = torch.tensor(json_affine["affine_params"], dtype=torch.float32)
            else:
                raise ValueError(f"Could not find 'affine_3x4' or 'affine_params' key in {label_path_affine}")
            
            # Process tone parameters
            if "tone_params" in json_tone:
                tone = torch.tensor(json_tone["tone_params"], dtype=torch.float32)
            else:
                raise ValueError(f"Could not find 'tone_params' key in {label_path_tone}")
        else:
            raise ValueError("Must provide either combined format or separate format label file paths")
        
        # 验证参数维度
        if affine is None or affine.numel() != 12:
            raise ValueError(f"Affine params must have 12 values, got {affine.numel() if affine is not None else 0}")
        if tone is None or tone.numel() != 4:
            raise ValueError(f"Tone params must have 4 values, got {tone.numel() if tone is not None else 0}")
        
        return affine, tone

    def __getitem__(self, idx: int):
        img_file, label_file_combined, label_file_tone = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_file)

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            rgb = _to_tensor(img)  # C,H,W ∈ [0,1]

        if self.transform is not None:
            rgb = self.transform(rgb)

        rgb_16x8, gray_16x1 = self._rgb_16x8_and_gray_16x1(rgb)

        if label_file_tone is None:
            label_path_combined = os.path.join(self.label_dir, label_file_combined)
            affine_target, tone_target = self._read_targets(
                label_path_combined=label_path_combined
            )
        else:
            label_path_affine = os.path.join(self.label_dir, label_file_combined)
            label_path_tone = os.path.join(self.label_dir, label_file_tone)
            affine_target, tone_target = self._read_targets(
                label_path_affine=label_path_affine,
                label_path_tone=label_path_tone
            )

        sample = {
            "rgb_16x8": rgb_16x8,  # (3,16,8)
            "gray_16x1": gray_16x1,  # (16,1)
            "affine": affine_target,  # (12,)
            "tone": tone_target,  # (4,)
        }
        return sample


class GRUSequenceDataset(Dataset):
    """
    Sequence dataset: organizes images in order to support GRU recurrent neural networks.
    Utilizes the continuity of image sequences, allowing subsequent images to use information from previous images.
    
    Expected file structure:
      - image_dir/: Contains sequentially named image files
        Example: frame000049.jpg, frame000050.jpg, frame000051.jpg...
      - label_dir/: Contains corresponding JSON label files
    """
    
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        sequence_length: int = 1,
        stride: int = 1,
        max_samples: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            sequence_length: Sequence length, i.e., how many consecutive images to return each time
            stride: Sliding window step, default is 1 (continuous sequence)
        """

        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        all_files = os.listdir(image_dir)
        image_extensions = {".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"}
        image_files = sorted([
            f for f in all_files 
            if os.path.splitext(f)[1].lower() in {ext.lower() for ext in image_extensions}
        ])
        
        if max_samples is not None:
            image_files = image_files[: max_samples]
        
        self.sequences: List[List[Tuple[str, Optional[str], Optional[str]]]] = []
        
        valid_samples: List[Tuple[str, Optional[str], Optional[str]]] = []
        for img_file in image_files:
            stem = os.path.splitext(img_file)[0]
            
            label_file_combined = f"{stem}_affine_tone.json"
            label_path_combined = os.path.join(label_dir, label_file_combined)
            
            label_file_affine = f"{stem}_affine.json"
            label_file_tone = f"{stem}_tone.json"
            label_path_affine = os.path.join(label_dir, label_file_affine)
            label_path_tone = os.path.join(label_dir, label_file_tone)
            
            if os.path.exists(label_path_combined):
                valid_samples.append((img_file, label_file_combined, None))
            elif os.path.exists(label_path_affine) and os.path.exists(label_path_tone):
                valid_samples.append((img_file, label_file_affine, label_file_tone))
        
        if len(valid_samples) == 0:
            raise FileNotFoundError(
                f"No samples found. Expected images in {image_dir} with matching label files in {label_dir}."
            )
        
        for i in range(0, len(valid_samples) - sequence_length + 1, stride):
            sequence = valid_samples[i:i + sequence_length]
            self.sequences.append(sequence)
        
        if len(self.sequences) == 0:
            raise ValueError(
                f"Cannot create sequences: need at least {sequence_length} samples, "
                f"but only have {len(valid_samples)} valid samples."
            )
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    @staticmethod
    def _rgb_16x8_and_gray_16x1(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same method as GRUDataset"""
        rgb_small = torch.nn.functional.interpolate(
            rgb.unsqueeze(0), size=(16, 8), mode="bilinear", align_corners=False
        ).squeeze(0)
        gray_small = (
            0.299 * rgb_small[0] + 0.587 * rgb_small[1] + 0.114 * rgb_small[2]
        )
        gray_16x1 = gray_small.mean(dim=1, keepdim=True)
        return rgb_small, gray_16x1
    
    def _read_targets(
        self, 
        label_path_combined: Optional[str] = None,
        label_path_affine: Optional[str] = None,
        label_path_tone: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same method as GRUDataset"""
        affine = None
        tone = None
        
        if label_path_combined is not None:
            json_data = _load_json(label_path_combined)
            
            if "affine_3x4" in json_data:
                matrix = json_data["affine_3x4"]
                affine_list = [item for row in matrix for item in row]
                affine = torch.tensor(affine_list, dtype=torch.float32)
            elif "affine_params" in json_data:
                affine = torch.tensor(json_data["affine_params"], dtype=torch.float32)
            else:
                raise ValueError(f"Could not find 'affine_3x4' or 'affine_params' key in {label_path_combined}")
            
            if "tone_params" in json_data:
                tone = torch.tensor(json_data["tone_params"], dtype=torch.float32)
            else:
                raise ValueError(f"Could not find 'tone_params' key in {label_path_combined}")
        
        elif label_path_affine is not None and label_path_tone is not None:
            json_affine = _load_json(label_path_affine)
            json_tone = _load_json(label_path_tone)
            
            if "affine_3x4" in json_affine:
                matrix = json_affine["affine_3x4"]
                affine_list = [item for row in matrix for item in row]
                affine = torch.tensor(affine_list, dtype=torch.float32)
            elif "affine_params" in json_affine:
                affine = torch.tensor(json_affine["affine_params"], dtype=torch.float32)
            else:
                raise ValueError(f"Could not find 'affine_3x4' or 'affine_params' key in {label_path_affine}")
            
            if "tone_params" in json_tone:
                tone = torch.tensor(json_tone["tone_params"], dtype=torch.float32)
            else:
                raise ValueError(f"Could not find 'tone_params' key in {label_path_tone}")
        else:
            raise ValueError("Must provide either combined format or separate format label file paths")
        
        if affine is None or affine.numel() != 12:
            raise ValueError(f"Affine params must have 12 values, got {affine.numel() if affine is not None else 0}")
        if tone is None or tone.numel() != 4:
            raise ValueError(f"Tone params must have 4 values, got {tone.numel() if tone is not None else 0}")
        
        return affine, tone
    
    def __getitem__(self, idx: int):
        """
        Return a sequence of data
        
        Returns:
            sample: Dictionary containing sequence data
                - rgb_16x8: (sequence_length, 3, 16, 8)
                - gray_16x1: (sequence_length, 16, 1)
                - affine: (sequence_length, 12)
                - tone: (sequence_length, 4)
        """
        sequence = self.sequences[idx]
        
        rgb_16x8_list = []
        gray_16x1_list = []
        affine_list = []
        tone_list = []
        
        for img_file, label_file_combined, label_file_tone in sequence:
            # Read image
            img_path = os.path.join(self.image_dir, img_file)
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                rgb = _to_tensor(img)
            
            if self.transform is not None:
                rgb = self.transform(rgb)
            
            rgb_16x8, gray_16x1 = self._rgb_16x8_and_gray_16x1(rgb)
            rgb_16x8_list.append(rgb_16x8)
            gray_16x1_list.append(gray_16x1)
            
            # Read labels
            if label_file_tone is None:
                label_path_combined = os.path.join(self.label_dir, label_file_combined)
                affine_target, tone_target = self._read_targets(
                    label_path_combined=label_path_combined
                )
            else:
                label_path_affine = os.path.join(self.label_dir, label_file_combined)
                label_path_tone = os.path.join(self.label_dir, label_file_tone)
                affine_target, tone_target = self._read_targets(
                    label_path_affine=label_path_affine,
                    label_path_tone=label_path_tone
                )
            
            affine_list.append(affine_target)
            tone_list.append(tone_target)
        
        # Stack into sequence
        rgb_16x8_seq = torch.stack(rgb_16x8_list, dim=0)  # (seq_len, 3, 16, 8)
        gray_16x1_seq = torch.stack(gray_16x1_list, dim=0)  # (seq_len, 16, 1)
        affine_seq = torch.stack(affine_list, dim=0)  # (seq_len, 12)
        tone_seq = torch.stack(tone_list, dim=0)  # (seq_len, 4)
        
        sample = {
            "rgb_16x8": rgb_16x8_seq,
            "gray_16x1": gray_16x1_seq,
            "affine": affine_seq,
            "tone": tone_seq,
        }
        return sample


