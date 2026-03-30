import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CMFDDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=256):
        """
        image_dir: path to images
        mask_dir: path to masks (must exist)
        """

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        valid_ext = [".jpg", ".png", ".tif"]

        self.images = []

        # Sort for reproducibility
        for f in sorted(os.listdir(image_dir)):

            ext = os.path.splitext(f)[1].lower()
            if ext not in valid_ext:
                continue

            if "_F" not in f:
                continue

            base = f.split("_F")[0]
            mask_name = base + "_M" + ext
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                self.images.append(f)

        print(f"{image_dir} -> {len(self.images)} samples")

        # Simple transform (no augmentation)
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load corresponding mask
        ext = os.path.splitext(img_name)[1]
        base = img_name.split("_F")[0]
        mask_name = base + "_M" + ext
        mask_path = os.path.join(self.mask_dir, mask_name)

        mask = Image.open(mask_path).convert("L")
        mask = self.transform(mask)

        # Convert to binary {0,1}
        mask = (mask > 0.5).float()

        return image, mask
