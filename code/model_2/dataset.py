# unaligned_dataset_standalone.py
import os, random, numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
from argparse import Namespace

# ─────────────────────────────────────────────────────────────
# 0)  כלי עזר בסיסיים
# ─────────────────────────────────────────────────────────────
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
def is_image_file(name):               # בדיקת סיומת-קובץ
    return name.lower().endswith(_IMG_EXT)

def make_dataset(dir, max_dataset_size=float("inf")):
    """אסף את כל קבצי-התמונה מתיקייה (רק נתיבים)"""
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), f"{dir} is not a valid directory"
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(os.path.join(root, fname))
    return images[: min(max_dataset_size, len(images))]

def copyconf(default_opt, **kwargs):
    """שכפול אובייקט ‎opt‎ עם override לשדות ספציפיים"""
    conf = Namespace(**vars(default_opt))
    for k, v in kwargs.items():
        setattr(conf, k, v)
    return conf

# ─────────────────────────────────────────────────────────────
# 1)  טרנספורמציות (הועתקו מ-BaseDataset, בגרסה מצומצמת)
# ─────────────────────────────────────────────────────────────
def _scale_width(img, target_w, crop_w, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_w and oh >= crop_w:
        return img
    h = int(max(target_w * oh / ow, crop_w))
    return img.resize((target_w, h), method)

def _make_power_2(img, base=4, method=Image.BICUBIC):
    ow, oh = img.size
    h, w = int(round(oh / base) * base), int(round(ow / base) * base)
    return img if (h == oh and w == ow) else img.resize((w, h), method)

def get_transform(opt, method=Image.BICUBIC):
    """רק מה שנדרש ל-CUT/FastCUT – resize→crop→flip→Tensor→Normalize"""
    transforms = []
    # resize
    if "resize" in opt.preprocess:
        size = [opt.load_size, opt.load_size]
        transforms.append(T.Resize(size, method))
    elif "scale_width" in opt.preprocess:
        transforms.append(T.Lambda(lambda img: _scale_width(img, opt.load_size,
                                                            opt.crop_size, method)))
    # crop
    if "crop" in opt.preprocess:
        transforms.append(T.RandomCrop(opt.crop_size))
    # adjust to power-of-2
    transforms.append(T.Lambda(lambda img: _make_power_2(img, base=4, method=method)))
    # flip
    if not opt.no_flip:
        transforms.append(T.RandomHorizontalFlip())
    # to tensor & normalization
    transforms += [T.ToTensor(),
                   T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return T.Compose(transforms)

# ─────────────────────────────────────────────────────────────
# 2)  הגדרה מחדש של UnalignedDataset (ללא ירושה)
# ─────────────────────────────────────────────────────────────
class UnalignedDataset(data.Dataset):
    """
    טעינת דאטה-סט לא מזווג (unaligned) לשימוש ב-CycleGAN/CUT.
    אינה יורשת מ-BaseDataset — כל התלותות הקריטיות משולבות כאן.
    """

    def __init__(self, opt):
        # ------------- שדות שה-trainer מצפה להם -------------
        self.opt           = opt
        self.root          = opt.dataroot
        self.current_epoch = 0          # ה-trainer יעדכן ערך זה כל epoch
        # -----------------------------------------------------
        phase_suffix = "" if opt.phase.endswith(("A", "B")) else opt.phase
        self.dir_A = os.path.join(opt.dataroot, f"{phase_suffix}A")
        self.dir_B = os.path.join(opt.dataroot, f"{phase_suffix}B")

        # Fallback ל-valA/valB בזמן test אם יש כאלה
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size  = len(self.A_paths)
        self.B_size  = len(self.B_paths)

        # מוכנים ל-transform ★
        self.base_transform = get_transform(opt)

    # ---------------------------------------------------------
    def __len__(self):
        """ CycleGAN/CUT לוקחים max(A,B) כדי לא להתקע כשהסטים לא שווים בגודל """
        return max(self.A_size, self.B_size)

    # ---------------------------------------------------------
    def __getitem__(self, index):
        """
        מחזיר מילון: {'A': tensor, 'B': tensor, 'A_paths': path, 'B_paths': path}
        """
        A_path = self.A_paths[index % self.A_size]

        # domain B – או סידורי או אקראי (כמו בקוד המקורי)
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # טעינת תמונות
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        # עדכון data-augmentation בשלב fine-tune (אם lr ב-decay)
        is_finetune = getattr(self.opt, "isTrain", False) and \
                      (self.current_epoch > self.opt.n_epochs)

        if is_finetune:   # בזמן fine-tune מוותלים resize-crop נוסף
            local_opt = copyconf(self.opt, load_size=self.opt.crop_size)
            transform = get_transform(local_opt)
        else:
            transform = self.base_transform

        return {
            "A":       transform(A_img),
            "B":       transform(B_img),
            "A_paths": A_path,
            "B_paths": B_path,
        }


class SimpleDataLoader:
    def __init__(self, dataloader, dataset, opt):
        self.dataloader = dataloader
        self.dataset    = dataset
        self.opt        = opt

    # מאפשר len(dataset)
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    # מאפשר for batch in dataset:
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    # אופציונלית – כמו במקור:
    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch