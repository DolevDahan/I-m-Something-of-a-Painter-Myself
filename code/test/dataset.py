# one_folder_test_dataset.py
import os, torch.utils.data as data
from PIL import Image
import torchvision.transforms as T

# ---- כלי עזר קטנים --------------------------------------------------------
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
def is_image_file(fname):           # בדיקה אם קובץ-תמונה
    return fname.lower().endswith(_IMG_EXT)

def make_dataset(dir, max_size=float("inf")):
    paths = []
    for root, _, files in os.walk(dir):
        for f in files:
            if is_image_file(f):
                paths.append(os.path.join(root, f))
    return sorted(paths)[: int(max_size)]

# ---- טרנספורמציות בסיס ----------------------------------------------------
def _make_power_2(img, base=4, method=Image.BICUBIC):
    ow, oh = img.size
    h, w = int(round(oh / base) * base), int(round(ow / base) * base)
    return img if (h == oh and w == ow) else img.resize((w, h), method)

def get_transform(opt, method=Image.BICUBIC):
    tf = []
    if "resize" in opt.preprocess:
        tf.append(T.Resize([opt.load_size, opt.load_size], method))
    tf.append(T.Lambda(lambda im: _make_power_2(im, 4, method)))
    if not opt.no_flip:
        tf.append(T.RandomHorizontalFlip())
    tf += [T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)]
    return T.Compose(tf)

# ---- Dataset חד-כיווני  ---------------------------------------------------
class OneFolderTestDataset(data.Dataset):
    """
    עובד עם תיקייה אחת בלבד (dir_A).
    מחזיר:
        dict( A=..., B=..., A_paths=..., B_paths=... )
    שם B הוא עותק של A – רק כדי לשמור על מבנה הקלט של המודל.
    """
    def __init__(self, dir_A, opt):
        assert os.path.isdir(dir_A), f"{dir_A} is not a directory"
        self.opt      = opt
        self.A_paths  = make_dataset(dir_A, opt.max_dataset_size)
        self.transform = get_transform(opt)

    def __len__(self):
        return len(self.A_paths)

    def __getitem__(self, idx):
        path = self.A_paths[idx]
        img  = Image.open(path).convert("RGB")
        A    = self.transform(img)
        return {
            "A": A,
            "B": A,            # דמי – לא ישמש ב-test
            "A_paths": path,
            "B_paths": path,
        }
