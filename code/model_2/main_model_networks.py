import torch
import networks
from patchNCE import PatchNCELoss
import os
# -------------------------------------------------------------
# 1)  בסיס מינימלי
# -------------------------------------------------------------
# ─────────────────────────────────────────────────────────────
# 1)  בסיס מינימלי
# -------------------------------------------------------------
import random, numpy as np           # ← הוספנו
import torch, networks, os
from patchNCE import PatchNCELoss

class BaseModel:
    """BaseModel מצומצם – רק מה שצריך לאימון."""
    def __init__(self, opt):
        # ── NEW: reproducibility ────────────────────────────
        if hasattr(opt, "seed") and opt.seed is not None:
            random.seed(opt.seed)
            np.random.seed(opt.seed)
            torch.manual_seed(opt.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(opt.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        # ----------------------------------------------------

        self.opt     = opt
        self.isTrain = bool(opt.isTrain)
        self.gpu_ids = opt.gpu_ids
        self.device  = torch.device(
            f"cuda:{self.gpu_ids[0]}" if (self.gpu_ids and torch.cuda.is_available()) else "cpu"
        )

        self.loss_names, self.visual_names, self.model_names = [], [], []
        self.optimizers, self.image_paths = [], []


    # קפיאת/שחרור גרדיאנטים
    @staticmethod
    def set_requires_grad(nets, flag=False):
        if not isinstance(nets, (list, tuple)):
            nets = [nets]
        for net in nets:
            for p in net.parameters():
                p.requires_grad = flag

    # לוגינג
    def get_current_losses(self):
        return {k: float(getattr(self, "loss_" + k)) for k in self.loss_names}

    def get_current_visuals(self):
        return {k: getattr(self, k) for k in self.visual_names}

# -------------------------------------------------------------
# 2)  MinimalCUTModel
# -------------------------------------------------------------
class MinimalCUTModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # לוגינג
        self.loss_names   = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers   = [int(i) for i in opt.nce_layers.split(',')]
        if opt.nce_idt and self.isTrain:
            self.loss_names.append('NCE_Y')
            self.visual_names.append('idt_B')

        # ---------- רשתות ----------
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf,
            opt.netG, opt.normG, not opt.no_dropout,
            opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up,
            self.gpu_ids, opt
        )
        self.netF = networks.define_F(
            opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
            opt.init_type, opt.init_gain, opt.no_antialias,
            self.gpu_ids, opt
        )

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                opt.normD, opt.init_type, opt.init_gain,
                opt.no_antialias, self.gpu_ids, opt
            )

            # קריטריונים
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = [PatchNCELoss(opt).to(self.device) for _ in self.nce_layers]

            # אופטימייזרים ל-G ו-D (F יוּצר מאוחר יותר)
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizers += [self.optimizer_G, self.optimizer_D]

            self.model_names = ['G', 'F', 'D']
        else:
            self.model_names = ['G']

    # ---------------------------------------------------------
    #  set_input → forward
    # ---------------------------------------------------------
    def set_input(self, data):
        AtoB = (self.opt.direction == 'AtoB')
        self.real_A      = data['A' if AtoB else 'B'].to(self.device)
        self.real_B      = data['B' if AtoB else 'A'].to(self.device)
        self.image_paths = data['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # הכנסת שני הדומיינים יחד אם nce_idt
        self.real = torch.cat((self.real_A, self.real_B), 0) \
                    if (self.opt.nce_idt and self.isTrain) else self.real_A

        self.flipped_for_equivariance = (
            self.isTrain and self.opt.flip_equivariance and np.random.rand() < 0.5
        )
        if self.flipped_for_equivariance:
            self.real = torch.flip(self.real, [3])

        self.fake   = self.netG(self.real)
        self.fake_B = self.fake[: self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    # ---------------------------------------------------------
    #  Losses
    # ---------------------------------------------------------
    def compute_D_loss(self):
        pred_fake       = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        pred_real       = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        return self.loss_D

    def calculate_NCE_loss(self, src, tgt):
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, self.nce_layers, encode_only=True)

        if self.flipped_for_equivariance:
            feat_q = [torch.flip(f, [3]) for f in feat_q]

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _          = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            total += crit(f_q, f_k).mean() * self.opt.lambda_NCE

        return total / len(self.nce_layers)

    def compute_G_loss(self):
        pred_fake        = self.netD(self.fake_B)
        self.loss_G_GAN  = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN

        self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B) \
                        if self.opt.lambda_NCE > 0 else torch.tensor(0.0, device=self.device)

        if self.opt.nce_idt:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            nce_total       = 0.5 * (self.loss_NCE + self.loss_NCE_Y)
        else:
            nce_total       = self.loss_NCE

        self.loss_G = self.loss_G_GAN + nce_total
        return self.loss_G

    # ---------------------------------------------------------
    #  Data-dependent init – יוצר optimizer_F אחרי forward ראשון
    # ---------------------------------------------------------
    def data_dependent_initialize(self, data):
        """מופעל על הבָּץ' הראשון – בונה את MLP-ים של netF ויוצר optimizer_F"""
        self.set_input(data)
        self.forward()                            # 1) pass ראשוני

        if self.isTrain:
            # 2) & 3) – מריצים איבודי־אובדן פעם אחת, מפעילים netF מבפנים
            self.compute_D_loss().backward()
            self.compute_G_loss().backward()

            # 4) – כעת ל-netF יש פרמטרים אמיתיים
            if self.opt.netF == 'mlp_sample' and not hasattr(self, 'optimizer_F'):
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(), lr=self.opt.lr,
                    betas=(self.opt.beta1, self.opt.beta2)
                )
                self.optimizers.append(self.optimizer_F)


    def setup(self, opt):
        """Setup: יצירת schedulers וטעינת משקלים אם צריך"""
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.print_networks(opt.verbose)



    def load_networks(self, epoch):
        """Load all the networks from disk (used for continue_train/test)."""
        for name in self.model_names:
            if isinstance(name, str):
                filename = f'{epoch}_net_{name}.pth'
                path = os.path.join(self.opt.checkpoints_dir, self.opt.name, filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print(f"loading the model from {path}")
                state_dict = torch.load(path, map_location=self.device)
                net.load_state_dict(state_dict)


    def print_networks(self, verbose):
        """Print total number of parameters in each network, and structure if verbose=True"""
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = sum(p.numel() for p in net.parameters())
                if verbose:
                    print(net)
                print(f"[Network {name}] Total number of parameters : {num_params / 1e6:.3f} M")
        print('-----------------------------------------------')

    def save_networks(self, epoch_label):
        """Save all the networks to disk."""
        save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(save_dir, exist_ok=True)  # ← הוספנו שורה זו

        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f'{epoch_label}_net_{name}.pth'
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                torch.save(net.state_dict(), save_path)

    def update_learning_rate(self):
        """Update learning rates (for linear lr_policy only)."""
        for scheduler in self.schedulers:
            scheduler.step()  # linear schedule doesn't need metrics
        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'learning rate = {lr:.7f}')

    def init_epoch_stats(self):
        """אתחול מצבר לאיבודים בתחילת epoch."""
        self.epoch_stats = {
            'G': 0.0,
            'D': 0.0,
            'GAN': 0.0,
            'NCE': 0.0,
            'samples': 0
        }

    def accumulate_epoch_stats(self, losses, batch_size):
        """צבירת איבודים של mini-batch בתוך מצבר epoch."""
        self.epoch_stats['samples'] += batch_size
        self.epoch_stats['G']   += losses['G'] * batch_size
        self.epoch_stats['GAN'] += losses['G_GAN'] * batch_size
        self.epoch_stats['D']   += 0.5 * (losses['D_real'] + losses['D_fake']) * batch_size

        if 'NCE_Y' in losses:
            self.epoch_stats['NCE'] += 0.5 * (losses['NCE'] + losses['NCE_Y']) * batch_size
        else:
            self.epoch_stats['NCE'] += losses['NCE'] * batch_size

    def print_epoch_stats(self, epoch):
        """הדפסת איבודים ממוצעים בסוף epoch."""
        n = max(1, self.epoch_stats['samples'])
        print(
            f'Epoch {epoch:03d}: '
            f'D={self.epoch_stats["D"]/n:.3f}  '
            f'G={self.epoch_stats["G"]/n:.3f}  '
            f'GAN={self.epoch_stats["GAN"]/n:.3f}  '
            f'NCE={self.epoch_stats["NCE"]/n:.3f}'
        )

    # ---------------------------------------------------------
    #  Optimize
    # ---------------------------------------------------------
    def optimize_parameters(self):
        self.forward()

        # --- עדכון D ---
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.compute_D_loss().backward()
        self.optimizer_D.step()

        # --- עדכון G (ו-F אם כבר קיים) ---
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if hasattr(self, 'optimizer_F'):
            self.optimizer_F.zero_grad()

        self.compute_G_loss().backward()
        self.optimizer_G.step()
        if hasattr(self, 'optimizer_F'):
            self.optimizer_F.step()

    @torch.no_grad()
    def test(self):
        """Forward pass בזמן בדיקה (ללא גרדיאנטים)"""
        self.forward()
        self.compute_visuals()      # כאן ריק – וזה מספיק

    # ---------- Visuals ----------
    def compute_visuals(self):
        """במקור מיועד לבנות ויזואליזציות; כאן אינו נדרש."""
        pass

    # ---------- Paths ----------
    def get_image_paths(self):
        """ה-DataLoader מעביר paths ב-set_input, כאן רק מחזירים אותם"""
        return self.image_paths

# -------------------------------------------------------------
# 3)  build_model – מחזיר מופע מוכן
# -------------------------------------------------------------
def build_model(opt):
    model = MinimalCUTModel(opt)
    print(f"model [{model.__class__.__name__}] was created")
    return model