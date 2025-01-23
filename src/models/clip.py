from enum import Enum
import time
import torch
from torch import nn, optim
import lightning as L
import numpy as np
import utils.wandb as wandb
import logging
from lightning.pytorch.loggers import WandbLogger
import torch.nn.functional as F
from torch import nn
from models.projection_head import ProjectionHead
from models.bert import BertModel, BertConfig, BertTokenizer
from models.mednext import MedNeXt
from models.unet import UNet
from models.resunet import resunet
from models.swinunetr_ssl import SwinUNETR_SSL
import os
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from utils.globals import text_layers_names, vision_layers_names

def _build_vision(checkpoint_swin_path = "swinunet_checkpoints/last.ckpt", vision_type = "swinunetr", use_pretrain=True):
    if vision_type == "swinunetr":
        path = os.path.join(os.environ.get("STATES"), checkpoint_swin_path)
        vision = SwinUNETR_SSL.load_from_checkpoint(path) if use_pretrain else SwinUNETR_SSL()
        return vision
    if vision_type == "mednext":
        path = os.path.join(os.environ.get("STATES"), "mednext_s3.ckpt")
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        vision = MedNeXt(1, kernel_size=3)
        state_dict = {}
        for name, value in checkpoint['state_dict'].items():
            if 'encoder' in name:
                new_name = name[16:]
                state_dict[new_name] = value
        vision.load_state_dict(state_dict)
        return vision
    if vision_type == "unet":
        path = os.path.join(os.environ.get("STATES"), "unet.ckpt")
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        vision = UNet()
        state_dict = {}
        for name, value in checkpoint['state_dict'].items():
            if 'encoder' in name:
                new_name = name[16:]
                state_dict[new_name] = value
        vision.load_state_dict(state_dict)
        return vision
    if vision_type == "resunet":
        path = os.path.join(os.environ.get("STATES"), "resunet_smallpatch.ckpt")
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        vision = resunet()
        state_dict = {}
        for name, value in checkpoint['state_dict'].items():
            if 'encoder' in name:
                new_name = name[6:]
                state_dict[new_name] = value
        vision.load_state_dict(state_dict)
        return vision


def _build_text(out_dim = 512, use_pretrain = True):
    config = BertConfig(vocab_size_or_config_json_file=30522)
    bert = BertModel(config)
    if use_pretrain:
        bert.load_state_dict(torch.load(os.environ.get("STATES")+"/bert-base-uncased"))
    text = bert

    return text

def _build_tokenizer():
    tokenizer = BertTokenizer(os.environ.get("STATES")+"/bert-base-uncased-vocab.txt")
    return tokenizer


# define the LightningModule
class CLIP(L.LightningModule):
    def __init__(
            self, 
            version_dir=None,
            projection_dim = 512,
            learning_rate = 1e-4,
            base_weight_decay = 0.01,
            logit_scale_init = np.log(1 / 0.07),
            accum_freq = 1,
            lr_scheduler = "cosine",
            optimizer="adam",
            cosine_period_ratio: float = 1,
            warmup_steps = 0,
            steps_per_epoch = 100,
            text_lr = 0,
            vision_lr = 0,
            text_weight_decay=0.01,
            vision_weight_decay=0.01,
            beta1 = 0.9, 
            beta2 = 0.999, 
            vision_checkpoint = "swinunet_checkpoints/swinunetr.ckpt",
            vision_type = "swinunetr",
            enabel_image_logging = False,
            use_pretrain_vision = True,
            use_pretrain_text = True,
            config = {}):
        super().__init__()
        if config != {}:
            self.num_classes = config["num_classes"]
            self.num_modalities = config["num_modalities"]
            self.version_dir = config["version_dir"]
            self.plans = config["plans"]
            self.plans_path = config["plans_path"]
            self.model_name = config["model_name"]
            self.model_dimensions = config["model_dimensions"]
            self.patch_size = config["patch_size"]
            self.task_type = config["task_type"]
            self.sliding_window_prediction = config["patch_based_training"]
            self.batch_size = config['batch_size']
            self.val_size = config['val_size']
            self.split_method = config['split_method']
            self.split_idx = config['split_idx']
        self.learning_rate = learning_rate
        self.base_weight_decay = base_weight_decay
        self.warmup_steps = warmup_steps
        self.cosine_period_ratio = cosine_period_ratio
        self.projection_dim = projection_dim
        self.use_pretrain_vision = use_pretrain_vision
        self.use_pretrain_text = use_pretrain_text
        self.vision = _build_vision(vision_checkpoint, vision_type = vision_type, use_pretrain=self.use_pretrain_vision)
        self.text = _build_text(out_dim=self.projection_dim, use_pretrain = self.use_pretrain_text)
        self.tokenizer = _build_tokenizer()
        self.steps_per_epoch = steps_per_epoch
        self.beta1 = beta1
        self.beta2 = beta2
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

        if isinstance(self.vision, SwinUNETR_SSL):
            self.vision_projection = ProjectionHead(self.vision.enc_out_dim, self.projection_dim, 0.1)        
        elif isinstance(self.vision, MedNeXt):
            self.vision_projection = ProjectionHead(self.vision.dim, self.projection_dim, 0.1)
        elif isinstance(self.vision, UNet):
            self.vision_projection = ProjectionHead(self.vision.dim, self.projection_dim, 0.1)

        else:
            self.vision_projection = ProjectionHead(self.vision.dim, self.projection_dim, 0.1)
        self.enable_image_logging = enabel_image_logging
        self.reconstruction = True
        self.version_dir = version_dir
        self.accum_freq = accum_freq
        if self.accum_freq > 1:
            self.accum_images, self.accum_texts, self.accum_features = [], [], {}
        self.i = 0
        self.automatic_optimization = False
        self.lr_scheduler = lr_scheduler
        self.optimizer=optimizer
        self.text_lr = text_lr
        self.vision_lr = vision_lr
        self.text_weight_decay = text_weight_decay
        self.vision_weight_decay = vision_weight_decay
        self.train_diff = 0

        self.text_projection = ProjectionHead(self.text.config.hidden_size, self.projection_dim, 0.1) 
            
        # split params into groups for different lr
        if self.text_lr > 0:
            self.base_params = list(filter(lambda kv: kv[0] not in text_layers_names, self.named_parameters()))
            self.text_params = list(filter(lambda kv: kv[0] in text_layers_names, self.named_parameters()))
        if self.vision_lr > 0:
            self.base_params = list(filter(lambda kv: kv[0] not in vision_layers_names, self.named_parameters()))
            self.vision_params = list(filter(lambda kv: kv[0] in vision_layers_names, self.named_parameters()))
        if self.vision_lr > 0 and self.text_lr > 0:
            not_base_layers_names = vision_layers_names + text_layers_names
            self.base_params = list(filter(lambda kv: kv[0] not in not_base_layers_names, self.named_parameters()))
        self.save_hyperparameters(ignore=['text','vision']) 

    def training_step(self, batch, batch_idx): 
        opt =  self.optimizers()
        sch = self.lr_schedulers()
        images = batch['image']
        texts = batch['text']
        labels = batch['label']
        assert labels.size() == images.size()
        if self.accum_freq == 1:
            t0 = time.time()
            self.standard_loop(images,texts, labels, opt, sch)
            self.log(f"utils/time_per_optimizer_step", time.time() - t0)

        else:
            self.accmulate_loop(images, texts, labels, opt, sch)
        sch.step()
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if self.logit_scale.requires_grad:
            with torch.no_grad():
                self.logit_scale.clamp_(0, np.log(100))

        if batch_idx == 0 and self.enable_image_logging:
            self._log_debug_images(images, stage="train", file_path=batch['file_path'][0], label=batch['label'][0], desc=batch['raw_text'][0])
    

    def embed_image(self, image):
        if isinstance(self.vision, MedNeXt):
            embedding = self.vision(image) #
            embedding = torch.flatten(embedding, start_dim=1)
        if isinstance(self.vision, UNet):
            embedding = self.vision(image)
            embedding = torch.flatten(embedding, start_dim=1)
        else:
            embedding = torch.flatten(self.vision(image), start_dim=1)
                
        if hasattr(self, "vision_projection"):
            return  self.vision_projection(embedding) 
        else:
            return embedding
            
    def embed_text(self,text):
        outputs = self.text(text)
        _, pooled = outputs 
    
        return self.text_projection(pooled) 
                

    def get_embeddings(self, images, texts):
        n = images.shape[0]

        # GET ENCODINGS
        I_f = self.embed_image(images)
        T_f = self.embed_text(texts) 

        I_f =  F.normalize(I_f)
        T_f =  F.normalize(T_f)

        return I_f, T_f
    
    def compute_loss(self, I_f, T_f):          
        n = I_f.shape[0] # BATCH SIZE    

        logits = (T_f @ I_f.T) * self.logit_scale
        
        labels = torch.arange(n, dtype=torch.long, device=self.device)
        loss_I = F.cross_entropy(logits,labels) 
        loss_T = F.cross_entropy(logits.T,labels) 
        loss   = torch.Tensor(loss_I + loss_T)/2
        return loss
        

    def standard_loop(self, images, texts,labels, opt, sch):
        opt.zero_grad()
        I_f, T_f = self.get_embeddings(images,texts)
        loss = self.compute_loss(I_f,T_f)

        # log metrics
        with torch.no_grad():
            positive_dist = F.pairwise_distance(I_f,T_f).mean()
            negative_dist = compute_mean_negative(I_f,T_f)
            distance_diff = negative_dist - positive_dist
            self.train_diff = distance_diff
            images_similarity, texts_similarity, targets_similarity = compute_similarity(I_f,T_f)

            self.log_dict(
                {
                    "train/percentage_with_labels" : self.percentage_ones(labels),
                    "train/loss" : loss,
                    "train/logit_scale": self.logit_scale,
                    "train/positive_distance": positive_dist,
                    "train/negative_distance": negative_dist,
                    "train/distance_diff": distance_diff,
                    "train/mean_image_diagonal_similarity_probability" : images_similarity,
                    "train/mean_text_diagonal_similarity_probability" : texts_similarity,
                    "train/mean_diagonal_target_probability" : targets_similarity
                },
                batch_size=images.size()[0]
            )
        self.manual_backward(loss)
        opt.step()

    def accmulate_loop(self, images, texts, labels, opt, sch):
        with torch.no_grad():
            I_f, T_f = self.get_embeddings(images,texts)
            features = {"text_features" : T_f, "image_features": I_f}
            for key, val in features.items():
                if key in self.accum_features:
                    self.accum_features[key].append(val)
                else:
                    self.accum_features[key] = [val]
            self.accum_images.append(images.detach())
            self.accum_texts.append(texts.detach())
        # If (i + 1) % accum_freq is not zero, move on to the next batch.
        if self.i == 0:
            self.t0 = time.time()
        if ((self.i + 1) % self.accum_freq) > 0:
            self.i += 1
        else:            
            opt.zero_grad()
            for j in range(self.accum_freq):
                images = self.accum_images[j]
                texts = self.accum_texts[j]
                I_f, T_f = self.get_embeddings(images,texts)
                cached_I_f = self.accum_features["image_features"]
                cached_T_f = self.accum_features["text_features"]
                I_f = torch.cat(cached_I_f[:j] + [I_f] + cached_I_f[j + 1:])
                T_f = torch.cat(cached_T_f[:j] + [T_f] + cached_T_f[j + 1:])
                loss = self.compute_loss(I_f,T_f)
                # log metrics
                with torch.no_grad():
                    positive_dist = F.pairwise_distance(I_f,T_f).mean()
                    negative_dist = compute_mean_negative(I_f,T_f)
                    distance_diff = negative_dist - positive_dist
                    self.train_diff = distance_diff
                    images_similarity, texts_similarity, targets_similarity = compute_similarity(I_f,T_f)
                    self.log_dict(
                        {
                            "train/percentage_with_labels" : self.percentage_ones(labels),
                            "train/loss" : loss,
                            "train/logit_scale": self.logit_scale,
                            "train/positive_distance": positive_dist,
                            "train/negative_distance": negative_dist,
                            "train/distance_diff": distance_diff,
                            "train/mean_image_diagonal_similarity_probability" : images_similarity,
                            "train/mean_text_diagonal_similarity_probability" : texts_similarity,
                            "train/mean_diagonal_target_probability" : targets_similarity
                        },
                        batch_size=images.size()[0]
                    )
                del I_f
                del T_f
                self.manual_backward(loss)
            opt.step()
            self.log(f"utils/time_per_optimizer_step", time.time() - self.t0)
            self.i=0
           
            # reset gradient accum, if enabled
            if self.accum_freq > 1:
                self.accum_images, self.accum_texts, self.accum_features = [], [], {}

    def test_step(self, batch, batch_idx):
        images = batch['image']
        texts = batch['text']   
        I_f, T_f = self.get_embeddings(images,texts)
        loss = self.compute_loss(I_f,T_f)

        self.log_dict(
            {
                "test/loss" : loss
            }
        )


    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.inference_mode():
            sch = self.lr_schedulers()
            images = batch['image']
            texts = batch['text']
            I_f, T_f = self.get_embeddings(images,texts)
            loss = self.compute_loss(I_f,T_f)

            # log metrics 
            positive_dist = F.pairwise_distance(I_f,T_f).mean()
            negative_dist = compute_mean_negative(I_f,T_f)
            distance_diff = negative_dist - positive_dist
            images_similarity, texts_similarity, targets_similarity = compute_similarity(I_f,T_f)

            self.log_dict(
                {
                    "val/loss" : loss,
                    "val/positive_distance": positive_dist,
                    "val/negative_distance": negative_dist,
                    "val/distance_diff": distance_diff,
                    "val/mean_image_diagonal_similarity_probability" : images_similarity,
                    "val/mean_text_diagonal_similarity_probability" : texts_similarity,
                    "val/mean_diagonal_target_probability" : targets_similarity,
                    "val/train_vs_val_diff" : torch.abs(self.train_diff - distance_diff),
                },
                batch_size=images.size()[0]
            )
        self.train()
        return loss
    


    def configure_optimizers(self):
        lr_params = []
        if self.text_lr == 0 and self.vision_lr == 0:
            lr_params.append({'params': self.parameters(), 'lr': self.learning_rate, 'name': "base", 'weight_decay': self.base_weight_decay, 'betas' : (self.beta1, self.beta2)})
        else:
            lr_params.append({'params': [p[1] for p in self.base_params], 'lr': self.learning_rate, 'name': 'base', 'weight_decay': self.base_weight_decay,  'betas' : (self.beta1, self.beta2)})
            if self.text_lr > 0:
                lr_params.append({'params':[p[1] for p in self.text_params], 'lr': self.text_lr, 'name':'text', 'weight_decay': self.text_weight_decay,  'betas' : (self.beta1, self.beta2)})
            if self.vision_lr > 0:
                lr_params.append({'params':[p[1] for p in self.vision_params], 'lr': self.vision_lr, 'name':'vision', 'weight_decay': self.vision_weight_decay,  'betas' : (self.beta1, self.beta2)})

        if self.optimizer == 'AdamW':
            optimizer = optim.AdamW(lr_params)
        else:
            optimizer = optim.Adam(lr_params)

        cosine_half_period = int(self.cosine_period_ratio * self.trainer.max_epochs) - int(self.warmup_steps/self.steps_per_epoch)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_half_period * self.steps_per_epoch, eta_min = 1e-8) 

        if self.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1.0 / 1000, total_iters=self.warmup_steps
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, lr_scheduler],
                milestones=[self.warmup_steps],
            )
        else:
            scheduler = lr_scheduler
        scheduler_config = {
            "scheduler": scheduler
        }
        return [optimizer], [scheduler_config]
    
    def _log_debug_images(self, y, stage, file_path, label, desc=""):
        examples = {}
        if self.current_epoch in [0, 1, 2, 3, 4, 8, 10, 15, 24, 49, 74, 99]:
            if self.reconstruction:
                gif = wandb.get_gif(
                    y,
                    slice_dim=0,
                    version_dir=self.version_dir,
                    epoch=self.current_epoch,
                    desc=file_path,
                    label=label,
                    text=desc
                )
                examples[f"{stage}/examples/gif"] = gif
        if examples == {}:
            print("no images")
        if examples != {}:
            
            if len(self.loggers) > 1:
                wandb_logger = self.loggers[1]

                assert isinstance(wandb_logger, WandbLogger), f"Tried to log 3d image, but provided logger was not a Wandb logger, but instead {type(wandb_logger)}"
                wandb_logger.experiment.log(examples)
            else:
                logging.warning("wandb logger not initialized")

    def percentage_ones(self,labels):
        flatten_labels = labels.flatten(start_dim=1)
        ones = (flatten_labels > 0).sum(dim=1) 
        percentage_ones = ones > 0
        return percentage_ones.float().mean()
# https://github.com/moein-shariatnia/OpenAI-CLIP/
# tree/master
    
def compute_mean_negative(I_F,T_F):
    n = T_F.size(0)
    count = 0
    accum = 0
    
    # Generate all possible flip combinations
    for i in range(n):
        for j in range(i + 1, n + 1):
            flipped_subset = torch.cat((T_F[:i], T_F[i:j].flip(0), T_F[j:]), dim=0)
            if not torch.equal(flipped_subset, T_F):
                accum += F.pairwise_distance(I_F, flipped_subset).mean()
                count += 1
    if count == 0 or accum == 0:
        print(count)
        return 0
    return accum / count

def compute_similarity(I_f, T_f):
    images_similarity = torch.diagonal(F.softmax((I_f @ I_f.T), dim=-1)).mean()
    texts_similarity = torch.diagonal(F.softmax((T_f @ T_f.T), dim=-1)).mean()
    target_similarity = torch.diagonal(F.softmax(((T_f @ I_f.T)), dim=-1)).mean()
    return images_similarity, texts_similarity, target_similarity
