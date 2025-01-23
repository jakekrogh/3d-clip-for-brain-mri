#!/usr/bin/env python
import dotenv
import os
import numpy as np
import torch
import lightning as L
import argparse

# Has to be before yucca import to avoid getting warnings
dotenv.load_dotenv()
from yucca.pipeline.configuration.split_data import get_split_config
from yucca.pipeline.configuration.configure_task import TaskConfig
from yucca.pipeline.configuration.configure_paths import get_path_config
# from yucca.training.configuration.configure_plans import get_plan_config
from yucca.pipeline.configuration.configure_checkpoint import get_checkpoint_config
from yucca.pipeline.configuration.configure_seed import seed_everything_and_get_seed_config
from yucca.pipeline.configuration.configure_input_dims import InputDimensionsConfig
from data_loading.YuccaClipDataModule import YuccaClipDataModule
from models.bert import  BertTokenizer
from models.clip import CLIP
from training.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from training.configuration.configure_plans import get_plan_config
from training.configuration.configure_callbacks import get_callback_config
from utils.generators import generate_enum_from_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    parser.add_argument("--project", type=str, default=os.environ.get("PROJECT") or "baselines")
    parser.add_argument("--model_name", type=str, default="GammaKnife")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--planner", type=str, default="YuccaClipPlanner")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--base_weight_decay", type=float, default=0.01)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--enable_logging", action="store_true")
    parser.add_argument("--run_classification", action="store_true")

    parser.add_argument("--new_version", action="store_true")

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)

    # Training Params
    parser.add_argument("--finetune_on_ckpt", type=str, help="Checkpoint to continue training from", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=250)

    # Experiment Params
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--name", type=str)
    parser.add_argument("--description", type=str)
    parser.add_argument("--lr_scheduler", type=str, default='cosine')
    parser.add_argument("--experiment", type=str, required=False, default='')

    parser.add_argument("--split_method", type=str, default="kfold")
    parser.add_argument("--split_param", type=str, default=5)
    parser.add_argument("--split_idx", type=int, required=True, help="Index of the split to use for kfold")
    parser.add_argument("--projection_dim", type=int, help="", default=512)
    parser.add_argument("--log_every_n_steps", type=int, help="", default=10)
    parser.add_argument("--logit_scale_init", type=float, help="", default=1.351)
    parser.add_argument("--cosine_period_ratio", type=float, help="", default=1)
    parser.add_argument("--accum_freq", type=int, help="", default=1)
    parser.add_argument("--warmup_steps", type=int, help="", default=2000)
    # parser.add_argument("--freeze_vision", action="store_true")
    # parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--text_lr", type=float, help="", default=1e-5)
    parser.add_argument("--vision_lr", type=float, help="", default=1e-4)
    parser.add_argument("--text_weight_decay", type=float, help="", default=0.01)
    parser.add_argument("--vision_weight_decay", type=float, help="", default=0.01)
    parser.add_argument("--beta1", type=float, help="", default=0.9)
    parser.add_argument("--beta2", type=float, help="", default=0.999)
    parser.add_argument("--vision_checkpoint", default='swinunet_checkpoints/swinunetr.ckpt')
    parser.add_argument("--vision_type", default='swinunetr')
    parser.add_argument("--no_pretrain_vision", default=False, action="store_true")
    parser.add_argument("--no_pretrain_text", default=False, action="store_true")

    args = parser.parse_args()
    planner = args.planner
    assert args.patch_size % 8 == 0, args.patch_size
    if args.split_method == "kfold":
        split_param = int(args.split_param)
    elif args.split_method == "simple_train_val_split":
        split_param = float(args.split_param)
    else:
        split_param = args.split_param

    yucca_preprocessed_data = os.environ["YUCCA_PREPROCESSED_DATA"]
    run_type = "from_scratch" if args.finetune_on_ckpt is None else "finetune"

    print(f"Using num_workers: {args.num_workers}, num_devices: {args.num_devices}")
    print("ARGS:", args)
    print("Enable logging", args.enable_logging)
    print(args.precision)

    task_config = TaskConfig(
        continue_from_most_recent= not args.new_version,
        manager_name="GammaKnife",
        model_dimensions="3D",
        model_name=args.model_name,
        patch_based_training=True,
        planner_name=planner,
        split_idx=args.split_idx,
        task=args.task,
        experiment=args.experiment,
        split_method=args.split_method,
        split_param=split_param,
    )

    path_config = get_path_config(task_config=task_config)
    ckpt_config = get_checkpoint_config(
        path_config=path_config,
        continue_from_most_recent=task_config.continue_from_most_recent,
        # A note on finetuning vs. continue training:
        # This config takes a checkpoint from where weights are restored, but trainer state will be
        # re-initialized from new. Lightning can also continue training, that is restoring both
        # weights _and_ trainer state, but this is done automatically by setting `ckpt_path="last"`
        # in trainer.fit and starting training with `path_config.version_dir` set to the run you wish
        # to continue training.
        ckpt_path=args.finetune_on_ckpt,
        current_experiment=task_config.experiment,
    )

    seed_config = seed_everything_and_get_seed_config(ckpt_seed=ckpt_config.ckpt_seed)

    plan_config = get_plan_config(
        ckpt_plans=ckpt_config.ckpt_plans,
        plans_path=path_config.plans_path,
        stage="fit",
    )

    splits_config = get_split_config(method=task_config.split_method, param=task_config.split_param, path_config=path_config)
    modalities =  1 #
    input_dims_config = InputDimensionsConfig(
        batch_size=args.batch_size, patch_size=(args.patch_size,) * 3, num_modalities=modalities 
    )

    augmenter = YuccaAugmentationComposer(
        patch_size=input_dims_config.patch_size, task_type_preset=plan_config.task_type  
    )

    # Load Bert Tokenizer
    tokenizer = BertTokenizer(os.environ.get("STATES")+"/bert-base-uncased-vocab.txt")


    data = YuccaClipDataModule(
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        tokenizer=tokenizer,
        input_dims_config=input_dims_config,
        train_data_dir=path_config.train_data_dir,
        plan_config=plan_config,
        splits_config=splits_config,
        split_idx=task_config.split_idx,
        num_workers= args.num_workers,
        n_batches=input_dims_config.batch_size * args.accum_freq * int(args.epochs * args.steps_per_epoch),
        epochs = args.epochs,
        val_sampler=None,
    )  

    effective_batch_size = args.num_devices * input_dims_config.batch_size * args.accum_freq
    train_dataset_size = len(data.splits_config.train(task_config.split_idx))
    val_dataset_size = len(data.splits_config.val(task_config.split_idx))
    print("VAL DATASET SIZE", val_dataset_size)
    steps_per_epoch = args.steps_per_epoch # note this value is in reality scaled by args.accum_freq since an optimizer step happens every accum_freq#int(train_dataset_size / input_dims_config.batch_size)
    max_iterations =  int(args.epochs * steps_per_epoch)
    print(" Steps per epoch", steps_per_epoch)
    print("Train dataset: ", data.splits_config.train(task_config.split_idx))
    print("Val dataset: ", data.splits_config.val(task_config.split_idx))

    assert train_dataset_size >= input_dims_config.batch_size
    wandb_run_name = f"{args.model_name}_k{args.split_idx}"

    # Callbacks
    print(wandb_run_name)
    callback_config = get_callback_config(
        save_dir=path_config.save_dir,
        version_dir=path_config.version_dir,
        version=path_config.version,
        ckpt_version_dir=ckpt_config.ckpt_version_dir,
        ckpt_wandb_id=ckpt_config.ckpt_wandb_id,
        enable_logging=args.enable_logging, 
        interval_ckpt_epochs=9999,
        latest_ckpt_epochs=1,
        log_lr=True,
        log_model=False, 
        profile=True,
        project=args.project,
        steps_per_epoch=steps_per_epoch,
        store_best_ckpt=True,
        write_predictions=False,
        run_name= args.name or wandb_run_name,
        run_description= args.description,
        experiment= "GammaKnife" if run_type == "from_scratch" else args.experiment,
    )

    print("run_type: ", run_type)   
    print("run_name: ", args.name or f"{run_type}_{task_config.experiment}")
    print("group in wandb: ", "GammaKnife" if run_type == "from_scratch" else args.experiment)

    print(
        f"Starting training with {max_iterations} max iterations over {args.epochs} epochs "
        f"with train dataset of size {train_dataset_size} datapoints and val dataset of size {val_dataset_size} "
        f"and effective batch size of {effective_batch_size}"
    )
    ## Init logger
    ## Init CLIP
   
    model = CLIP(
        config=task_config.lm_hparams()
        | path_config.lm_hparams()
        | ckpt_config.lm_hparams()
        | seed_config.lm_hparams()
        | splits_config.lm_hparams()
        | plan_config.lm_hparams()
        | input_dims_config.lm_hparams()
        | callback_config.lm_hparams()
        | {"precision": args.precision}
        | {"val_size": val_dataset_size},
        projection_dim=args.projection_dim, 
        learning_rate=args.learning_rate,
        logit_scale_init=args.logit_scale_init,
        version_dir=path_config.version_dir,
        accum_freq=args.accum_freq,
        lr_scheduler=args.lr_scheduler,
        optimizer=args.optimizer,
        warmup_steps=args.warmup_steps,
        steps_per_epoch = args.steps_per_epoch,
        cosine_period_ratio=args.cosine_period_ratio,
        text_lr=args.text_lr,
        vision_lr=args.vision_lr,
        base_weight_decay = args.base_weight_decay,
        vision_weight_decay = args.vision_weight_decay,
        text_weight_decay = args.text_weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        vision_type=args.vision_type,
        vision_checkpoint=args.vision_checkpoint,
        use_pretrain_vision = not args.no_pretrain_vision,
        use_pretrain_text = not args.no_pretrain_text,
        )
    

            
    trainer = L.Trainer(
            accelerator=os.environ["ACCELERATOR"] if "ACCELERATOR" in os.environ.keys() else "auto",
            strategy="ddp_find_unused_parameters_true" if args.num_devices > 1 else "auto",
            num_nodes=1,
            devices=args.num_devices,
            callbacks=callback_config.callbacks,
            default_root_dir=path_config.save_dir,
            max_epochs=args.epochs,
            limit_train_batches=args.steps_per_epoch, # scale by accum frequency since a step only happens every accum freq
            logger= callback_config.loggers, # wandb_logger
            precision=args.precision,
            fast_dev_run= args.fast_dev_run,
            profiler=callback_config.profiler,
            log_every_n_steps=args.log_every_n_steps
        )
    print(args.log_every_n_steps if args.log_every_n_steps >= input_dims_config.batch_size * args.accum_freq else input_dims_config.batch_size * args.accum_freq)

    trainer.fit(model=model, datamodule=data)
    trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


