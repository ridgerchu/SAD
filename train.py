import os
import time
import socket
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb

from sad.config import get_parser, get_cfg
from sad.datas.dataloaders import prepare_dataloaders
from sad.trainer import TrainingModule



def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)
    # print(cfg)
    trainloader, valloader = prepare_dataloaders(cfg)
    print("load data!!!")
    model = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            cfg.PRETRAINED.PATH, map_location='cpu'
        )['state_dict']
        state = model.state_dict()
        pretrained_model_weights = {k: v for k, v in pretrained_model_weights.items() if k in state and 'decoder' not in k}
        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H_%M_%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    # tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    wandb_logger = WandbLogger(name='sad-preception', project='sad', entity="ncg_ucsc", config=cfg, log_model="all")

    # if os.environ.get('LOCAL_RANK', '0') == '0':
    #     wandb.init(name='sad-preception', project='sad', entity="snn_ad", config=cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor='step_val_seg_iou_dynamic',
        save_top_k=-1,
        save_last=True,
        period=1,
        mode='min'
    )
    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator='ddp',
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary='full',
        logger=wandb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        plugins=DDPPlugin(find_unused_parameters=False),
        profiler='simple',
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()
