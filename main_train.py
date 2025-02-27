from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from util.dataset import PathDataModule
from util.arg_parser import create_config
from util.utils import (
    fix_seed,
    create_logger,
) 
from util.model import (
    FlowSimTransformer_Path,
)
from util.callback import OverrideEpochStepCallback
import logging, os
import torch
import yaml
import numpy as np

torch.set_float32_matmul_precision(precision="high")
args = create_config()
config_file = args.train_config if args.mode == "train" else args.test_config
with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]

shard = dataset_config["shard"]#0
fix_seed(shard)
# 设置随机种子，确保代码在每次运行时结果相同，使其可复现

# get dataset configurations
shard_list_config = dataset_config["shard_list"]#[0,100,100]

shard_list = sorted(
    np.random.choice(
        np.arange(shard_list_config[0], shard_list_config[1]),
        size=shard_list_config[2],
        replace=False,
    )
)
# 从shard_list_config的第0个维度到第1个维度所表示的数中，选取shard_list_config第3个维度个数并且按照升序排序
n_flows_list = dataset_config["n_flows_list"]#[100]
n_hosts_list = dataset_config["n_hosts_list"]# [3]
sample_list_config = dataset_config["sample_list"]# [0,1,1]
sample_list = sorted(
    np.random.choice(
        np.arange(sample_list_config[0], sample_list_config[1]),
        size=sample_list_config[2],
        replace=False,
    )
)

lr = dataset_config["lr"] #link rate: 10Gbps
n_params=dataset_config["n_params"] # 19

note_str = f"{args.note}_" if args.note else ""#生成一些标识
# 生成程序的名字
program_name = f"{note_str}shard{len(shard_list)}_nflows{len(n_flows_list)}_nhosts{len(n_hosts_list)}_nsamples{len(sample_list)}_lr{lr}Gbps"
# 该回调类的作用是覆盖每个 epoch 结束时的行为，主要是在训练、验证和测试过程中将当前 epoch 记录到日志中。
# 此外，它还可以在训练的第一个 epoch 结束时切换数据模块的逻辑
override_epoch_step_callback = OverrideEpochStepCallback()
dir_output=args.dir_output
dir_input=args.dir_input


if args.mode == "train":
    # 训练日志
    tb_logger = TensorBoardLogger(dir_output, name=program_name)
    
    # configure logging at the root level of Lightning
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    # 生成模型名字
    hidden_dims_str = "-".join([str(x) for x in model_config["hidden_dims"]])
    model_name = "bs{}_lr{}_bt{}_nlayer{}_nhead{}_nembd{}_block{}_vocab{}_hd{}".format(
        training_config["batch_size"],
        training_config["learning_rate"],
        dataset_config["bucket_thold"],
        model_config["n_layer"],
        model_config["n_head"],
        model_config["n_embd"],
        model_config["block_size"],
        model_config["vocab_size"],
        hidden_dims_str,
    )
    create_logger(os.path.join(tb_logger.log_dir, f"{model_name}.log"))
    logging.info(f"Save to: {tb_logger.log_dir}")
    logging.info(args)
    # 是否启用分布式训练，是否启用验证集评估
    enable_dist = training_config["enable_dist"]
    enable_val = training_config["enable_val"]
    # 配置分布式训练策略
    if enable_dist:
        from pytorch_lightning.strategies import DDPStrategy

        logging.info(
            f"gloo: {torch.distributed.is_gloo_available()}, nccl: {torch.distributed.is_nccl_available()}"
        )
        
        ddp_strategy = DDPStrategy(
            process_group_backend="gloo", find_unused_parameters=True
        )
        
    with open(f"{tb_logger.log_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    print("topo_type:",dataset_config.get("topo_type", ""))
    # 数据模块
    datamodule = PathDataModule(
        dir_input=dir_input,
        shard_list=shard_list,
        n_flows_list=n_flows_list,
        n_hosts_list=n_hosts_list,
        sample_list=sample_list,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        train_frac=dataset_config["train_frac"],
        dir_output=tb_logger.log_dir,
        lr=lr,
        bucket_thold=dataset_config["bucket_thold"],
        topo_type=dataset_config.get("topo_type", ""),
        enable_context=dataset_config.get("enable_context", False),
    )

    # 检查点
    if enable_val:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_sync" if enable_dist else "val_loss",
            dirpath=f"{tb_logger.log_dir}/checkpoints",
            filename="model-{epoch:03d}-{step:05d}-{val_loss_sync:.2f}"
            if enable_dist
            else "model-{epoch:03d}-{step:05d}-{val_loss:.2f}",
            save_top_k=5,  #最多保留五个
            save_last=True,#额外保存最新的那个
            # every_n_train_steps=5,
            mode="min",    #方向为最小化
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss_sync" if enable_dist else "train_loss",
            dirpath=f"{tb_logger.log_dir}/checkpoints",
            filename="model-{epoch:03d}-{step:05d}-{train_loss_sync:.2f}"
            if enable_dist
            else "model-{epoch:03d}-{step:05d}-{train_loss:.2f}",
            save_top_k=5,
            save_last=True,
            # every_n_train_steps=5,
            mode="min",
        )

    callbacks = [checkpoint_callback, override_epoch_step_callback]
    # 训练框架
    trainer = Trainer(
        logger=[tb_logger],  #  训练日志记录器，使用 TensorBoard 记录训练过程
        callbacks=callbacks,  #  训练回调，如 Checkpoint、EarlyStopping 等
        max_epochs=training_config["n_epochs"],  #  训练的最大 epoch 数，从配置文件中获取
        accelerator="gpu",  #  使用 GPU 进行训练，可选："cpu", "gpu", "tpu", "mps"（Mac）等
        devices=training_config["gpu"],  #  使用的 GPU 数量，从配置文件 `training_config` 读取
        strategy=ddp_strategy if enable_dist else "auto",  #  使用 DDP（分布式数据并行）策略，单机模式下自动选择最佳策略
        default_root_dir=dir_output,  #  训练输出的默认目录（如日志、模型检查点）
        log_every_n_steps=1,  #  每 1 个训练步骤记录一次日志（默认较高频率以便 Debug）
        val_check_interval=1.0,  #  每个 epoch 结束时执行一次验证（1.0 表示每个 epoch 都验证）
        
        # 额外可选参数（已被注释）
        # log_every_n_steps=args.n_epochs_every_log,  #  也可以自定义 `n_epochs_every_log` 来设置日志间隔
        # fast_dev_run=args.debug,  #  如果 `args.debug=True`，快速测试一小部分数据以检查代码
        # limit_train_batches=1,  #  限制训练时的 batch 数（仅用于 Debug）
        # limit_val_batches=1,  #  限制验证时的 batch 数（仅用于 Debug）
        # enable_progress_bar=True,  #  训练时是否显示进度条
    )
    model_name = model_config["model_name"]
    # 具体训练模型——transformer
    if model_name == "transformer":
        model = FlowSimTransformer_Path(
            n_layer=model_config["n_layer"],               # Transformer 模型的层数（Encoder 的数量）。
            n_head=model_config["n_head"],                 # Transformer 中多头注意力机制的头数。
            n_embd=model_config["n_embd"],                 # 每个嵌入向量的维度。
            block_size=model_config["block_size"],         # Transformer 模型的块大小（输入序列的最大长度）。
            vocab_size=model_config["vocab_size"],         # 词汇表大小（输入特征的总数或类别数）。
            dropout=model_config["dropout"],               # Dropout 比例，用于防止过拟合。
            compile=model_config["compile"],               # 是否使用 PyTorch 的编译功能（加速模型）。
            loss_fn_type=model_config["loss_fn_type"],     # 损失函数的类型（如 L1 损失或 MSE 损失）。
            weight_decay=training_config["weight_decay"],  # 权重衰减系数，用于优化器中的正则化。
            learning_rate=training_config["learning_rate"],# 模型的学习率，用于优化器。
            betas=training_config["betas"],                # Adam 优化器的动量参数 beta，用于一阶和二阶动量。
            batch_size=training_config["batch_size"],      # 每个训练批次的样本数量。
            enable_masked_loss=training_config["enable_masked_loss"], # 是否启用掩码损失，用于处理序列填充。
            enable_weighted_loss=training_config["enable_weighted_loss"], # 是否启用加权损失，用于调整特定样本的损失权重。
            enable_context=dataset_config.get("enable_context", False), # 是否启用上下文信息，用于增强模型输入特征。
            hidden_sizes=model_config["hidden_dims"],      # MLP 模型中的隐藏层大小列表。
            enable_position=model_config["enable_position"], # 是否启用位置编码（序列中元素的顺序信息）。
            enable_val=enable_val,                         # 是否启用验证过程。
            enable_dist=enable_dist,                       # 是否启用分布式训练。
            enable_log=training_config["enable_log"],      # 是否启用日志记录，用于训练过程的跟踪。
            n_params=n_params,                             # 传递给特征维度的额外参数数量。
        )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path) # 使用 PyTorch Lightning 的 Trainer 开始训练。
    '''
    分别调用
    1.setup(stage = "fit");
    2.train_dataloader()
    '''
else:
    DEVICE = torch.device(training_config["gpu"][0])
    dir_train = (
        f"{dir_output}/{program_name}/version_{args.version_id}"
    )
    print(f"load model: {dir_train}")

    tb_logger = TensorBoardLogger(dir_train, name="test")

    # configure logging at the root level of Lightning
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    create_logger(os.path.join(tb_logger.log_dir, f"console.log"))
    logging.info(f"Save to: {tb_logger.log_dir}")
    logging.info(args)
    
    datamodule = PathDataModule(
        dir_input=dir_input,
        shard_list=shard_list,
        n_flows_list=n_flows_list,
        n_hosts_list=n_hosts_list,
        sample_list=sample_list,
        batch_size=training_config["batch_size"], 
        num_workers=training_config["num_workers"],
        train_frac=dataset_config["train_frac"],
        dir_output=dir_train,
        # customized config
        lr=dataset_config["lr"],
        bucket_thold=dataset_config["bucket_thold"],
        enable_context=dataset_config.get("enable_context", False),
        topo_type=dataset_config.get("topo_type", ""),#_topo-pl-x_
        mode=args.mode,
        test_on_train=args.test_on_train,
        test_on_empirical=args.test_on_empirical,
        test_on_manual=args.test_on_manual,
    )

    callbacks = [override_epoch_step_callback]

    trainer = Trainer(
        logger=[tb_logger],
        callbacks=callbacks,
        # max_epochs=training_config["n_epochs"],
        accelerator="gpu",
        devices=training_config["gpu"],
        # strategy=ddp_strategy if enable_dist else "auto",
        default_root_dir=dir_train,
        log_every_n_steps=1,
        # val_check_interval=0.1
        # fast_dev_run=args.debug,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # enable_progress_bar=False,
    )
    model_name = model_config["model_name"]
    if model_name == "transformer":
        model = FlowSimTransformer_Path.load_from_checkpoint(
            f"{dir_train}/checkpoints/last.ckpt",
            map_location=DEVICE,
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            n_embd=model_config["n_embd"],
            block_size=model_config["block_size"],
            vocab_size=model_config["vocab_size"],
            dropout=model_config["dropout"],
            compile=model_config["compile"],
            loss_fn_type=model_config["loss_fn_type"],
            weight_decay=training_config["weight_decay"],
            learning_rate=training_config["learning_rate"],
            betas=training_config["betas"],
            batch_size=training_config["batch_size"],
            enable_val=training_config["enable_val"],
            enable_dist=training_config["enable_dist"],
            enable_masked_loss=training_config["enable_masked_loss"],
            enable_weighted_loss=training_config["enable_weighted_loss"],
            enable_context=dataset_config.get("enable_context", False),
            hidden_sizes=model_config["hidden_dims"],
            enable_position=model_config["enable_position"],
            enable_log=training_config["enable_log"],
            n_params=n_params,
            save_dir=tb_logger.log_dir,
        )
    trainer.test(model, datamodule=datamodule)
