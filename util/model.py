import torch

from pytorch_lightning import LightningModule
import torch.nn as nn
from .consts import (
    EPS,
    SIZE_BUCKET_LIST_LABEL,
    SIZE_BUCKET_LIST_LABEL_OUTPUT,
    P99_PERCENTILE_LIST,
)
from .utils import (
    serialize_fp32
)
import numpy as np
import logging
import struct
import os

from .model_llama import Transformer, ModelArgs

class ExpActivation(nn.Module):
    '''
    指数激活函数
    '''
    def __init__(self):
        super(ExpActivation, self).__init__()
        return

    def forward(self, x):
        return torch.exp(x)
class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, prediction, target, weight):
        # Calculate the weighted L1 loss
        loss = torch.abs(prediction - target) * weight

        # Calculate the mean loss
        mean_loss = torch.mean(loss)

        return mean_loss
class FlowSimTransformer_Base(LightningModule):
    def __init__(
        self,
        n_layer,         # Transformer 层数
        n_head,          # 注意力头数
        n_embd,          # Embedding 维度（隐藏层维度）
        block_size,      # 句子/序列的最大长度
        vocab_size,      # 词表大小
        dropout,         # Dropout 概率
        compile,         # 是否启用 PyTorch 编译优化
        loss_fn_type,    # 损失函数类型（"l1" 或 "mse"）
        enable_dist,     # 是否启用分布式训练
        enable_val,      # 是否启用验证步骤
        enable_position, # 是否使用位置编码
        save_dir=None,   # 模型保存路径
    ):
        super().__init__()
        if loss_fn_type == "l1":
            # self.loss_fn = nn.L1Loss()
            self.loss_fn = WeightedL1Loss()
        elif loss_fn_type == "mse":
            self.loss_fn = nn.MSELoss()
        conf = ModelArgs(
            dim=n_embd,
            n_layers=n_layer,
            n_heads=n_head,
            vocab_size=vocab_size,
            multiple_of = 32,
            max_seq_len=block_size,
            dropout=dropout,
        )
        self.model_transformer = Transformer(conf)
        self.enable_dist = enable_dist
        self.enable_val = enable_val
        self.save_dir = save_dir
        logging.info(
            f"loss_fn: {loss_fn_type}, n_layer: {n_layer}, n_head: {n_head}, n_embd: {n_embd}, block_size: {block_size}, vocab_size: {vocab_size}, dropout: {dropout}, enable_position: {enable_position}, enable_dist: {enable_dist}, enable_val: {enable_val}"
        )
    def export_to_bin_llama_v0(self, filepath):
        """ Original export of llama2.c bin files, i.e. version v0 """
        model=self.model_transformer
        out_file = open(filepath, 'wb')

        # first write out the header
        hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
        p = model.params
        shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
        # legacy format uses negative/positive vocab size as a shared classifier flag
        if not shared_classifier:
            p.vocab_size = -p.vocab_size
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                        n_kv_heads, p.vocab_size, p.max_seq_len)
        out_file.write(header)

        # next write out the embedding weights
        serialize_fp32(out_file, model.tok_embeddings.weight)
        serialize_fp32(out_file, model.tok_embeddings.bias)
        
        # now all the layers
        # attention weights
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention_norm.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wq.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wk.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wv.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.wo.weight)
        # ffn weights
        for layer in model.layers:
            serialize_fp32(out_file, layer.ffn_norm.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w1.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w2.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.feed_forward.w3.weight)
        # final rmsnorm
        serialize_fp32(out_file, model.norm.weight)
        # freqs_cis
        serialize_fp32(out_file, model.freqs_cos[:p.max_seq_len])
        serialize_fp32(out_file, model.freqs_sin[:p.max_seq_len])

        # final classifier weights
        if not shared_classifier:
            serialize_fp32(out_file, model.output.weight)

        # write to binary file
        out_file.close()
        print(f"wrote {filepath}")

    def step(self, batch, batch_idx, tag=None):
        return None

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag="train")

    def validation_step(self, batch, batch_idx):
        if self.enable_val:
            return self.step(batch, batch_idx, tag="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, tag="test")

class FlowSimTransformer_Path(FlowSimTransformer_Base):
    def __init__(
        self,
        n_layer=4,                   # Transformer 模型的层数
        n_head=4,                    # Transformer 中多头自注意力机制的头数
        n_embd=64,                   # 每个嵌入向量的维度
        block_size=64,               # 输入块的大小
        vocab_size=50257,            # 词汇表大小（词汇表维度）
        dropout=0.0,                 # Dropout 率，用于防止过拟合
        compile=False,               # 是否使用 PyTorch 的编译功能以加速
        loss_fn_type="l1",           # 损失函数的类型，默认是 L1 损失
        weight_decay=1e-2,           # 权重衰减系数，用于优化器中的正则化
        learning_rate=6e-4,          # 学习率
        betas=[0.9, 0.95],           # Adam 优化器中的 β 参数
        batch_size=400,              # 批次大小
        enable_masked_loss=False,    # 是否启用掩码损失
        enable_weighted_loss=False,  # 是否启用加权损失
        enable_context=False,        # 是否启用上下文数据
        hidden_sizes=None,           # MLP 模型中的隐藏层大小列表
        activation=nn.ReLU,          # 使用的激活函数，默认是 ReLU
        output_activation=nn.Identity,  # 输出层激活函数，默认是 Identity（无激活）
        enable_dist=False,           # 是否启用分布式训练
        enable_val=True,             # 是否启用验证过程
        enable_position=True,        # 是否启用位置嵌入
        enable_log=False,            # 是否启用日志记录
        enable_const_opt=True,       # 是否启用常量优化
        n_params=1,                  # 传递给特征维度的额外参数数量
        save_dir=None,               # 保存模型的目录
    ):
        # SIZE_BUCKET_LIST_LABEL:用于将数据流量按大小分桶（bucket）
        # P99_PERCENTILE_LIST:[1,2,3 ... 100]
        # feat_dim:特征维度，即大小桶和时延百分位数的乘积数量,同时res=len(fcts_list)*feat_dim
        # 增加额外的特征维度
        feat_dim = len(SIZE_BUCKET_LIST_LABEL) * len(P99_PERCENTILE_LIST)
        feat_dim+=n_params
        super().__init__(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            vocab_size=feat_dim,
            dropout=dropout,
            compile=compile,
            loss_fn_type=loss_fn_type,
            enable_dist=enable_dist,
            enable_val=enable_val,
            enable_position=enable_position,
            save_dir=save_dir,
        )
        if enable_log:
            output_activation = ExpActivation
            logging.info(f"use ExpActivation")
        self.n_embd = n_embd
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = tuple(betas)
        self.batch_size = batch_size
        self.enable_masked_loss = enable_masked_loss
        self.enable_weighted_loss = enable_weighted_loss
        self.enable_context = enable_context
        self.enable_const_opt=enable_const_opt
        

        input_dim = feat_dim + n_embd if enable_context else feat_dim
        #input_dim = feat_dim + n_embd + 1 if enable_context else feat_dim + 1
        # 增加一个新的维度,max_fct_flowsim
        # 确定输出维度（前景流和背景流大小桶的输出）
        output_dim = len(P99_PERCENTILE_LIST) * len(SIZE_BUCKET_LIST_LABEL_OUTPUT)
        # 设置 MLP 的层次结构，sizes 包含输入层、隐藏层和输出层大小
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        self.feat_dim = feat_dim
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dim_list=hidden_sizes
        self.model_mlp = self.mlp(
            sizes=sizes,
            activation=activation,
            output_activation=output_activation,
            # dropout=dropout
        )
        self.y_len=len(P99_PERCENTILE_LIST)
        if enable_const_opt:
            self.const_tensor=nn.Parameter(torch.zeros(self.y_len))
        logging.info(
            f"model: {sizes}, enable_context: {enable_context},enable_const_opt:{enable_const_opt}")

    def mlp(self, sizes, activation, output_activation, dropout=None):
        layers = []
        # 为每个层次构建线性层和激活函数
        for j in range(len(sizes) - 1):
            # 如果是输出层，使用输出激活函数
            act = activation if j < len(sizes) - 2 else output_activation
            # 如果有 Dropout，并且是第一层，添加 Dropout 层
            if j == 0 and dropout:
                layers += [
                    nn.Linear(sizes[j], sizes[j + 1]),  # 线性层
                    nn.Dropout(dropout),                # Dropout 层
                    act(),                              # 激活函数
                ]
            else:
                # 添加线性层和激活函数
                layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        # 返回由多个层组成的顺序模型
        return nn.Sequential(*layers)
    
    def export_to_bin_mlp(self, filepath="model_mlp.bin"):
        """export the model weights in fp32 into .bin file to be read from C"""
        f = open(filepath, "wb")

        print(f"mlp: {self.input_dim}, {self.hidden_dim_list[0]}, {self.hidden_dim_list[1]}, {self.output_dim}")
        header = struct.pack(
            "iiiii", self.input_dim, self.hidden_dim_list[0], self.hidden_dim_list[1], self.output_dim, self.y_len
        )
        print(header)
        f.write(header)
        # now all the layers
        for layer in self.model_mlp:
            if isinstance(layer, nn.Linear):
                # print(f"{layer.weight.shape}, {layer.bias.shape}")
                # print(
                #     f"[{layer.weight[0,0]}, {layer.weight[-1,-1]}], [{layer.bias[0]}, {layer.bias[-1]}]"
                # )
                serialize_fp32(f,layer.weight)
                serialize_fp32(f,layer.bias)
        if self.enable_const_opt:
            print(f"const_tensor: {self.const_tensor[0]}, {self.const_tensor[-1]}")
            serialize_fp32(f,self.const_tensor)
        # write to binary file
        f.close()
        print(f"wrote {filepath}")
        
    def step(self, batch, batch_idx, tag=None):
        '''
        return (
            res,                            # 特征图 (foreground 和 background 流的 size bucket 和时延分布百分位数)
            num_flows_per_cell,             # 每个特征图桶中流的数量，用于描述流量分布的密度
            res_output,                     # 前景流（目标主机对）的真实特征图，作为网络训练的目标
            num_flows_per_cell_output,      # 前景流中每个桶的流量数量
            spec,                           # 工作负载规格的字符串表示，标识仿真实验参数，如 shard、流量、主机数量等
            n_input,                        # 前景流和背景流的总数，表示特征图的数量
            src_dst_pair_target_str,        # 目标主机对的字符串表示，用于唯一标识某个流的通信对
        )
        '''


        # 从 batch 中解包提取的数据：
        # sizebucket_to_sldn_flowsim: 流量仿真相关的大小桶到时延比例数据（仿真数据）
        # num_flows_per_cell_flowsim: 仿真中的每个流量桶中流的数量
        # sizebucket_to_sldn: 前景流的数据（真实数据，用于损失计算）
        # num_flows_per_cell: 每个流量桶中前景流的数量
        # spec: 具体的仿真配置参数，用于标识该批次中的配置情况
        # sizebucket_to_sldn_flowsim_idx: 仿真数据中每个流量桶的索引
        # src_dst_pair_target_str: 源和目标主机对，用于唯一标识某个通信对
        (
            sizebucket_to_sldn_flowsim, #res
            num_flows_per_cell_flowsim, #num_flows_per_cell 
            sizebucket_to_sldn, #res_output
            num_flows_per_cell, #num_flows_per_cell_output
            spec, #spec
            sizebucket_to_sldn_flowsim_idx, #n_input
            src_dst_pair_target_str #src_dst_pair_target_str
        ) = batch
        #global_max_fct = torch.max(max_fct_flowsim)  # 获取整个批次的最大值，形状为标量 ()
        
        # 如果启用了常量优化（即 enable_const_opt 为 True），
        # 将那些流量较少的桶设置为常量值，以减少波动和优化模型性能
        if self.enable_const_opt:
            # 重塑流量桶数量，变成三维：批次大小、每个流量桶、和 y_len（时延百分位数的长度）
            num_flows_per_cell_flowsim = num_flows_per_cell_flowsim.reshape(
                (num_flows_per_cell_flowsim.shape[0], -1, self.y_len)
            ).mean(dim=-1)  # 计算沿着最后一个维度的平均值

            # 遍历每个流量桶并检查其流量是否低于阈值 EPS，如果低于，则使用常量张量
            for idx_1 in range(num_flows_per_cell_flowsim.shape[0]):
                for idx_2 in range(num_flows_per_cell_flowsim.shape[1]):
                    if num_flows_per_cell_flowsim[idx_1, idx_2] < EPS:
                        # 将相应的流量桶的值设置为常量张量
                        sizebucket_to_sldn_flowsim[
                            idx_1, idx_2 * self.y_len : (idx_2 + 1) * self.y_len
                        ] = self.const_tensor

        # 如果启用了上下文模式（即 enable_context 为 True），将背景流和前景流的特征结合起来作为输入
        if self.enable_context:
            idx_start = 0
            # 初始化用于存储前景流特征和背景流上下文特征的张量
            sizebucket_to_sldn_foreground = sizebucket_to_sldn.new(
                len(spec), self.feat_dim
            )
            sizebucket_to_sldn_context = sizebucket_to_sldn.new(
                len(spec), self.n_embd
            )

            # 遍历批次中的每个仿真配置 spec
            for i in range(len(spec)):
                # 提取当前 spec 对应的前景流特征
                sizebucket_to_sldn_foreground[i] = sizebucket_to_sldn_flowsim[idx_start]

                # 获取该 spec 对应的流量桶数量（即 idx_interval），用于提取背景流
                idx_interval = sizebucket_to_sldn_flowsim_idx[i]

                # 提取背景流数据，范围是从 idx_start + 1 到 idx_start + idx_interval
                tmp = sizebucket_to_sldn_flowsim[
                    idx_start + 1 : idx_start + idx_interval
                ]

                # 使用 Transformer 模型处理背景流，返回背景流的特征
                sizebucket_to_sldn_background, _ = self.model_transformer(tmp[None, :])

                # 对背景流特征取平均值，得到上下文特征，并将其存储到上下文张量中
                sizebucket_to_sldn_context[i] = torch.mean(sizebucket_to_sldn_background, dim=1)

                # 更新 idx_start，用于处理下一个 spec
                idx_start += idx_interval

            # 将前景流和上下文特征拼接在一起，形成最终输入
            sizebucket_to_sldn_input = torch.cat(
                [sizebucket_to_sldn_foreground, sizebucket_to_sldn_context], dim=-1
            )
        else:
            # 如果未启用上下文模式，仅使用前景流特征
            sizebucket_to_sldn_foreground = sizebucket_to_sldn_flowsim[:, 0, :]
            sizebucket_to_sldn_input = sizebucket_to_sldn_foreground
        # sizebucket_to_sldn_input.shape = (batch_size, feat_dim + n_embd)
        # 通过 MLP 模型对前景流（或者前景加上下文流）特征进行处理，得到预测的大小桶到时延估计
        sizebucket_to_sldn_est = self.model_mlp(sizebucket_to_sldn_input)
        sizebucket_to_sldn_est.add_(1)  # 对预测结果进行 In-place 加法操作

        # 使用 num_flows_per_cell 作为加权矩阵，计算损失权重
        loss_weights = num_flows_per_cell > 0.0

        # 计算损失值，使用自定义损失函数
        # - sizebucket_to_sldn_est: 预测的大小桶到时延比例
        # - sizebucket_to_sldn: 真实的大小桶到时延比例
        # - loss_weights: 损失的权重（基于流量桶中的流数）
        # breakpoint()
        loss = self.loss_fn(
            torch.div(sizebucket_to_sldn_est, sizebucket_to_sldn),  # 计算预测与真实值的比率
            torch.ones_like(sizebucket_to_sldn),  # 使用全 1 张量作为参考
            loss_weights  # 权重
        )

        # 如果启用了分布式训练，则同步日志记录
        if self.enable_dist:
            self.log(
                f"{tag}_loss_sync",
                loss,
                sync_dist=True,  # 同步所有设备上的损失值
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,  # 在进度条上显示损失
                batch_size=self.batch_size,
            )
        else:
            # 如果未启用分布式训练，直接记录损失
            self.log(
                f"{tag}_loss",
                loss,
                on_step=True,   # 每步记录
                on_epoch=True,  # 每个 epoch 结束时记录
                logger=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )

        if tag == "test":
            test_dir = f"{self.save_dir}/{spec[0]}_{src_dst_pair_target_str[0]}"
            os.makedirs(test_dir, exist_ok=True)
            sizebucket_to_sldn_flowsim = sizebucket_to_sldn_flowsim.cpu().numpy()[0]
            sizebucket_to_sldn_est = sizebucket_to_sldn_est.cpu().numpy()[0]
            sizebucket_to_sldn = sizebucket_to_sldn.cpu().numpy()[0]
            num_flows_per_cell = num_flows_per_cell.cpu().numpy()[0]
            np.savez(
                f"{test_dir}/res.npz",
                sizebucket_to_sldn_est=sizebucket_to_sldn_est,
                sizebucket_to_sldn_flowsim=sizebucket_to_sldn_flowsim,
                sizebucket_to_sldn=sizebucket_to_sldn,
                num_flows_per_cell=num_flows_per_cell,
            )
        return loss

    def configure_optimizers(self):
        optimizer = self.model_transformer.configure_optimizers(
            self.weight_decay, self.learning_rate, self.betas
        )
        optimizer.add_param_group(
            {"params": self.model_mlp.parameters(), "weight_decay": 0.0}
        )
        if self.enable_const_opt:
            optimizer.add_param_group(
                {"params": self.const_tensor, "weight_decay": 0.0}
            )
        return optimizer