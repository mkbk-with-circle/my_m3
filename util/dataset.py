from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import torch
from .consts import (
    SRC2BTL,
    BTL2DST,
    SIZE_BUCKET_LIST,
    P99_PERCENTILE_LIST,
    PERCENTILE_METHOD,
    MTU,
    HEADER_SIZE,
    BYTE_TO_BIT,
    DELAY_PROPAGATION,
    DELAY_PROPAGATION_BASE,
    SIZEDIST_LIST_EMPIRICAL,
    UTIL_LIST,
    IAS_LIST,
    BDP_DICT,
    LINK_TO_DELAY_DICT,
    BDP,
    get_size_bucket_list,
    get_size_bucket_list_output,
    get_base_delay_pmn,
)
from .utils import decode_dict
import json
import logging
import os

def my_collate(batch):
    # 这里将多个样本中的sizebucket_to_sldn_flowsim (flowsim的流量大小桶数据) 进行拼接，整合工作负载数据
    sizebucket_to_sldn_flowsim = [item[0] for item in batch]
    sizebucket_to_sldn_flowsim = np.concatenate(sizebucket_to_sldn_flowsim, 0)

    num_flows_per_cell_flowsim = [item[1] for item in batch]
    num_flows_per_cell_flowsim = np.concatenate(num_flows_per_cell_flowsim, 0)

    # 同时也整合了sizebucket_to_sldn (真实数据) 和 num_flows_per_cell
    sizebucket_to_sldn = np.array([item[2] for item in batch])
    num_flows_per_cell = np.array([item[3] for item in batch])
    spec = np.array([item[4] for item in batch])
    sizebucket_to_sldn_flowsim_idx = np.array([item[5] for item in batch])
    src_dst_pair_target_str = np.array([item[6] for item in batch])
    
    # 返回拼接后的工作负载数据
    return (
        torch.tensor(sizebucket_to_sldn_flowsim),
        torch.tensor(num_flows_per_cell_flowsim),
        torch.tensor(sizebucket_to_sldn),
        torch.tensor(num_flows_per_cell),
        spec,
        sizebucket_to_sldn_flowsim_idx,
        src_dst_pair_target_str
    )

class PathDataModule(LightningDataModule):

    # 构造函数中接收网络仿真相关参数，其中包括工作负载相关的流量数量和主机数量
    def __init__(
        self,
        dir_input,               # 输入数据的目录路径，存放原始仿真数据或工作负载数据
        shard_list,              # 分片列表，包含多个 shard 编号，用于指定不同的工作负载或数据分片
        n_flows_list,            # 流量数量列表，每个仿真任务使用不同的流量数
        n_hosts_list,            # 主机数量列表，用于仿真不同规模的网络拓扑
        sample_list,             # 样本列表，表示不同的采样点，用于增加工作负载的多样性
        batch_size,              # 批次大小，每次训练或验证时使用多少个样本
        num_workers,             # 数据加载的并行线程数，决定同时使用多少个线程来加载数据
        train_frac,              # 训练集的比例，表示数据集中分多少用于训练，剩余的用于验证
        dir_output,              # 输出目录路径，存放训练或测试生成的结果数据
        lr,                      # 链路速率，通常表示网络中的带宽，如 10Gbps
        bucket_thold,            # 桶阈值，用于特征图生成时的流量分桶
        mode="train",            # 模式选择，默认为 "train"，可以是 "train" 或 "test"，决定执行训练或测试
        test_on_train=False,     # 是否在训练集上进行测试，用于模型验证
        test_on_empirical=False, # 是否在经验数据上进行测试，评估仿真结果的准确性
        test_on_manual=False,    # 是否使用手动设定的测试集进行测试
        enable_context=False,    # 是否启用上下文模式，影响数据加载和处理方式
        topo_type="",            # 拓扑类型，指定仿真的网络拓扑结构，"_topo-pl-x_"
    ) -> None:
        """
        Initializes a new instance of the class with the specified parameters.

        Args:
            positive_ratio (float, optional): The ratio of positive to negative samples to use for training.
                Defaults to 0.8.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.dir_input = dir_input
        self.dir_output = dir_output
        data_list = []
        if mode == "train":
            # 在训练模式下，循环遍历 shard、流量数量、主机数量，并生成仿真工作负载的不同组合
            # 准备训练用的数据组合
            for shard in shard_list:
                for n_flows in n_flows_list:
                    for n_hosts in n_hosts_list:
                        topo_type_cur = topo_type.replace(
                            "-x_", f"-{n_hosts}_"
                        )
                        spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{lr}Gbps"
                        for sample in sample_list:
                            data_list.append(
                                (spec, (0, n_hosts - 1), topo_type_cur+f"s{sample}")
                            )
                            
            np.random.shuffle(data_list)
        
        self.data_list = data_list
        self.lr = lr
        self.bucket_thold = bucket_thold
        self.test_on_train = test_on_train
        self.test_on_empirical = test_on_empirical
        self.test_on_manual = test_on_manual
        self.enable_context = enable_context
        self.topo_type = topo_type

        # 初始化统计数据
        flowsim_count = 0
        flowsim_mean = 0
        flowsim_M2 = 0

        flowsim_overall_variance = 0
        overall_variance = 0

        count = 0
        mean = 0
        M2 = 0

        for spec, src_dst_pair_target, topo_type in self.data_list:
            dir_input_tmp = f"{dir_input}/{spec}"

            fct_flowsim_path = os.path.join(dir_input_tmp, "fct_flowsim.npy")
            if os.path.exists(fct_flowsim_path):
                # 逐个加载fct_flowsim数据
                fct_flowsim = np.load(fct_flowsim_path)
                # 遍历当前文件中的每个数据项，先对数据进行对数变换，然后使用Welford算法更新均值和方差
                for x_new in fct_flowsim:
                    if(x_new==0):
                        continue
                    # 取对数
                    log_x_new = np.log(x_new)

                    flowsim_count += 1
                    # 更新均值
                    delta = log_x_new - flowsim_mean
                    flowsim_mean = flowsim_mean*((flowsim_count-1)/flowsim_count) + log_x_new/flowsim_count
                    # 更新方差
                    flowsim_M2 += delta * (log_x_new - flowsim_mean)

            fct_path = f"{dir_input_tmp}/fct{topo_type}.npy"
            if os.path.exists(fct_path):
                fcts = np.load(fct_path)
                for x_new in fcts:
                    if(x_new==0):
                        continue
                    log_x_new = np.log(x_new)
                    count += 1
                    delta = log_x_new - mean
                    mean = mean*((count-1)/count) + log_x_new/count
                    M2 += delta * (log_x_new - mean)


        # 最后计算的全局均值和方差
        if flowsim_count > 0:
            flowsim_overall_variance = flowsim_M2 / flowsim_count
        if count > 0:
            overall_variance = M2 / count
        
        self.flowsim_mean = flowsim_mean
        self.flowsim_overall_variance = flowsim_overall_variance
        self.mean = mean
        self.overall_variance = overall_variance



    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders.

        Args:
            stage (str): The current stage of the training process. Either "fit" or "test".

        Returns:
            None
        """
        # 训练阶段，分割数据集为训练集和验证集
        '''
        此时data_list中有：
            1. spec: 仿真的具体配置描述（分片编号、流量数量、主机数量、链路速率）
            2. 主机范围: 表示当前仿真配置的主机编号范围
            3. 拓扑类型和样本编号: 表示网络拓扑类型和样本编号，用于区分不同的仿真实例

        '''
        if stage == "fit":
            self.train_list, self.val_list = self.__random_split_list(
                self.data_list,
                self.train_frac,
            )
            num_train, num_val = (
                len(self.train_list),
                len(self.val_list),
            )
            logging.info(f"#tracks: train-{num_train}, val-{num_val}")
            self.train = self.__create_dataset(
                self.train_list,
                self.dir_input,
            )
            self.val = self.__create_dataset(
                self.val_list,
                self.dir_input,
            )

            self.__dump_data_list(self.dir_output)

        if stage == "test":
            if self.test_on_empirical:
                data_list_test = []
                # 循环生成工作负载，并根据仿真结果加载数据
                for shard in np.arange(10000, 10200):
                    for n_flows in [30000]:
                        for n_hosts in [2, 3, 4, 5, 6, 7, 8]:
                            topo_type_cur = self.topo_type.replace(
                                "x-x", f"{n_hosts}-{n_hosts}"
                            )
                            spec = f"shard{shard}_nflows{n_flows}_nhosts{n_hosts}_lr{self.lr}Gbps"
                            dir_input_tmp = f"{self.dir_input}/{spec}"
                            if not os.path.exists(
                                f"{dir_input_tmp}/flow_src_dst.npy"
                            ):
                                continue
                            flow_src_dst = np.load(
                                f"{dir_input_tmp}/flow_src_dst.npy"
                            )
                            stats = decode_dict(
                                np.load(
                                    f"{dir_input_tmp}/stats.npy",
                                    allow_pickle=True,
                                    encoding="bytes",
                                ).item()
                            )
                            n_flows_total = stats["n_flows"]
                            if (
                                n_flows_total < 5000000
                                and len(flow_src_dst) == n_flows_total
                            ):
                                data_list_test.append(
                                    (spec, (0, n_hosts - 1), topo_type_cur)
                                )
                                
            else:
                # 测试阶段读取之前保存的工作负载列表
                data_list = self.__read_data_list(self.dir_output)
                if self.test_on_train:
                    data_list_test = data_list["train"]
                else:
                    data_list_test = data_list["test"]
            self.test = self.__create_dataset(
                data_list_test,
                self.dir_input,
            )
            logging.info(f"#tracks: test-{len(data_list_test)}")

    def switch_to_other_epochs_logic(self):
        self.train.use_first_epoch_logic = False
        
    def train_dataloader(self):
        """
        Returns a PyTorch DataLoader for the training data.

        :return: A PyTorch DataLoader object.
        :rtype: torch.utils.data.DataLoader
        """

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=None
            if (not self.enable_context)
            else my_collate,
        )

    def val_dataloader(self):
        """
        Returns a PyTorch DataLoader for the validation set.

        :return: A PyTorch DataLoader object.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None
            if (not self.enable_context)
            else my_collate,
        )

    # Create test dataloader
    def test_dataloader(self):
        """
        Returns a PyTorch DataLoader object for the test dataset.

        :return: DataLoader object with test dataset
        :rtype: torch.utils.data.DataLoader
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None
            if (not self.enable_context)
            else my_collate,
        )

    def __random_split_list(self, lst, percentage):
        '''
        将数据集按照给定的比例随机分割为训练集和验证集
        '''
        length = len(lst)
        split_index = int(length * percentage / self.batch_size) * self.batch_size

        train_part = lst[:split_index]
        test_part = lst[split_index:]

        return train_part, test_part

    def __create_dataset(self, data_list, dir_input):
        #在这里加载src、dst主机、流大小等数据
        return PathDataset_Context(
            data_list,
            dir_input,
            lr=self.lr,
            bucket_thold=self.bucket_thold,
            enable_context=self.enable_context,
            flowsim_mean=self.flowsim_mean,
            flowsim_overall_variance=self.flowsim_overall_variance,
            mean=self.mean,
            overall_variance=self.overall_variance
        )

    def __dump_data_list(self, path):
        with open(f"{path}/data_list.json", "w") as fp:
            data_dict = {
                "train": self.train_list,
                "val": self.val_list,
                "test": self.val_list,
            }
            json.dump(data_dict, fp)

    def __read_data_list(self, path):
        f = open(f"{path}/data_list.json", "r")
        return json.loads(f.read())


class PathDataset_Context(Dataset):
    '''
    PathDataset_Context 是继承自 torch.utils.data.Dataset 的类，主要负责处理数据集的具体细节，
    包括如何从磁盘中加载数据、如何对数据进行预处理等。具体功能包括：

    1. 数据加载: 根据传入的 data_list 以及从磁盘中读取的 .npy 文件（例如 fsize.npy、flow_src_dst.npy 等）
        加载流量数据、主机对信息、流的完成时间等特征。
    2. 数据预处理: 它会对从文件中读取的数据进行计算，比如计算流完成时间 (FCT) 和理论完成时间，
        然后生成特征图（feature map），这些特征会在后续的模型训练中被使用。
    3. 返回数据: 它实现了 __getitem__ 方法，用于返回模型训练所需的具体数据，
        如 res（sizebucket_to_sldn 特征）、num_flows_per_cell（每个流量桶中的流数量）等。
    '''
    '''
    负责数据的组织与管理，划分训练/验证/测试集，创建数据加载器，并将数据传递给训练过程中的模型
    '''
    def __init__(
        self,
        data_list,
        dir_input,
        lr,
        bucket_thold,
        enable_context,
        flowsim_mean,
        flowsim_overall_variance,
        mean,
        overall_variance,
    ):
        self.data_list = data_list
        self.use_first_epoch_logic = True
        
        self.dir_input = dir_input
        self.lr = lr
        self.bucket_thold = bucket_thold
        self.enable_context = enable_context
        self.flowsim_mean = flowsim_mean
        self.flowsim_overall_variance = flowsim_overall_variance
        self.mean = mean
        self.overall_variance = overall_variance
        logging.info(
            f"call PathDataset_Context: bucket_thold={bucket_thold}, enable_context={enable_context}, data_list={len(data_list)}, use_first_epoch_logic={self.use_first_epoch_logic}"
        )
        bdp_dict_db = {}
        bdp_dict_db_output = {}
        # 根据主机数量生成大小桶列表
        # BDP: 带宽延迟积,MTU: 最大传输单元
        for n_hosts in [3,5,7]:
            # MTU=1000,BDP=10*MTU=10000
            BDP = 10 * MTU
            bdp_dict_db[n_hosts] = get_size_bucket_list(mtu=MTU, bdp=BDP)
            '''
            def get_size_bucket_list(mtu, bdp):
                return np.array(
                    [
                        mtu // 4,
                        mtu // 2,
                        mtu * 3 // 4,
                        mtu,
                        bdp // 5,
                        bdp // 2,
                        bdp * 3 // 4,
                        bdp,
                        5 * bdp,
                    ]
                )
            def get_size_bucket_list_output(mtu, bdp):
                return np.array([mtu, bdp, 5 * bdp])
            '''
            bdp_dict_db_output[n_hosts] = get_size_bucket_list_output(mtu=MTU, bdp=BDP)
        self.bdp_dict_db = bdp_dict_db
        self.bdp_dict_db_output = bdp_dict_db_output


    def __len__(self):
        return len(self.data_list)
    # 在训练的时候被调用
    def __getitem__(self, idx):
        # 在训练，验证，测试的时候会调用DataLoader来加载数据

        
        # 获取工作负载项的规格和目标主机对(spec, src_dst_pair_target, topo_type)：
        spec, src_dst_pair_target, topo_type = self.data_list[idx]
        # 将目标主机对转换为字符串，连接主机对的起点和终点
        src_dst_pair_target_str = "_".join([str(x) for x in src_dst_pair_target])
        
        # 加载仿真数据文件，包括流量大小、流的源与目的地址等
        dir_input_tmp = f"{self.dir_input}/{spec}"
        feat_path=f"{dir_input_tmp}/feat{topo_type}.npz"
        # 加载工作负载相关的流量数据，例如流量大小和主机对信息
        if 1:
            n_hosts = int(spec.split("_")[2][6:])
            
            size_bucket_list = self.bdp_dict_db[n_hosts]
            size_bucket_list_output = self.bdp_dict_db_output[n_hosts]

            # 进一步处理这些数据，生成仿真所需的特征图
            # 这里计算的是延迟和传输速率相关的数据
            # file.write("0 {} {} {} {} {} {} {} {} {} {} {}\n".format
            # (bfsz, fwin, enable_pfc, cc, dctcp_k, dcqcn_k_min, dcqcn_k_max, u_tgt, hpai, timely_t_low, timely_t_high))
            param_data = np.load(f"{dir_input_tmp}/param{topo_type}.npy")
            if param_data[3]==1.0:
                param_data=np.insert(param_data,4,0)
            else:
                param_data=np.insert(param_data,4,1)
                
            param_data=np.insert(param_data,0,[0,0,0])
            param_data[n_hosts//2-1]=1.0
            param_data[3]=BDP_DICT[n_hosts]/MTU
            #加载 流的索引信息
            fid=np.load(f"{dir_input_tmp}/fid{topo_type}.npy")
            #加载流的大小
            sizes_flowsim = np.load(f"{dir_input_tmp}/fsize.npy")
            #加载src和dst路径
            flow_src_dst_flowsim = np.load(f"{dir_input_tmp}/fsd.npy")
            #根据流的索引信息筛选相应的流数据
            sizes=sizes_flowsim[fid]
            flow_src_dst=flow_src_dst_flowsim[fid]
            # 保存了每个流的 流完成时间（FCT, Flow Completion Time） 信息
            fcts = np.load(f"{dir_input_tmp}/fct{topo_type}.npy")
            # 保存了每个流的 理论完成时间 信息
            i_fcts = np.load(f"{dir_input_tmp}/fct_i{topo_type}.npy")
            sldns = np.divide(fcts, i_fcts)

            fcts = np.log(fcts)
            fcts = ((fcts - self.mean) / np.sqrt(self.overall_variance))
            
            # 找出目标主机对的前景流和背景流索引
            '''
            前景流通常是根据特定条件（如特定的主机对）筛选出来的，是仿真中需要精确评估的目标流量(目标主机对的流)
            背景流是相对于前景流而言的其他流量
            '''
            # 通过逻辑运算符确定哪些是前景流，哪些是背景流
            flow_idx_target_flowsim = np.logical_and(
                flow_src_dst_flowsim[:, 0] == src_dst_pair_target[0],
                flow_src_dst_flowsim[:, 1] == src_dst_pair_target[1],
            )
            flow_idx_nontarget_flowsim=~flow_idx_target_flowsim
            # 确定哪些是背景流
            flow_idx_nontarget_internal_flowsim=np.logical_and(
                flow_src_dst_flowsim[:, 0] != src_dst_pair_target[0],
                flow_src_dst_flowsim[:, 1] != src_dst_pair_target[1],
            )

            # 计算流的传播延迟
            # 计算流经过的链路数量，这部分计算了每条流从源主机到目的主机之间所经过的主机数
            n_links_passed = abs(flow_src_dst_flowsim[:, 0] - flow_src_dst_flowsim[:, 1])+flow_idx_nontarget_flowsim+flow_idx_nontarget_internal_flowsim
            # 它是源主机和目的主机的链路延迟之和，表示从源主机到目的主机的传播延迟的基础值
            delay_comp=LINK_TO_DELAY_DICT[n_hosts][flow_src_dst_flowsim[:,0]]+LINK_TO_DELAY_DICT[n_hosts][flow_src_dst_flowsim[:,1]]
            # 计算传播延迟和传输延迟
            DELAY_PROPAGATION_PERFLOW = get_base_delay_pmn(
                sizes=sizes_flowsim, n_links_passed=n_links_passed, lr_bottleneck=self.lr,flow_idx_target=flow_idx_target_flowsim,flow_idx_nontarget_internal=flow_idx_nontarget_internal_flowsim
            )+delay_comp

            # 加载 flowsim 中的流完成时间并加上传播延迟
            fcts_flowsim = (
                np.load(f"{dir_input_tmp}/fct_flowsim.npy") + DELAY_PROPAGATION_PERFLOW
            )
            #标准归一化
            fcts_flowsim = np.log(fcts_flowsim - DELAY_PROPAGATION_PERFLOW)
            fcts_flowsim = ((fcts_flowsim - self.flowsim_mean) / np.sqrt(self.flowsim_overall_variance))
            #将fct进行对数化
            #fcts_flowsim = np.log(fcts_flowsim)
            #同时除以最大值
            #fcts_flowsim = np.divide(fcts_flowsim, self.max_fct_flowsim)
            
            # 总数据量（数据+报头）/链路带宽+ （传播延迟+传输延迟）
            i_fcts_flowsim = (
                sizes_flowsim + np.ceil(sizes_flowsim / MTU) * HEADER_SIZE
            ) * BYTE_TO_BIT / self.lr + DELAY_PROPAGATION_PERFLOW
            sldns_flowsim = np.divide(fcts_flowsim, i_fcts_flowsim)


            sldns_flowsim = np.clip(sldns_flowsim, a_max=None, a_min=1.0)
            # sldns_flowsim : 仿真流的时延比例
            # compute sldns from flowsim for each link
            #sldns_list = []
            fcts_list = []  # 用于存储 fcts_flowsim 的分组数据
            bins = []
            x_len = len(size_bucket_list) + 1
            y_len = len(P99_PERCENTILE_LIST)

            # 前景流的数据添加到列表中
            # sldns_flowsim_target: 目标主机对（前景流）对应的时延比例数据
            #sldns_flowsim_target = sldns_flowsim[flow_idx_target_flowsim]
            fcts_flowsim_target = fcts_flowsim[flow_idx_target_flowsim]
            #print("flow_idx_target_flowsim:",flow_idx_target_flowsim)
            fcts_list.append(fcts_flowsim_target)#是[array()]的列表
            #print("fcts_list:",fcts_list)
            # sldns_list: 保存前景流和背景流的时延比例数据，稍后会用于生成特征图
            #sldns_list.append(sldns_flowsim_target)
            # 这段代码的作用是将流量大小（sizes_flowsim[flow_idx_target_flowsim]）
            # 根据预定义的大小桶列表（size_bucket_list）进行分组
            bins_target = np.digitize(sizes_flowsim[flow_idx_target_flowsim], size_bucket_list)
            bins.append(bins_target)
            
            # 添加背景流
            # add the background traffic 
            if self.enable_context:
                for link_idx_internal in range(
                    src_dst_pair_target[0], src_dst_pair_target[1]
                ):
                    # 选择源主机编号小于或等于link_idx_internal的流
                    # 并且目的主机编号大于link_idx_internal的流
                    flow_idx_selected = np.logical_and(
                        flow_src_dst_flowsim[:, 0] <= link_idx_internal,
                        flow_src_dst_flowsim[:, 1] > link_idx_internal,
                    )
                    # 排除目标主机对（前景流）
                    flow_idx_selected = np.logical_and(flow_idx_selected, ~flow_idx_target_flowsim)
                    # 获取选中的流的大小
                    sizes_perlink = sizes_flowsim[flow_idx_selected]
                    #sldns_flowsim_perlink = sldns_flowsim[flow_idx_selected]
                    fcts_flowsim_perlink = fcts_flowsim[flow_idx_selected]
                    
                    #sldns_list.append(sldns_flowsim_perlink)
                    fcts_list.append(fcts_flowsim_perlink)
                    bins.append(np.digitize(sizes_perlink, size_bucket_list))
                    
            # 生成特征图（可能是前景流和后景流的结合）
            # n_sldns_list = len(sldns_list)  # 前景流和背景流的数量
            n_fcts_list = len(fcts_list)  # 前景流和背景流的数量
            # x_len: 大小桶的数量
            # y_len: 时延百分位数的数量
            sizebucket_to_fcts = np.zeros((n_fcts_list, x_len, y_len))  # 特征图存储
            num_flows_per_cell = np.zeros((n_fcts_list, x_len, y_len))  # 每个大小桶中的流量数量
            n_sizes_effective = np.ones((n_fcts_list, 1))  # 有效流数量
            #sizebucket_to_sldn = np.zeros((n_sldns_list, x_len, y_len))  # 初始化特征图存储，形状为 (流数量, 大小桶数量, 百分位数数量)
            #num_flows_per_cell = np.zeros((n_sldns_list, x_len, y_len))  # 每个大小桶中的流数量存储
            #n_sizes_effective = np.ones((n_sldns_list, 1))  # 有效流数量的存储，用于归一化

            # 遍历前景流和背景流，为每个流生成特征图
            for fcts_idx in range(n_fcts_list):
                # 如果当前流没有对应的大小桶，跳过
                if len(bins[fcts_idx]) == 0:
                    continue
                # 遍历每个大小桶
                for x_idx in range(x_len):
                    # 找到当前流中属于该大小桶的流量索引
                    fcts_idx_target = np.nonzero(bins[fcts_idx] == x_idx)[0]
                    # 如果该大小桶中的流量数少于阈值，则跳过
                    if len(fcts_idx_target) < self.bucket_thold:
                        continue
                    
                    # 提取该大小桶中的时延数据
                    fcts_tmp = fcts_list[fcts_idx][fcts_idx_target]

                    # 计算该大小桶中的时延百分位数（如50th, 99th），并保存到特征图
                    sizebucket_to_fcts[fcts_idx, x_idx] = np.percentile(
                        fcts_tmp, P99_PERCENTILE_LIST, method=PERCENTILE_METHOD
                    )

                    # 保存该大小桶中的流量数量
                    num_flows_per_cell[fcts_idx, x_idx] = len(fcts_idx_target)
                    # 更新有效流量数
                    n_sizes_effective[fcts_idx] += len(fcts_idx_target)

            # 将三维的特征图展平成二维数组，形状为 (流数量, 大小桶 * 百分位数)，用于模型输入
            res = sizebucket_to_fcts.reshape((n_fcts_list, -1)).astype(np.float32)

            # 同样将每个大小桶中的流量数量展平成二维数组
            num_flows_per_cell = num_flows_per_cell.reshape((n_fcts_list, -1)).astype(np.float32)

            # 对每个大小桶中的流量数量进行归一化处理
            num_flows_per_cell = np.divide(num_flows_per_cell, n_sizes_effective)

            # 寻找前景流的索引，即目标主机对的流量索引
            flow_idx_target = np.logical_and(
                flow_src_dst[:, 0] == src_dst_pair_target[0],  # 源主机匹配
                flow_src_dst[:, 1] == src_dst_pair_target[1]   # 目的主机匹配
            )

            # 生成前景流的真实特征图（ground truth feature map）
            fcts_output = fcts[flow_idx_target]  # 提取前景流的时延比例
            bins_output = np.digitize(sizes[flow_idx_target], size_bucket_list_output)  # 按照大小桶对流进行分组
            x_len_output = len(size_bucket_list_output) + 1  # 前景流的大小桶数量

            # 初始化前景流的特征图和流量数量存储
            sizebucket_to_fcts_output = np.ones((x_len_output, y_len))  # 真实的特征图初始化为全1
            num_flows_per_cell_output = np.zeros((x_len_output, y_len))  # 真实特征图中每个桶中的流量数量
            n_sizes_effective_output = 0  # 有效流量计数

            # 遍历每个大小桶，生成真实的特征图
            for x_idx in range(x_len_output):
                # 找到属于该大小桶的流量索引
                fcts_idx_target = np.nonzero(bins_output == x_idx)[0]
                # 如果该大小桶的流量数小于阈值，跳过
                if len(fcts_idx_target) < self.bucket_thold:
                    continue
                
                # 提取该大小桶中的时延数据
                fcts_tmp = fcts_output[fcts_idx_target]

                # 计算并保存该大小桶中的时延百分位数
                sizebucket_to_fcts_output[x_idx] = np.percentile(
                    fcts_tmp, P99_PERCENTILE_LIST, method=PERCENTILE_METHOD
                )

                # 保存该大小桶中的流量数量
                num_flows_per_cell_output[x_idx] = len(fcts_idx_target)
                # 更新有效流量计数
                n_sizes_effective_output += len(fcts_idx_target)

            # 将前景流的真实特征图展平成一维数组，形状为 (大小桶 * 百分位数)
            res_output = sizebucket_to_fcts_output.reshape((-1)).astype(np.float32)

            # 同样将前景流的每个大小桶中的流量数量展平成一维数组
            num_flows_per_cell_output = num_flows_per_cell_output.reshape((-1)).astype(np.float32)

            # 对前景流每个大小桶中的流量数量进行归一化
            num_flows_per_cell_output = np.divide(num_flows_per_cell_output, n_sizes_effective_output)

            # [size_bucket, percentile]
            n_input = n_fcts_list  # 输入的大小桶数量
            
            #assert (res>=0).all()

            # 将仿真任务的参数插入到特征图的末尾
            res = np.insert(res, res.shape[1], param_data[:, None], axis=1)  # 在特征图中添加仿真参数
            
            np.savez(feat_path, res=res, num_flows_per_cell=num_flows_per_cell, res_output=res_output, num_flows_per_cell_output=num_flows_per_cell_output,n_input=n_input)
        else:
            print("feat_path:",feat_path,"oioioioioi\n\n")
            data = np.load(feat_path)
            
            res = data["res"]
            num_flows_per_cell = data["num_flows_per_cell"]
            res_output = data["res_output"]
            num_flows_per_cell_output = data["num_flows_per_cell_output"]
            n_input = data["n_input"]


        #print("max_fct_flowsim:",max_fct_flowsim)
        #res = res / max_fct_flowsim  # 归一化
        return (
            res,                            # 特征图 (foreground 和 background 流的 size bucket 和时延分布百分位数)
            num_flows_per_cell,             # 每个特征图桶中流的数量，用于描述流量分布的密度
            res_output,                     # 前景流（目标主机对）的真实特征图，作为网络训练的目标
            num_flows_per_cell_output,      # 前景流中每个桶的流量数量
            spec,                           # 工作负载规格的字符串表示，标识仿真实验参数，如 shard、流量、主机数量等
            n_input,                        # 前景流和背景流的总数，表示特征图的数量
            src_dst_pair_target_str         # 目标主机对的字符串表示，用于唯一标识某个流的通信对
        )

