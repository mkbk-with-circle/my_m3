#include "my_lib.h"


using namespace std;


vector<vector<pair<int,int>>> all_path_list;




int main(int argc, char *argv[]) {
    char *ckpt_path_llama = NULL;  // LLaMA模型的检查点路径，例如 "out/model.bin"
    char *ckpt_path_mlp = NULL;    // MLP模型的检查点路径，例如 "out/model.bin"
    char *data_path = NULL;        // 数据路径，用于存储流量模拟数据，例如 "/data1/.../flows.txt"
    char data_path_input[256];     // 输入数据文件路径
    char data_path_output[256];    // 输出数据文件路径

    char mode[] = "generate";      // 模式，可能的值为 "generate"（生成）或 "chat"（对话模式）

    int bw = 10;                   // 链路基础带宽，单位为Gbps
    float bw_list[2];              // 带宽列表，存储两个节点之间的带宽，基于 `topo` 和 `bw` 计算
    int topo[2] = {1, 4};          // 网络拓扑中两个节点的编号
    for (int i = 0; i < 2; i++)
        bw_list[i] = (float) topo[i] * bw;
    double link_to_delay[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
                               // 链路到延迟的映射数组，每条链路的传输延迟，单位为秒

    int n_size_bucket_input = 10;  // 输入大小分桶的数量
    int n_size_bucket_output = 4;  // 输出大小分桶的数量
    int n_percentiles = 100;       // 计算的百分位数个数
    double percentiles[100];       // 百分位数组，表示从1%到100%对应的百分位数值
    for (int i = 0; i < n_percentiles; i++)
        percentiles[i] = i+1;


        
    int n_param = 19;              // 参数数量，通常表示模型的输入特征数量
    float bfsz = 20;               // 缓存大小，单位为KB
    float fwin = 18000;            // 流控制窗口大小，单位为字节
    int n_hosts;                   // 主机数量，表示网络中的终端设备数量
    float enable_pfc = 1.0;        // 是否启用PFC（优先级流控），1.0 表示启用，0 表示禁用
    float cc = 0;                  // 拥塞控制算法的选择参数
    float param_1 = 0.0;           // 其他模型参数1
    float param_2 = 0.0;           // 其他模型参数2
    int n_embd;                    // 嵌入向量的维度，用于上下文特征表示

    double size_bucket_input[9] = {MTU/4.0, MTU/2.0, MTU*3.0/4.0, MTU, BDP/5.0, BDP/2.0, BDP*3.0/4.0, BDP, 5.0*BDP}; 
                                   // 输入大小桶的分布，单位为字节。
                                   // 根据 MTU（最大传输单元）和 BDP（带宽时延积）计算。

    
                               // 参数列表，包括缓存大小、窗口大小等模型参数
    ModelWeightsMLP weights_mlp;
    //一些run.c的参数 
    // ./run ../ckpts/model_llama{model_suffix}.bin ../ckpts/model_mlp{model_suffix}.bin {data_dir} -b 10 -e 576 -n {n_hosts} -t 1 -f {bfsz} -k {window} -p {enable_pfc} -c {cc} -x {param_1} -y {param_2} > {data_dir}/output.txt 2>&1
        //./run 
        // 0 ../ckpts/model_llama{model_suffix}.bin 
        // 1 ../ckpts/model_mlp{model_suffix}.bin {data_dir} 
        // 3 -b 10 带宽
        // 5 -e 576 
        // 7 -n {n_hosts} (path_length)
        // 9 -t 1 
        // 11 -f {bfsz} 
        // 13 -k {window} 
        // 15 -p {enable_pfc} 
        // 17 -c {cc} 
        // 19 -x {param_1} 
        // 21 -y {param_2} > {data_dir}/output.txt 2>&1
    int rng_seed = 10;             // 随机数种子，用于生成可重复的随机数
    int bucket_thold = 1;          // 桶的阈值，用于流量分桶时的最小值
    if (argc >= 4) { 
        printf("argv[1]: %s\n", argv[1]);
        printf("argv[2]: %s\n", argv[2]);
        printf("argv[3]: %s\n", argv[3]);
        ckpt_path_llama = argv[1];
        ckpt_path_mlp = argv[2];
        data_path = argv[3];
        
        // 构造输入路径
        snprintf(data_path_input, sizeof(data_path_input), "%s/flows.txt", data_path);

        // 构造输出路径
        snprintf(data_path_output, sizeof(data_path_output), "%s/fct_mlsys.txt", data_path);
    
        // 如果需要验证结果
        //std::cout << "ckpt_path_llama: " << ckpt_path_llama << std::endl;
        //std::cout << "ckpt_path_mlp: " << ckpt_path_mlp << std::endl;
        //std::cout << "data_path_input: " << data_path_input << std::endl;
        //std::cout << "data_path_output: " << data_path_output << std::endl;
    }else { error_usage(); }
    for (int i = 4; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }  // 配置随机数种子（rng_seed）
        else if (argv[i][1] == 'b') { bw = atoi(argv[i + 1]); }  // 配置带宽（bw）
        else if (argv[i][1] == 'e') { n_embd = atoi(argv[i + 1]); }  // 配置嵌入参数数量（n_embd）
        else if (argv[i][1] == 'n') { n_hosts = atoi(argv[i + 1]); }  // 配置主机数量（n_hosts）
        else if (argv[i][1] == 't') { bucket_thold = atoi(argv[i + 1]); }  // 配置桶的阈值（bucket_thold）

        else if (argv[i][1] == 'f') { bfsz = atof(argv[i + 1]); }  // 配置缓冲区大小（bfsz）
        else if (argv[i][1] == 'k') { fwin = atof(argv[i + 1]); }  // 配置窗口大小（fwin）
        else if (argv[i][1] == 'p') { enable_pfc = atof(argv[i + 1]); }  // 配置是否启用优先级流控（enable_pfc）
        else if (argv[i][1] == 'c') { cc = atof(argv[i + 1]); }  // 配置拥塞控制参数（cc）
        else if (argv[i][1] == 'x') { param_1 = atof(argv[i + 1]); }  // 配置参数1（param_1）
        else if (argv[i][1] == 'y') { param_2 = atof(argv[i + 1]); }  // 配置参数2（param_2）
        else { error_usage(); }
    }
    int n_feat_input = n_size_bucket_input * n_percentiles; //1000
                                   // 输入特征的数量，等于输入桶数量乘以百分位数数量
    int n_feat_map = (n_feat_input + n_param) * n_hosts; 
                                   // 特征映射的总维度，基于输入特征、参数和主机数量计算
    int n_feat_context = n_embd;   // 上下文特征的维度，与嵌入向量维度一致
    int n_feat_concat = n_feat_input + n_param + n_embd; 
                                   // 输入、参数和上下文特征的总维度（拼接后的）
    int n_feat_output = n_size_bucket_output * n_percentiles; 
                                   // 输出特征的总维度，等于输出桶数量乘以百分位数数量

    int src_dst_pair_target[2] = {0, n_hosts - 1}; 
                                   // 源目标对的索引，用于表示网络中流量的起点和终点

    
    
    float param_list[19] = {0, 0, 0, 0, bfsz, fwin / (float)1000.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; 
    //输入 
    //Step1: 读取流的工作负载,// ID,src,dst,size,fat，序号到流的映射
    std::unordered_map<int, Flow> vec_flows = readFlowsAsMap("flow.txt");
    //path_list
    std::vector<std::vector<std::pair<int, int>>> path_list = convertToPathList(vec_flows);

    //Topology topology = readTopology("topology.txt");
    ChannelToFlowIDMap channel_to_flowid_map = constructChannelToFlowIDMap(path_list, vec_flows);

    // 哈希表储存路径到若干条流的映射
    unordered_map<vector<pair<int,int>>, vector<int>, VectorPairHash> path_to_flowids_map;


    for (std::unordered_map<int, Flow>::iterator it = vec_flows.begin(); it != vec_flows.end(); ++it) {
        int flow_id = it->first;       // 获取 flow_id
        Flow flow = it->second;        // 获取 flow
        // 使用 flow.path 和 flow_id进行映射，表示路径到流的映射
        path_to_flowids_map[all_path_list[flow.path_id]].push_back(flow_id);
    }
    
    //Step2: 根据流数量加权采样若干条路径，并去重
    time_t start = time(nullptr);

    vector<vector<pair<int,int>>> paths;
    vector<int> weights;

    // 收集路径和权重
    // paths 是vector<vector<int>>，表示所有路径
    // weights 是vector<int>，表示每条路径的权重（流的数量）
    for (auto it = path_to_flowids_map.begin(); it != path_to_flowids_map.end(); ++it) {
        const vector<pair<int,int>> &path = it->first;
        const vector<int> &flow_ids = it->second;
        paths.push_back(path);
        weights.push_back(flow_ids.size());
    }
    //加权随机采样
    mt19937 rng(42); // 固定种子保证结果一致
        //收集权重
    discrete_distribution<> dist(weights.begin(), weights.end());
        //收集采样的路径
    vector<vector<pair<int,int>>> sampled_paths;
    const int NUM_SAMPLES = 500; // 采样次数
        //开始采样路径
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        int sampled_index = dist(rng);
        sampled_paths.push_back(paths[sampled_index]);
    }
    // Step 4: 去重并排序
    writeSampledPathsToFile(sampled_paths, "sampled_paths.txt");
    sort(sampled_paths.begin(), sampled_paths.end(), [&](const vector<pair<int, int>> &a, const vector<pair<int, int>> &b) {
        // 比较路径长度，优先按长度降序排序
        if (a.size() != b.size()) {
            return a.size() > b.size(); // 长度从大到小
        }
        // 如果路径长度相同，则按流数量降序排序
        return path_to_flowids_map.at(a).size() > path_to_flowids_map.at(b).size();
    });

    // 去重
    sampled_paths.erase(unique(sampled_paths.begin(), sampled_paths.end()), sampled_paths.end());

    //double work_time=difftime(time(nullptr),start);
    //cout << "Elapsed time for sampling: " << work_time << " seconds\n";

    // Step 5: 开始枚举，对每一个路径上的流进行操作
    int size_sampled_path=sampled_paths.size();
    time_t start_path = time(nullptr);
    for(int idx = 0; idx < size_sampled_path ; ++idx ){
        // Step 6: 逐个处理，将完整的流进行分段
        const std::vector<pair<int,int>>& path = sampled_paths[idx]; 
        time_t start_each = time(nullptr); 
        // Flowset : unordered_set<int>
        FlowSet flow_ids_in_f_prime;//记录有流经当前路径上某一链路的流ID
        // PathSegmentMap : unordered_map<int, vector<pair<int, int>>>
        PathSegmentMap flow_to_srcdst_map_in_flowsim;//从流id映射用vector储存的每一段路径段起终点

        int path_length = 1;
        int path_l = path.size();   

        // 遍历路径，逐段更新信息
        for (int i = 1; i < path.size(); ++i) {
            pair<int, int> src_dst_pair = path[i];

            // 获取路径段上的流 ID
            unordered_map<pair<int, int>, vector<int>, PairHash>::iterator flows_on_path_it = channel_to_flowid_map.find(src_dst_pair);
            if (flows_on_path_it != channel_to_flowid_map.end()) {
                const vector<int> &flows_on_path = flows_on_path_it->second;

                // 更新路径段上的流集合
                flow_ids_in_f_prime.insert(flows_on_path.begin(), flows_on_path.end());

                // 更新路径段信息
                for (int key_flowid : flows_on_path) {
                    vector<pair<int, int>> &count_vec = flow_to_srcdst_map_in_flowsim[key_flowid];
                    if (!count_vec.empty() && count_vec.back().second == path_length - 1) {
                        count_vec.back().second = path_length; // 更新路径段终点
                    } else {
                        count_vec.emplace_back(path_length - 1, path_length); // 新路径段
                    }
                }
            }
            path_length += 1; // 更新路径长度
        }

        //从流ID集合 flow_ids_in_f_prime(vector<int>)中提取流FLow，
        vector<Flow> flows_remaining;
        for (int flow_id : flow_ids_in_f_prime) {
            auto flow_it = vec_flows.find(flow_id); // flows 是存储流信息的映射
            if (flow_it != vec_flows.end()) {
                flows_remaining.push_back(flow_it->second);
            }
        }
        // 根据路径段扩展流信息
        vector<Flow> flow_extra;
        // 遍历所有的FLOW数据，全部进行分段
        for (auto &flow : flows_remaining) {
            // 找到某一条流id对应的路径段的向量集合
            // count_vec_it 指向 flow_to_srcdst_map_in_flowsim 中的一个键值对
            auto count_vec_it = flow_to_srcdst_map_in_flowsim.find(flow.flowID);
            // 如果找到了
            if (count_vec_it != flow_to_srcdst_map_in_flowsim.end()) {
                // count_vec 是路径段的vector，每个路径段是一个 std::pair<int, int>
                const auto &count_vec = count_vec_it->second;

                // 更新流的起点和终点为路径段的第一个起点和终点
                flow.src = count_vec[0].first;
                flow.dst = count_vec[0].second;

                // 如果路径段有多个部分，扩展流
                // 某些背景流会多次扩展路径
                for (size_t i = 1; i < count_vec.size(); ++i) {
                    Flow tmp = flow; // 拷贝当前流
                    tmp.src = count_vec[i].first;
                    tmp.dst = count_vec[i].second;
                    tmp.flowID = flow_extra.size() + NR_FLOWS; // 给扩展的流一个新的 ID
                    flow_extra.push_back(tmp);
                }
            }
        }
        // 将 flow_extra 的内容追加到 flows_remaining
        flows_remaining.insert(flows_remaining.end(), flow_extra.begin(), flow_extra.end());

        // 按照流的 start 属性排序
        std::sort(flows_remaining.begin(), flows_remaining.end(), [](const Flow &a, const Flow &b) {
            return a.fat < b.fat;
        });
        time_t finish_each = time(nullptr);
        cout << "Elapsed time for sampling: " << difftime(finish_each ,start_each) << " seconds\n";
        

        // Step 7 :分段结束之后，对每条路径调用mlsys（run.c 1164开始 ）

        // 原rust代码中，因为需要调用c，所以先将流信息保存到flow.txt里边了
        // 之后又读取（在其中加上3 100可能是为了保护数据安全）读取完之后又删掉了
        // 总之就是可以省略一大部分的中间过程
        // 但是还有一部分是读取参数，这部分还未实现

        // 源代码中读取流信息的时候还给每个流的idx赋值了，以及bittleneck
        int n_flows = flows_remaining.size();
        Flow *flows = (Flow *)malloc(n_flows * sizeof(Flow));
        for(int i=0;i< n_flows;i++){
            flows[i]=flows_remaining[i];
            flows[i].idx=i;
            flows[i].bw_bottleneck = bw_list[0];
        }

        long start = time_in_ms();
        // 正式开始flowsim生成FCT
        n_hosts = path_length;
        n_feat_map = (n_feat_input + n_param) * n_hosts; 
        config_link_to_delay(n_hosts,link_to_delay,param_list);
        //dctcp的配置
        param_list[8]=1.0;
        param_list[12]=param_1;
        
        //./run 
        // 0 ../ckpts/model_llama{model_suffix}.bin 
        // 1 ../ckpts/model_mlp{model_suffix}.bin {data_dir} 
        // 3 -b 10 
        // 5 -e 576 
        // 7 -n {n_hosts} (path_length)
        // 9 -t 1 
        // 11 -f {bfsz} 
        // 13 -k {window} 
        // 15 -p {enable_pfc} 
        // 17 -c {cc} 
        // 19 -x {param_1} 
        // 21 -y {param_2} > {data_dir}/output.txt 2>&1
        //解析可选命令行参数，更新运行配置参数（随机种子、带宽、嵌入向量大小、主机数、阈值、缓存大小等）。

        //主要修改了flow.fct_est和flows.remaining_size
        get_fct_mmf(n_flows,flows,path_length,topo,PL_ONE_LAYER, PL_ECMP_ROUTING, PL, bw);
        //把每个流修改后的fct_est和remaining_size输出到文件sampled_paths.txt
        writeFlowsToFile(flows, n_flows, "sampled_paths.txt");
        cout << "get_fct_mmf complete" << endl;
        long end = time_in_ms();
        //printf( "maxmin-fair: %fs\n",(double)(end-start)/1000);
        printf("check the get_fct_mmfoutput\n");
        //Check the output
        for (int i = 0; i < n_flows; ++i) {
            printf("Flow %d: %lf %lf %lf\n",
                   flows[i].flowID, flows[i].remaining_size,flows[i].fct_est, flows[i].sldn);
        }
        printf("check the get_fct_mmfoutput end\n");


        // 计算：基础延迟=传播延迟+排队延迟+处理延迟
        // 每个流的流理论完成时间
        double base_delay=0.0;
        double pkt_header=0.0;
        double i_fct=0.0;
        int n_link=1;
        bool flow_idx_target=false;
        bool flow_idx_nontarget_spec=false;
        bool bottle_link_per_flow=false;
        printf("n_flows: %d\n", n_flows);
        fflush(stdout);
        for (int i = 0; i < n_flows; ++i) {
            // 计算数据包头的大小，考虑最小值和最大传输单元 (MTU)
            pkt_header = flows[i].size < MTU ? flows[i].size : MTU;
            // 数据包头加上 HEADER_SIZE 转换为比特
            pkt_header = (pkt_header + HEADER_SIZE) * BYTE_TO_BIT;

            // 计算流经过的链路数量，取源和目标节点之间的绝对差值
            n_link = abs(static_cast<int>(flows[i].src - flows[i].dst));
            
            // 判断流是否为目标流（源节点为 0 且目标节点为 n_hosts-1）
            flow_idx_target = (flows[i].src == 0) && (flows[i].dst == n_hosts - 1);
            
            // 判断流是否为非目标流（源和目标节点均不为 0 和 n_hosts-1）
            flow_idx_nontarget_spec = (flows[i].src != 0) && (flows[i].dst != n_hosts - 1);

            // 如果不是目标流，链路数量增加 1
            if (!flow_idx_target) {
                n_link += 1;
            }
            // 如果是非目标流且符合特定条件，链路数量再增加 1
            if (flow_idx_nontarget_spec) {
                n_link += 1;
            }

            // 计算基础延迟，包含：
            // - 数据包头传输时间在每条链路上的影响
            // - 固定的传播延迟
            // - 源节点和目标节点的特定延迟
            base_delay = pkt_header / flows[i].bw_bottleneck / 4.0 * (n_link - 2) +
                        DELAY_PROP_BASE * n_link +
                        link_to_delay[flows[i].src] +
                        link_to_delay[flows[i].dst];

            // 如果是目标流，增加额外延迟
            if (flow_idx_target) {
                base_delay += pkt_header / flows[i].bw_bottleneck;
            }

            // 如果是非目标流，减少部分延迟
            if (flow_idx_nontarget_spec) {
                base_delay -= pkt_header / flows[i].bw_bottleneck;
            }

            // 计算理论流完成时间（流大小加上头部开销的比特量除以瓶颈带宽）
            i_fct = (flows[i].size + ceil(flows[i].size / MTU) * HEADER_SIZE) * BYTE_TO_BIT / flows[i].bw_bottleneck;

            // 计算标准化流完成时间 (SLDN)
            flows[i].sldn = (flows[i].fct_est + base_delay) / (i_fct + base_delay);

            // 调试输出，打印关键变量
            printf("\n\n");
            printf("Flow ID: %u\n", flows[i].flowID);
            printf("Src: %u, Dst: %u\n", flows[i].src, flows[i].dst);
            printf("Size: %lf, bw_bottleneck: %f\n", flows[i].size, flows[i].bw_bottleneck);
            printf("Pkt Header: %lf, Base Delay: %lf, i_fct: %lf\n", pkt_header, base_delay, i_fct);
            printf("SLDN: %lf\n", flows[i].sldn);
            printf("\n\n");
            fflush(stdout); // 确保输出立即刷新到终端

            // 检查 SLDN 是否大于等于 1.0，如果不符合，程序中止
            assert(flows[i].sldn >= 1.0);
        }
        //此段计算了flows[i].sldn,



        // load_model
        
        // build the Transformer via the model .bin file
        Transformer transformer;
        build_transformer(&transformer, ckpt_path_llama);
    
        // load MLP 
        ConfigMLP config_mlp;
        ModelWeightsMLP weights_mlp;
        RunStateMLP state_mlp;
        build_mlp(ckpt_path_mlp, &config_mlp, &weights_mlp, &state_mlp);
        printf("const_opt: %f, %f\n", weights_mlp.const_opt[0], weights_mlp.const_opt[n_percentiles-1]);

        printf("\nconfig: %d,%d,%d,%d\n", config_mlp.input_dim, config_mlp.hidden_dim_1, config_mlp.hidden_dim_2, config_mlp.output_dim);
        printf("MLP loaded!\n");
        // 动态分配feat_map
        printf("n_feat_map: %d\n", n_feat_map);
        printf("n_feat_concat: %d\n", n_feat_concat);
        printf("now allocate feat_map, size: %d\n", n_feat_map);
        float *feat_map = (float *)malloc(n_feat_map * sizeof(float));
        float *feat_concat = (float *)calloc(n_feat_concat, sizeof(float));
        // 初始化feat_map
        for (int i = 0; i < n_feat_map; ++i) {
            feat_map[i] = 0.0;
        }

        // 动态分配n_flows_per_link
        int *n_flows_per_link=(int *)malloc(n_hosts * sizeof(int));
        for (int i = 0; i < n_hosts; ++i) {
            n_flows_per_link[i] = 0;
        }

        
        // 遍历flows并统计流，n_flows_per_link用于统计每个主机（host）作为 链路起点 时，与其相关的流（flow）的数量
        for (int i = 0; i < n_flows; i++) {
            if (flows[i].src == src_dst_pair_target[0] && flows[i].dst == src_dst_pair_target[1]) {
                n_flows_per_link[n_hosts - 1]++;
            } else {
                for (int j = flows[i].src; j < flows[i].dst; j++) {
                    n_flows_per_link[j]++;
                }
            }
        }


    
        // 输出结果（可选）
        for (int i = 0; i < n_hosts; i++) {
            std::cout << "Host " << i << ": " << n_flows_per_link[i] << " flows" << std::endl;
        }
    
        
        int feat_pos = 0;
        // 初始化特征位置索引，用于标记特征映射 `feat_map` 的写入位置。

        int n_flows_fg = n_flows_per_link[n_hosts - 1];
        // 获取目标流（前景流，Foreground flows）的数量。
        // 目标流是指从源 `src_dst_pair_target[0]` 到目标 `src_dst_pair_target[1]` 的流。

        int* flow_ids_bg;
        // 指向背景流（Background flows）的 ID 的数组指针，用于存储非目标流的索引。

        int n_flows_bg = n_flows - n_flows_fg;
        // 计算背景流的数量，即总流数量减去目标流数量。
        cout<<"creating flow_ids_bg: n_flows_bg: "<<n_flows_bg<<endl;
        flow_ids_bg = (int *)malloc(n_flows_bg * sizeof(int));
        if (flow_ids_bg == NULL&&n_flows_bg>0) {
            std::cerr << "Error: Memory allocation failed for flow_ids_bg." << std::endl;
            exit(EXIT_FAILURE); // 或者采取其他错误处理措施
        }
        // 为背景流 ID 分配内存，大小为背景流数量乘以每个整数的大小。

        Flow* flows_fg;
        // 指向目标流（前景流）的数组指针，用于存储目标流的信息。
        cout<<"creating flows_fg: n_flows_fg: "<<n_flows_fg<<endl;
        if (n_flows_fg>0){
            flows_fg = (Flow *)malloc(n_flows_fg * sizeof(Flow));
        }
        // 如果存在目标流，为目标流分配内存，大小为目标流数量乘以每个流结构的大小。

        int index_fg = 0;
        // 初始化目标流索引，用于存储目标流到 `flows_fg`。

        int index_bg = 0;
        // 初始化背景流索引，用于存储背景流到 `flow_ids_bg`。

        for (int i = 0; i < n_flows; ++i) {
            if (flows[i].src == src_dst_pair_target[0] && flows[i].dst == src_dst_pair_target[1]) {
                flows_fg[index_fg++] = flows[i];
            }
            else {
                flow_ids_bg[index_bg++] = i;
            }
        }
        // 遍历所有流：
        // - 如果流是目标流，将其存储到 `flows_fg` 中，并更新目标流索引 `index_fg`。
        // - 否则，将流的索引存储到 `flow_ids_bg` 中，并更新背景流索引 `index_bg`。

        if (n_flows_fg > 0) {
            // 如果存在目标流，计算它们的特征，并将结果存储到 `feat_map` 中。
            printf("start calculate_and_save_percentiles, n_flows_fg: %d\n , n_size_bucket_input: %d\n", n_flows_fg , n_size_bucket_input);
            calculate_and_save_percentiles(
                flows_fg, // 目标流数组
                n_flows_fg, // 目标流数量
                size_bucket_input, // 桶大小（输入）
                n_size_bucket_input, // 桶数量（输入）
                percentiles, // 百分位点
                n_percentiles, // 百分位点数量
                feat_map, // 特征映射数组
                feat_pos, // 当前特征写入位置
                bucket_thold, // 桶阈值
                weights_mlp.const_opt // 权重优化参数
            );

            // 释放分配的目标流数组内存
            free(flows_fg);
        }

        for (int i = 0; i < n_param; i++) {
            feat_map[feat_pos + n_feat_input + i] = param_list[i];
        }
        // 将参数 `param_list` 中的内容追加到 `feat_map` 中，
        // 从 `feat_pos + n_feat_input` 开始写入，写入长度为 `n_param`。
        // printf("feat-input-%d: %d, %lf, %lf, %lf\n", feat_pos, n_flows_fg, feat_map[feat_pos], feat_map[feat_pos+n_feat_input-1], feat_map[feat_pos+n_feat_input]);
    
        int flow_id_bg;
        for (int linkid_idx = 0; linkid_idx < n_hosts-1; ++linkid_idx) {
            //根据链路索引获取前景流数量 n_flows_fg
            n_flows_fg=n_flows_per_link[linkid_idx];
            index_fg = 0;
            feat_pos=(linkid_idx+1)*(n_feat_input+n_param);
            if (n_flows_fg>0){
                cout<<"n_flows_fg: "<<n_flows_fg<<endl;
                //为前景流分配内存
                flows_fg = (Flow *)malloc(n_flows_fg * sizeof(Flow));//2
                if(flows_fg==NULL){
                    std::cerr << "Error: Memory allocation failed for flows_fg." << std::endl;
                    exit(EXIT_FAILURE); // 或者采取其他错误处理措施
                }
                // Count the number of flows satisfying the condition
                // 遍历背景流，统计满足条件的流
                // 如果某条流的源节点小于等于当前链路索引且目标节点大于当前链路索引，将其拷贝到 flows_fg
                for (int i = 0; i < n_flows_bg; ++i) {
                    flow_id_bg=flow_ids_bg[i];
                    cout<<"flow_id_bg: "<<flow_id_bg<<endl;//0
                    if (flows[flow_id_bg].src <= linkid_idx && flows[flow_id_bg].dst > linkid_idx) {
                        cout<<"now index_fg: "<<index_fg<<endl;//0
                        cout.flush();
                        Flow tmp=flows[flow_id_bg];
                        flows_fg[index_fg] = tmp;
                        index_fg++;
                    }
                }
                // calcuate the feature map
                printf("start calculate_and_save_percentiles, n_flows_fg: %d , n_size_bucket_input: %d\n", n_flows_fg , n_size_bucket_input);
                calculate_and_save_percentiles(flows_fg, n_flows_fg, size_bucket_input, n_size_bucket_input, percentiles, n_percentiles, feat_map,feat_pos,bucket_thold,weights_mlp.const_opt);
                
                free(flows_fg);
            }
            // for(int i=0; i<n_feat_input; i++){
            //     feat_map[feat_pos+i]-=1.0;
            // }
            for(int i=0; i<n_param; i++){
                feat_map[feat_pos+n_feat_input+i]=param_list[i];
            }
            // printf("feat_input-%d: %d, %lf, %lf, %lf\n", linkid_idx+1,n_flows_fg, feat_map[feat_pos], feat_map[feat_pos+n_feat_input-1], feat_map[feat_pos+n_feat_input]);
        }
        end = time_in_ms();
        printf( "feat-map: %fs\n",(double)(end-start)/1000);
        
        
        //调用模型
        
        if (strcmp(mode, "generate") == 0) {
            for (int i = 0; i < n_feat_input+n_param; ++i) {
                feat_concat[i] = feat_map[i];
            }
            generate(&transformer, feat_map, n_hosts,n_feat_input+n_param,n_feat_context,feat_concat);
            // printf("feat_mlp %lf, %lf, %lf, %lf\n", feat_concat[0], feat_concat[n_feat_input], feat_concat[n_feat_input+n_param], feat_concat[n_feat_input+n_param+n_feat_context-1]);
    
            end = time_in_ms();
            printf( "transformer: %fs\n",(double)(end-start)/1000);
            start = time_in_ms();
    
            //run mlp
            forward_MLP(feat_concat, &config_mlp, &weights_mlp, &state_mlp);
    
            float *feat_output = state_mlp.logits;
    
            end = time_in_ms();
            printf( "mlp: %fs\n",(double)(end-start)/1000);
            start = time_in_ms();
    
            printf("feat_output: %lf, %lf\n", feat_output[0], feat_output[n_feat_output-1]);
            write_vecs_to_file(data_path_output, feat_output, n_feat_output,feat_concat,n_feat_concat);
    
        } else {
            fprintf(stderr, "unknown mode: %s\n", mode);
            error_usage();
        }
        end = time_in_ms();
        // memory and file handles cleanup
        free(flows);
        free(feat_map);
        free(feat_concat);
        free(n_flows_per_link);
        free(flow_ids_bg);
        free_transformer(&transformer);
        free_run_state_mlp(&state_mlp);
    }
    cout<<"complete";
}
