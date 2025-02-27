#include "my_lib.h"
#include "topo.h"

using namespace std;

   


// 读取流的函数
std::unordered_map<int, Flow> readFlowsAsMap(const std::string &filename) {
    std::unordered_map<int, Flow> flows_map;
    std::ifstream infile(filename);
    std::string line;
    int path_id = 0;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        Flow flow;
        std::string path_str;

        // 按顺序解析流信息: ID,src,dst,size,fat,path
        if (!(iss >> flow.flowID >> flow.src >> flow.dst >> flow.size >> flow.fat)) {
            std::cerr << "Error reading flow data\n";
            continue;
        }

        // 读取路径信息
        if (!(iss >> path_str)) {
            std::cerr << "Error reading path data\n";
            continue;
        }

        // 解析路径（逗号分隔），并将其转换为节点对
        std::istringstream path_stream(path_str);
        std::string node;
        std::vector<int> nodes; // 临时存储节点序列

        while (std::getline(path_stream, node, ',')) {
            nodes.push_back(std::stoi(node)); // 将节点字符串转换为整数
        }
        vector<pair<int,int>> one_path;  // 从源到目的的路径（节点ID的序列）
        // 将节点序列转换为节点对
        for (size_t i = 1; i < nodes.size(); ++i) {
            one_path.emplace_back(nodes[i - 1], nodes[i]); // 添加节点对
        }
        flow.path_id = path_id;
        all_path_list.push_back(one_path);
        path_id++;

        // 插入到 unordered_map 中
        flows_map[flow.flowID] = flow;
    }

    return flows_map;
}

// 定义函数，将 path_list 构造为 channel_to_flowid_map
ChannelToFlowIDMap constructChannelToFlowIDMap(
    const std::vector<std::vector<std::pair<int, int>>>& path_list,
    const std::unordered_map<int, Flow>& vec_flows
) {
    ChannelToFlowIDMap channel_to_flowid_map;

    // 遍历每一条路径
    for (size_t flow_idx = 0; flow_idx < path_list.size(); ++flow_idx) {
        const auto& path = path_list[flow_idx];

        // 遍历路径中的每一段链路
        for (const auto& link : path) {
            channel_to_flowid_map[link].push_back(flow_idx);
        }
    }

    return channel_to_flowid_map;
}

// 定义函数，将 vec_flows 转换为 path_list
std::vector<std::vector<std::pair<int, int>>> convertToPathList(const std::unordered_map<int, Flow>& vec_flows) {
    std::vector<std::vector<std::pair<int, int>>> path_list;

    // 遍历 vec_flows
    for (auto it = vec_flows.begin(); it != vec_flows.end(); ++it) {
        const Flow& flow = it->second; // 当前流
        const std::vector<pair<int, int>>& path = all_path_list[flow.path_id]; // 当前流的路径
        std::vector<std::pair<int, int>> path_pairs;

        // 构造节点对
        for (size_t i = 0; i < path.size(); ++i) {
            path_pairs.emplace_back(path[i]);
        }

        // 添加到 path_list
        if (!path_pairs.empty()) {
            path_list.push_back(path_pairs);
        }
    }

    return path_list;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}


void get_fct_mmf(unsigned int n_flows, Flow* flows, int h, int *topo, int method_mmf, int method_routing, int type_topo, int base_lr)
{
    assert (type_topo==PL);  
    assert (method_routing==PL_ECMP_ROUTING);
    if (method_mmf==PL_TWO_LAYER){
        long long int BW[2]; 
        for (int i = 0; i < 2; i++)
            BW[i] = topo[i] * ((long long int)base_lr);
        pl_topology_init_two_layer(h, BW);
        pl_routing_init_two_layer();
    }
    else if (method_mmf==PL_ONE_LAYER){
        long long int BW[2];
        for (int i = 0; i < 2; i++)
            BW[i] = topo[i] * ((long long int)base_lr);
        pl_topology_init_one_layer(h, BW);
        pl_routing_init_one_layer();
    }
    else{
        assert(false);
    }

    double t = 0.0;
    unsigned int j = 0;
    unsigned int t_index = 0;
    unsigned int *active_flows_idx = (unsigned int *)malloc(n_flows * sizeof(unsigned int));
    double *t_flows = (double *)malloc((2 * n_flows) * sizeof(double));
    unsigned int *num_flows = (unsigned int *)malloc((2 * n_flows) * sizeof(unsigned int));
    unsigned int *num_flows_enq = (unsigned int *)malloc((n_flows) * sizeof(unsigned int));
    // double lr = 10.0;
 
    memset(num_flows, 0, 2 * n_flows * sizeof(unsigned int));
    memset(num_flows_enq, 0, n_flows * sizeof(unsigned int));
    // double a_nan = strtod("NaN", NULL);
    double time_to_next_arrival = NAN;
    double time_to_next_completion = NAN;
    unsigned int num_active_flows = 0;
    double sum_weights = 0.0;
    int min_remaining_time_index = -1;

    int *src_active = (int *)malloc(n_flows * sizeof(int));
    int *dst_active = (int *)malloc(n_flows * sizeof(int));

    while (true)
    {
        if (j < n_flows)
        {
            time_to_next_arrival = flows[j].fat - t;//原本这里的start_time是fat
            // printf("time_to_next_arrival:%f\n", time_to_next_arrival);
            assert(time_to_next_arrival >= 0);
        }
        else
        {
            time_to_next_arrival = NAN;
        }
        min_remaining_time_index = -1;
        if (num_active_flows)
        {
            update_rate_mmf(num_active_flows, src_active, dst_active, method_mmf, type_topo);

            time_to_next_completion = INFINITY;
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                double remaining_time = flows[flow_idx].remaining_size / final_flow_vector[i];
                if (remaining_time < time_to_next_completion)
                {
                    time_to_next_completion = remaining_time;
                    min_remaining_time_index = i;
                }
            }
        }
        else
        {
            time_to_next_completion = NAN;
        }

        if (num_active_flows > 0 && (j >= n_flows || time_to_next_completion <= time_to_next_arrival))
        {
            // Completion Event
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                flows[flow_idx].fct_est += time_to_next_completion;
                flows[flow_idx].remaining_size -= time_to_next_completion * final_flow_vector[i];
            }
            t += time_to_next_completion;
            num_active_flows -= 1;
            assert(min_remaining_time_index != -1);
            active_flows_idx[min_remaining_time_index] = active_flows_idx[num_active_flows];
            src_active[min_remaining_time_index] = src_active[num_active_flows];
            dst_active[min_remaining_time_index] = dst_active[num_active_flows];
        }
        else
        {
            // Arrival Event
            if (j >= n_flows)
            {
                // No more flows left - terminate
                break;
            }
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                flows[flow_idx].fct_est += time_to_next_arrival;
                flows[flow_idx].remaining_size -= time_to_next_arrival * final_flow_vector[i];
            }
            t += time_to_next_arrival;
            flows[j].remaining_size = (flows[j].size + ceil(flows[j].size/ MTU) * HEADER_SIZE) * BYTE_TO_BIT;
            flows[j].fct_est = 0.0;
            // active_flows[j].remaining_size = sizes[j] * 8.0;
            active_flows_idx[num_active_flows] = j;
            src_active[num_active_flows] = flows[j].src;
            dst_active[num_active_flows] = flows[j].dst;
            num_active_flows += 1;
            num_flows_enq[j] = num_active_flows;
            j += 1;
        }
        if (method_mmf==PL_TWO_LAYER) {
            pl_reset_topology_two_layer();
        }
        else if (method_mmf==PL_ONE_LAYER) {
            pl_reset_topology_one_layer();
        }
        else{
            assert(false);
        }
        t_flows[t_index] = t;
        num_flows[t_index] = num_active_flows;
        t_index += 1;
        // if (j % 100000 == 0)
        // {
        //     printf("%d/%d simulated in seconds\n", j, n_flows);
        // }
    }

    free(active_flows_idx);
    free(t_flows);
    free(num_flows);
    free(num_flows_enq);
    free(src_active);
    free(dst_active);
}



 
void config_link_to_delay(int n_hosts,double *link_to_delay,float *param_list){
    if (n_hosts==3){
        param_list[0]=1.0;
        param_list[3]=5.0;
        // link_to_delay[1]=2*DELAY_PROP_BASE;
    }
    else if (n_hosts==5){
        param_list[1]=1.0;
        param_list[3] = 10.0;
        // link_to_delay[1]=2*DELAY_PROP_BASE;
        link_to_delay[2]=1*DELAY_PROP_BASE;
        // link_to_delay[3]=2*DELAY_PROP_BASE;
    }
    else if (n_hosts==7){
        param_list[2]=1.0;
        param_list[3] = 15.0;
        // link_to_delay[1]=2*DELAY_PROP_BASE;
        link_to_delay[2]=1*DELAY_PROP_BASE;
        link_to_delay[3]=2*DELAY_PROP_BASE;
        link_to_delay[4]=1*DELAY_PROP_BASE;
        // link_to_delay[5]=2*DELAY_PROP_BASE;
    }
    

}



void calculate_and_save_percentiles(Flow* flows, int n_flows, double buckets[], int n_buckets,
                                    double percentiles[], int n_percentiles, float *feat_map,int pos_start,int bucket_thold, float *const_opt) {
    int feat_index = 0;
    //buckets:  250 500 750 1000 2000 5000 7500 10000 50000
    //cout<<"buckets: ";for(int i=0;i<9;i++)cout<<" "<<buckets[i];cout<<endl;
    // qsort(flows, n_flows, sizeof(Flow), compare_flow_sldn);

    // float *sldn_total = (float *)calloc(n_percentiles, sizeof(float));
    
    // for (int i = 0; i < n_percentiles; ++i) {
    //     sldn_total[i] = 1.0;
    // }

    // int bucket_start=0;
    // int bucket_end=n_flows;
    // for (int k = 0; k < n_percentiles; ++k) {
    //     double flow_idx_thres = bucket_start + ((bucket_end - bucket_start-1) * percentiles[k] / 100);
    //     if (flow_idx_thres == (int)flow_idx_thres) {
    //         // Exact index, no interpolation needed
    //         sldn_total[feat_index++] = flows[(int)flow_idx_thres].sldn;
    //     } else {
    //         // Interpolate between two adjacent values
    //         int lower_index = (int)flow_idx_thres;
    //         int upper_index = lower_index + 1;

    //         // double lower_value = flows[lower_index].sldn;
    //         // double upper_value = flows[upper_index].sldn;

    //         // sldn_total[feat_index++]= flows[upper_index].sldn;
            
    //         double fraction = flow_idx_thres - lower_index;

    //         // sldn_total[feat_index++] =  lower_value + fraction * (upper_value - lower_value);
    //         if (fraction >=0.5){
    //             sldn_total[feat_index++]= flows[upper_index].sldn;
    //         }
    //         else{
    //             sldn_total[feat_index++]= flows[lower_index].sldn;
    //         }
            
    //     }
    // }
    // sldn_total[0]=0.5;
    qsort(flows, n_flows, sizeof(Flow), compare_flow_size);
    for(int i=0;i< n_flows;i++){
        cout<<"flows["<<i<<"]:"<<flows[i].size<<endl;
    }
    // feat_index = 0;
    // 初始化桶边界
    int* bucket_starts = (int*)malloc(n_buckets * sizeof(int));
    int* bucket_ends = (int*)malloc(n_buckets * sizeof(int));
    std::cout << "n_buckets: " << n_buckets <<endl;
    if (bucket_starts == NULL || bucket_ends == NULL) {
        std::cerr << "Error: Memory allocation failed for buckets." << std::endl;
        exit(EXIT_FAILURE);
    }

    memset(bucket_starts, 0, n_buckets * sizeof(int));
    memset(bucket_ends, 0, n_buckets * sizeof(int));

    
    // Initialize bucket boundaries
    // 划分桶的范围，计算每个桶的起始和结束索引
    for (int i = 0; i < n_buckets; ++i) {
        // 检查当前桶是否包含流
        if (i == 0 && flows[0].size < buckets[i]) {
            bucket_starts[i] = 0;
            bucket_ends[i] = 0;
        } else if (bucket_starts[i] != -1) {
            bucket_ends[i] = bucket_starts[i];
        }
        if(i==n_buckets-1){bucket_starts[i]=bucket_ends[i-1];bucket_ends[i]=n_flows-1;break;}

        // 调试信息：打印当前桶的起始和结束索引
        // std::cout << "Processing bucket " << i << ":" << std::endl;
        // std::cout << "  bucket_starts[" << i << "] = " << bucket_starts[i] << std::endl;
        // std::cout << "  bucket_ends[" << i << "] = " << bucket_ends[i] << std::endl;
        // std::cout << "  buckets[" << i << "] = " << buckets[i] << std::endl;

        // 查找当前桶的结束索引
        while (1) {
                // 打印调试信息
                //std::cout << "i: " << i << std::endl;
                //std::cout << "bucket_ends[" << i << "]: " << bucket_ends[i] << std::endl;
                //if (bucket_ends[i] >= 0 && bucket_ends[i] < n_flows) {
                //    std::cout << "flows[bucket_ends[" << i << "]].size: " << flows[bucket_ends[i]].size << std::endl;
                //} else {
                //    std::cout << "bucket_ends[" << i << "] is out of bounds for flows array." << std::endl;
                //}
//
                if(bucket_ends[i] >= n_flows || flows[bucket_ends[i]].size > buckets[i])break;
                //// 调试信息：打印当前流的索引和大小
                //std::cout << "  Inspecting flow at index " << bucket_ends[i] << ":" << std::endl;
                //std::cout << "    flows[" << bucket_ends[i] << "].size = " << flows[bucket_ends[i]].size << std::endl;

                // 检查数组是否越界（防御性检查）
                //if (bucket_ends[i] < 0 || bucket_ends[i] >= n_flows) {
                //    std::cerr << "Error: bucket_ends[" << i << "] is out of bounds: " << bucket_ends[i] << std::endl;
                //    std::cerr << "  n_flows = " << n_flows << std::endl;
                //    exit(EXIT_FAILURE);
                //}
//
                //if (flows[bucket_ends[i]].size >= buckets[i]) {
                //    std::cerr << "Flow size exceeds bucket limit:" << std::endl;
                //    std::cerr << "  flows[" << bucket_ends[i] << "].size = " << flows[bucket_ends[i]].size << std::endl;
                //    std::cerr << "  buckets[" << i << "] = " << buckets[i] << std::endl;
                //    exit(EXIT_FAILURE);
                //}

            // 增加结束索引
            ++bucket_ends[i];
        }

        // 调试信息：打印更新后的结束索引
        //std::cout << "  Updated bucket_ends[" << i << "] = " << bucket_ends[i] << std::endl;

        // 设置下一个桶的起始索引
        if (i < n_buckets - 1 && bucket_ends[i] < n_flows) {
            bucket_starts[i + 1] = bucket_ends[i];
            // 调试信息：打印下一个桶的起始索引
            //std::cout << "  Setting bucket_starts[" << i + 1 << "] = " << bucket_starts[i + 1] << std::endl;
        }
    }

    


    if (bucket_starts[n_buckets-1]!=-1){
        bucket_ends[n_buckets-1] = n_flows;
    }
    
    // 跳过空桶或流数量少于 bucket_thold 的桶。
	// 对空桶或过小的桶，直接填充默认值 const_opt
    for (int i = 0; i < n_buckets; ++i) {
        int bucket_start = bucket_starts[i];
        int bucket_end = bucket_ends[i];

        if (bucket_start == -1 || abs(bucket_end-bucket_start)<bucket_thold) {
            // No flows in this bucket
            // feat_index+=n_percentiles;
            for (int j=0;j<n_percentiles;j++){
                feat_map[pos_start+feat_index++] = const_opt[j];
            }
            // feat_map[pos_start+feat_index-1] = 0;
            continue;
        }
        // printf("Sort Bucket %d: %d\n", i, bucket_end-bucket_start);
        // Resort flows within the current bucket based on completion time
        qsort(&flows[bucket_start], bucket_end - bucket_start, sizeof(Flow), compare_flow_sldn);

        // if(i==0){
        //     for (int j = 0; j < bucket_end-bucket_start; ++j) {
        //         printf("%d:%f ", flows[bucket_start+j].flowId, flows[bucket_start+j].size);
        //     }
        //     printf("\n");
        // }
        // Calculate percentiles for the flows in the current bucket
        for (int k = 0; k < n_percentiles; ++k) {
            double flow_idx_thres = bucket_start + ((bucket_end - bucket_start-1) * percentiles[k] / 100);
            if (flow_idx_thres == (int)flow_idx_thres) {
                // Exact index, no interpolation needed
                feat_map[pos_start+feat_index++] = flows[(int)flow_idx_thres].sldn;
            } else {
                // Interpolate between two adjacent values
                int lower_index = (int)flow_idx_thres;
                int upper_index = lower_index + 1;

                // double lower_value = flows[lower_index].sldn;
                // double upper_value = flows[upper_index].sldn;

                // feat_map[pos_start+feat_index++] = flows[upper_index].sldn;

                double fraction = flow_idx_thres - lower_index;

                // feat_map[pos_start+feat_index++] =  lower_value + fraction * (upper_value - lower_value);
                if (fraction >=0.5){
                    feat_map[pos_start+feat_index++]= flows[upper_index].sldn;
                }
                else{
                    feat_map[pos_start+feat_index++]= flows[lower_index].sldn;
                }
                
            }
        }
        // feat_map[pos_start+feat_index-n_percentiles]=1.0;
        // feat_map[pos_start+feat_index-1]=log(bucket_end - bucket_start);
    }
    // for (int j=0;j<n_percentiles;j++){
    //     feat_map[pos_start+feat_index++] = sldn_total[j];
    // }
    free(bucket_starts);
    free(bucket_ends);
    // free(sldn_total);
}
// Compare function for sorting FlowData based on flow size
int compare_flow_size(const void *a, const void *b) {
    double diff = (*(Flow*)a).size - (*(Flow*)b).size;

    // Use a small epsilon to account for rounding errors
    if (fabs(diff) < 1e-9) {
        return 0;  // Values are considered equal
    }

    return (diff > 0) ? 1 : -1;
}

int compare_flow_sldn(const void *a, const void *b) {
    float diff = (*(Flow*)a).sldn - (*(Flow*)b).sldn;

    // Use a small epsilon to account for rounding errors
    if (fabs(diff) < 1e-9) {
        return 0;  // Values are considered equal
    }

    return (diff > 0) ? 1 : -1;
}
void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}
void build_transformer(Transformer *t, char* ckpt_path_llama) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(ckpt_path_llama, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}
void read_checkpoint(char* checkpoint, Config* config, ModelWeightsTransformer* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    printf("checkpoint: %s\n", checkpoint);
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = static_cast<float*>(mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0));
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}
void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = static_cast<float*>(calloc(p->dim, sizeof(float)));
    s->xb = static_cast<float*>((calloc(p->dim, sizeof(float))));
    s->xb2 = static_cast<float*>(calloc(p->dim, sizeof(float)));
    s->hb = static_cast<float*>(calloc(p->hidden_dim, sizeof(float)));
    s->hb2 = static_cast<float*>(calloc(p->hidden_dim, sizeof(float)));
    s->q = static_cast<float*>(calloc(p->dim, sizeof(float)));
    s->key_cache = static_cast<float*>(calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)));
    s->value_cache = static_cast<float*>(calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)));
    s->att = static_cast<float*>(calloc(p->n_heads * p->seq_len, sizeof(float)));
    s->logits = static_cast<float*>(calloc(p->dim, sizeof(float)));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}
void build_mlp(char* ckpt_path_mlp, ConfigMLP *config_mlp, ModelWeightsMLP *weights_mlp, RunStateMLP *state_mlp) {
    // read in the Config and the Weights from the checkpoint
    // 加载多层感知机（MLP）模型权重的函数。它从一个二进制检查点文件（checkpoint file）中
    // 读取模型配置和权重数据，并将这些数据映射到内存中以供后续操作使用
    int fd = 0;
    float *data = NULL;
    long file_size;

    FILE *file = fopen(ckpt_path_mlp, "rb");
    if (!file){ fprintf(stderr, "Couldn't open file %s\n", ckpt_path_mlp); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config_mlp, sizeof(ConfigMLP), 1, file) != 1) { exit(EXIT_FAILURE); }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    file_size = ftell(file);  // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    fd = open(ckpt_path_mlp, O_RDONLY); // open in read only mode
    if (fd == -1){ fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    data = static_cast<float*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float *weights_ptr = data + sizeof(ConfigMLP) / sizeof(float);
    memory_map_weights_mlp(weights_mlp,config_mlp, weights_ptr);
    // create and init the application RunState
    malloc_run_state_mlp(state_mlp, config_mlp);
}
void memory_map_weights_mlp(ModelWeightsMLP *w, ConfigMLP *p, float *f)
{
    float *ptr = f;
    w->w1 = ptr;
    ptr += p->input_dim * p->hidden_dim_1 + p->hidden_dim_1;

    w->w2 = ptr;
    ptr += p->hidden_dim_1 * p->hidden_dim_2 + p->hidden_dim_2;

    w->w3 = ptr;
    ptr += p->output_dim * p->hidden_dim_2 + p->output_dim;

    w->const_opt = ptr;
    ptr += p->y_len;
}
void malloc_run_state_mlp(RunStateMLP *s, ConfigMLP *p)
{
    // we calloc instead of malloc to keep valgrind happy
    s->x = static_cast<float*>(calloc(p->input_dim, sizeof(float)));
    s->h1 = static_cast<float*>(calloc(p->hidden_dim_1, sizeof(float)));
    s->h2 = static_cast<float*>(calloc(p->hidden_dim_2, sizeof(float)));
    s->logits = static_cast<float*>(calloc(p->output_dim, sizeof(float)));
    // ensure all mallocs went fine
    if (!s->x || !s->h1 || !s->h2 || !s->logits)
    {
        printf("malloc failed!\n");
        exit(1);
    }
}
void memory_map_weights(ModelWeightsTransformer *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim + p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void generate(Transformer *transformer, float* feat_map, int n_hosts, int n_feat_input,int n_feat_context, float* feat_concat) {
    // start the main loop
    // long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int pos = 0;     // position in the sequence
    float *feat_input = (float*)malloc((n_feat_input) * sizeof(float));
    float* feat_start = feat_map + n_feat_input;
    while (pos < n_hosts-1) {
        memcpy(feat_input, feat_start, n_feat_input*sizeof(*feat_input));
        // printf("logits-%d: %f, %f\n ",pos, feat_input[0],feat_input[n_feat_input-2]);
        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, feat_input,pos);
        // printf("logits-%d: %f, %f\n ",pos, logits[0],logits[n_feat_context-1]);
        feat_start+=n_feat_input;
        pos++;
        for (int i = 0; i < n_feat_context; i++) {
            feat_concat[n_feat_input+i] += logits[i];
        }

    }
    for (int i = 0; i < n_feat_context; ++i) {
        feat_concat[n_feat_input+i] /= pos;
    }
    // printf("\n");
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    // long end = time_in_ms();
    // fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    free(feat_input);
}
void matmul_with_bias(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val + w[d * n + i];
        // xout[i] = val;
    }
}

void forward_MLP(float *x, ConfigMLP *p, ModelWeightsMLP *w, RunStateMLP *s)
{
// #pragma omp parallel for
    matmul_with_bias(s->h1, x, w->w1, p->input_dim, p->hidden_dim_1);
    for (int i = 0; i < p->hidden_dim_1; i++)
    {
        if (s->h1[i]<0.0){
            s->h1[i]=0.0;
        }
    }

    matmul_with_bias(s->h2, s->h1, w->w2, p->hidden_dim_1, p->hidden_dim_2);
    for (int i = 0; i < p->hidden_dim_2; i++)
    {
        if (s->h2[i]<0.0){
            s->h2[i]=0.0;
        }
    }

    matmul_with_bias(s->logits, s->h2, w->w3, p->hidden_dim_2, p->output_dim);
    
    //ReLU activation function for the output layer
    // for (int i = 0; i < p->output_dim; i++)
    // {
    //     if (s->logits[i]<0.0){
    //         s->logits[i]=0.0;
    //     }
    // }

    // No activation function for the output layer
    // for (int i = 0; i < p->output_dim; i++)
    // {
    //     s->logits[i] =(1.0f / (1.0f + exp(-s->logits[i])));
    // }
}
void write_vecs_to_file(const char *filename, float *vec_1, int vec_size_1, float *vec_2, int vec_size_2) {
    FILE *file = fopen(filename, "w");
    
    if (file == NULL) {
        // Handle file opening error
        perror("Error opening file");
        return;
    }
    for (int i = 0; i < vec_size_1; ++i) {
        fprintf(file, "%lf ", vec_1[i]+1.0);
    }
    fprintf(file, "\n");
    for (int i = 0; i < vec_size_2; ++i) {
        fprintf(file, "%lf ", vec_2[i]);
    }
    fclose(file);
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
} 

void free_run_state_mlp(RunStateMLP *s)
{
    free(s->x);
    free(s->h1);
    free(s->h2);
    free(s->logits);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

float* forward(Transformer* transformer, float* feat,int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    ModelWeightsTransformer* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    // printf("feat-%d: %f, %f\n ",pos, feat[0],feat[p->vocab_size-2]);
    // printf("weight-%d: %f, %f\n ",pos, w->token_embedding_table[0],w->token_embedding_table[p->vocab_size*p->dim-1]);
    // printf("bias-%d: %f, %f\n ",pos, w->token_embedding_table[p->vocab_size*p->dim],w->token_embedding_table[p->vocab_size*p->dim+p->dim-1]);
    matmul_with_bias(x, feat, w->token_embedding_table, p->vocab_size, p->dim);
    // printf("token_embedding_table-%d: %f, %f\n ",pos, x[0],x[p->dim-1]);
    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->dim);
    return s->logits;
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize 
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


void writeSampledPathsToFile(const std::vector<std::vector<std::pair<int, int>>>& sampled_paths, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    for (const auto& path : sampled_paths) {
        // 遍历每条路径中的点对
        for (size_t i = 0; i < path.size(); ++i) {
            outFile << path[i].first << "," << path[i].second;
            // 如果不是最后一个点对，添加分隔符
            if (i < path.size() - 1) {
                outFile << ";";
            }
        }
        outFile << "\n";  // 每条路径结束后换行
    }

    outFile.close();
}

void writeFlowsToFile(const Flow* flows, int n_flows, const string& filename) {
    // 打开文件以写入
    std::ofstream outfile(filename,std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return;
    }
    //写入n_flows
    outfile<<"n_flows: "<<n_flows<<endl;
    // 遍历所有流并将 fct_est 和 remaining_size 写入文件
    for(int i=0;i<10;i++)outfile<<endl;
    for (int i = 0; i < n_flows; ++i) {
        outfile << "Flow " << i << ": fct_est = " << flows[i].fct_est 
                << ", remaining_size = " << flows[i].remaining_size << std::endl;
    }

    // 关闭文件
    outfile.close();
    std::cout << "成功将 " << n_flows << " 条流的信息写入文件 " << filename << std::endl;
}