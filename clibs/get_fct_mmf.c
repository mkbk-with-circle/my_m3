#include "topo.h"
struct FCTStruct
{
    double *estimated_fcts;
    double *t_flows;
    unsigned int *num_flows;
    unsigned int *num_flows_enq;
};

typedef struct
{
    double remaining_size;
} Flow;

struct FCTStruct get_fct_mmf(unsigned int n_flows, double *fats, double *sizes, int *src, int *dst, int h, int *topo, int method_mmf, int method_routing, int type_topo, int base_lr);
void free_fctstruct(struct FCTStruct input_struct);
void update_rate_mmf(unsigned int traffic_count, int *src, int *dst, int method_mmf, int type_topo);

void update_rate_mmf(unsigned int traffic_count, int *src, int *dst, int method_mmf, int type_topo)
{
    int iteration_count = 0;
    double exec_time = 0.0;
    
    pl_ppf_from_array(traffic_count, src, dst, &iteration_count, &exec_time);
      
    // long long int tot_mmf_bw=0;
    // for(int i=0;i<traffic_count;i++){
    //     //fprintf(ofd, "%lld\n", final_flow_vector[i]);
    //     tot_mmf_bw+=final_flow_vector[i];
    // }

    // printf("\nAggregate throughput for method %d is %8.6lf\n %d\t %8.6lf (s)\n", method_mmf, tot_mmf_bw*1.0 / traffic_count, iteration_count, exec_time);

    // int i;
    // printf("final_flow_vector = [");
    // for (i = 0; i < traffic_count - 1; i++)
    //     printf("%f, ", final_flow_vector[i]);
    // printf("%f]\n", final_flow_vector[i]);
    // printf("%d\t %8.6lf (s)\n", iteration_count, exec_time);
    // printf("%d\t %d\n", iteration_count, traffic_count);
}

// res = C_LIB.get_fct_mmf(n_flows, fats_pt, sizes_pt, src_pt, dst_pt, nhost, topo_pt, 1, 8, 2, bw)
struct FCTStruct get_fct_mmf(unsigned int n_flows, double *fats, double *sizes, int *src, int *dst, int h, int *topo, int method_mmf, int method_routing, int type_topo, int base_lr)
{
    // 确保传入的拓扑类型为停车场拓扑（PL），并且使用 ECMP 路由方法
    assert (type_topo == PL);
    assert (method_routing == PL_ECMP_ROUTING);
    
    // 根据 method_mmf 确定是使用两层或一层拓扑，这里是一层
    if (method_mmf == PL_TWO_LAYER) {
        long long int BW[2];  // 定义带宽数组
        // 为每一层分配带宽
        for (int i = 0; i < 2; i++)
            BW[i] = topo[i] * ((long long int)base_lr);
        pl_topology_init_two_layer(h, BW);  // 初始化两层拓扑
        pl_routing_init_two_layer();  // 初始化两层路由
    }
    else if (method_mmf == PL_ONE_LAYER) {
        long long int BW[2];  // 定义带宽数组
        for (int i = 0; i < 2; i++)
            BW[i] = topo[i] * ((long long int)base_lr);
        pl_topology_init_one_layer(h, BW);  
        // 初始化一层拓扑
        pl_routing_init_one_layer();  // 初始化一层路由
    }
    else {
        assert(false);  // 如果 method_mmf 不匹配，程序停止
    }

    // printf("n_flows: %u\n", n_flows);
    // printf("fats:");
    // for (int i = 0; i < n_flows; i++)
    // {
    //     printf("%f ", fats[i]);
    // }
    // printf("\n");
    // printf("sizes:");
    // for (int i = 0; i < n_flows; i++)
    // {
    //     printf("%u ", sizes[i]);
    // }
    // printf("\n");
    // printf("weights:");
    // for (int i = 0; i < n_flows; i++)
    // {
    //     printf("%f ", weights[i]);
    // }
    // printf("\n");
    /*
    定义 FCTStruct 结构体存储结果
    res：用于存储计算结果。
    t：当前的仿真时间。
    j：当前处理的流量索引。
    t_index：时间索引。
    active_flows：存储活动流的数组。
    active_flows_idx：存储活动流的索引。
    estimated_fcts：存储每个流的估计完成时间。
    t_flows 和 num_flows：存储流量和活动流的信息。
    num_flows_enq：存储每个流量到达时的活动流数量。
    time_to_next_arrival：下一个流量的到达时间。
    time_to_next_completion：下一个流量的完成时间。
    num_active_flows：当前活跃的流量数量。
    min_remaining_time_index：剩余时间最少的流的索引。
    src_active 和 dst_active：存储活动流的源和目的地址
    */
    struct FCTStruct res;
    double t = 0.0;  // 初始化当前时间
    unsigned int j = 0;  // 当前处理的流索引
    unsigned int t_index = 0;  // 时间索引
    Flow *active_flows = (Flow *)malloc(n_flows * sizeof(Flow));  // 活动流的数组
    unsigned int *active_flows_idx = (unsigned int *)malloc(n_flows * sizeof(unsigned int));  // 活动流的索引数组
    double *estimated_fcts = (double *)malloc(n_flows * sizeof(double));  // 估计的流完成时间
    double *t_flows = (double *)malloc((2 * n_flows) * sizeof(double));  // 流时间数组
    unsigned int *num_flows = (unsigned int *)malloc((2 * n_flows) * sizeof(unsigned int));  // 活动流的数量
    unsigned int *num_flows_enq = (unsigned int *)malloc((n_flows) * sizeof(unsigned int));  // 每个流到达时的活动流数量
    // double lr = 10.0;

    memset(estimated_fcts, 0.0, n_flows * sizeof(double));
    memset(num_flows, 0, 2 * n_flows * sizeof(unsigned int));
    memset(num_flows_enq, 0, n_flows * sizeof(unsigned int));
    // double a_nan = strtod("NaN", NULL);
    double time_to_next_arrival = NAN;  // 下一个流的到达时间
    double time_to_next_completion = NAN;  // 下一个流的完成时间
    unsigned int num_active_flows = 0;  // 当前的活动流数量
    double sum_weights = 0.0;  // 权重和
    int min_remaining_time_index = -1;  // 最小剩余时间流的索引

    int *src_active = (int *)malloc(n_flows * sizeof(int));
    int *dst_active = (int *)malloc(n_flows * sizeof(int));

    while (true)
    {
        if (j < n_flows)
        {
            // 表示计算流的到达时间差，类似于算法中的优先级队列
            time_to_next_arrival = fats[j] - t;
            // printf("time_to_next_arrival:%f\n", time_to_next_arrival);
            assert(time_to_next_arrival >= 0);
        }
        else
        {
            time_to_next_arrival = NAN;// 没有更多流到达
        }
        min_remaining_time_index = -1;
        // 检查当前是否有活动流
        if (num_active_flows)
        {   
            // 更新活动流的速率，计算公平分配的速率
            update_rate_mmf(num_active_flows, src_active, dst_active, method_mmf, type_topo);
            // 初始化到下一个流完成事件的时间为无穷大
            time_to_next_completion = INFINITY;
            // 遍历所有活动流，找到剩余时间最短的流
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                double remaining_time = active_flows[flow_idx].remaining_size / final_flow_vector[i];
                 // 更新最短完成时间
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
        // 判断是完成事件还是完成事件
        if (num_active_flows > 0 && (j >= n_flows || time_to_next_completion <= time_to_next_arrival))
        {
            // Completion Event
            /*
            时间更新：更新当前的仿真时间到到达事件的时间点。
            活动流管理：将新流添加到活动流列表中，并记录其源地址和目的地址。
            初始化剩余大小：为新流初始化剩余的数据传输大小，这取决于流的大小和协议开销。
            计算速率分配：新流加入后会触发速率分配的重新计算，考虑新流对资源的占用            
            */
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                estimated_fcts[flow_idx] += time_to_next_completion;
                active_flows[flow_idx].remaining_size -= time_to_next_completion * final_flow_vector[i];
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
            /*
            时间更新：将当前时间更新到完成事件的时间点。
            数据传输更新：根据当前速率更新完成事件前所有活动流的剩余数据传输量。
            活动流移除：从活动流列表中移除已完成的流。
            重新计算速率分配：移除完成的流后，重新分配资源，确保剩余的活动流占用合适的带宽。
            */
            if (j >= n_flows)
            {
                // No more flows left - terminate
                break;
            }
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                estimated_fcts[flow_idx] += time_to_next_arrival;
                active_flows[flow_idx].remaining_size -= time_to_next_arrival * final_flow_vector[i];
            }
            t += time_to_next_arrival;
            active_flows[j].remaining_size = (sizes[j] + ceil(sizes[j] / 1000.0) * 48.0) * 8.0;
            // active_flows[j].remaining_size = sizes[j] * 8.0;
            active_flows_idx[num_active_flows] = j;
            src_active[num_active_flows] = src[j];
            dst_active[num_active_flows] = dst[j];
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

    res.estimated_fcts = estimated_fcts;
    res.t_flows = t_flows;
    res.num_flows = num_flows;
    res.num_flows_enq = num_flows_enq;
    free(active_flows_idx);
    free(src_active);
    free(dst_active);
    free(active_flows);
    // free(estimated_fcts);
    // free(t_flows);
    // free(num_flows);
    // free(num_flows_enq);
    return res;
}

void free_fctstruct(struct FCTStruct input_struct)
{
    free(input_struct.estimated_fcts);
    free(input_struct.t_flows);
    free(input_struct.num_flows);
    free(input_struct.num_flows_enq);
}

int main(int argc, char *argv[])
{

    int method_routing = atoi(argv[1]);
    int method_mmf = atoi(argv[2]);
    int type_topo = atoi(argv[3]);

    int l = 5;
    int topo[2] = {1, 4};
    long long int BW[2];

    for (int i = 0; i < 2; i++)
        BW[i] = topo[i] * ((long long int)10);

    if (method_mmf==PL_TWO_LAYER){
        pl_topology_init_two_layer(l, BW);
        pl_routing_init_two_layer();
    }
    else if (method_mmf==PL_ONE_LAYER){
        pl_topology_init_one_layer(l, BW);
        pl_routing_init_one_layer();
    }
    else{
        assert(false);
    }
    
    
    unsigned int num_scenarios = atoi(argv[4]);

    for (int i = 0; i < num_scenarios; i++)
    {
        unsigned int num_active_flows = atoi(argv[5 + i]);

        // int* src_active = (int *) malloc(sizeof(int) * num_active_flows);
        // for (int i=0; i<num_active_flows; i++)
        //     src_active[i] = i%totPE;

        // int* dst_active = (int *) malloc(sizeof(int) * num_active_flows);
        // for (int i=0; i<num_active_flows; i++)
        //     dst_active[i] = (totPE-1-i)%totPE;

        // int array1[6] = {0, 1, 2, 3, 0, 1};
        // int(*src_active)[6] = &array1;

        // int array2[6] = {4, 4, 4, 4, 1, 2};
        // int(*dst_active)[6] = &array2;
        int array1[6] = {0, 1, 1, 1, 2, 3};
        int(*src_active)[6] = &array1;

        int array2[6] = {4, 2, 2, 3, 3, 4};
        int(*dst_active)[6] = &array2;

        update_rate_mmf(num_active_flows, src_active, dst_active, method_mmf, type_topo);
        pl_reset_topology_two_layer();
        if (method_mmf==PL_TWO_LAYER){
            pl_reset_topology_two_layer();
        }
        else if (method_mmf==PL_ONE_LAYER){
            pl_reset_topology_one_layer();
        }
        else{
            assert(false);
        }
    }
}