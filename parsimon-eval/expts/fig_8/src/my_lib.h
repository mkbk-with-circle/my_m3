#ifndef MY_LIB
#define MY_LIB


#include <vector>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <fcntl.h>
#include "topo.h"
#include <unistd.h>

#include <sys/mman.h> 


using namespace std;

// define
#define NR_FLOWS 10000000
struct Flow;
const string MLSYS_PATH = "../../../clibs";
const string DATA_PATH = " ";
using PathSegment = pair<int, int>; // 起点和终点
using Path = vector<int>;          // 完整路径
using PathSegmentMap = unordered_map<int, vector<PathSegment>>; // FlowID -> 路径段列表
using FlowSet = unordered_set<int>; // FlowID 集合
using FlowMap = unordered_map<int, Flow>; // 流 ID 到流的映射
const double MTU=1000.0;
const double BDP=10.0*MTU;
const double DELAY_PROP_BASE=1000.0;
const double BYTE_TO_BIT=8.0;
const double HEADER_SIZE=48.0;
extern vector<vector<pair<int,int>>> all_path_list;  // 从源到目的的路径（节点ID的序列）



// Flow
struct Flow {
    uint32_t flowID;                 // 流ID
    uint32_t src;             // 源节点
    uint32_t dst;        // 目标节点
    double size;             // 流的字节数
    int path_id;
    double fat;
    uint32_t idx;
    float bw_bottleneck;
    double remaining_size;  // 流的剩余大小，表示当前还未传输完成的数据量
    double fct_est;         // 流的估计完成时间 (Estimated Flow Completion Time)
    float sldn;
    // 将 Flow 转换为字符串的函数
    std::string to_string() const {
        std::ostringstream oss;
        oss << flowID << " " 
            << src << " " 
            << dst << " " 
            << size << " " 
            << fat;
        return oss.str();
    }
};

typedef struct
{
    float *w1; // (hidden_dim, dim)
    float *w2; // (hidden_dim, hidden_dim)
    float *w3; // (dim, hidden_dim)
    float *const_opt; // const placeholder for optimization
} ModelWeightsMLP;


struct VectorPairHash {
    size_t operator()(const std::vector<std::pair<int, int>>& vec) const {
        size_t seed = vec.size();
        for (const auto& pair : vec) {
            seed ^= std::hash<int>()(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>()(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// 构造 PairHash
struct PairHash {
    template <typename T1, typename T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
    }
};


// 定义类型别名
typedef std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash> ChannelToFlowIDMap;





//Bytes
class Bytes {
public:
    // Constructor
    explicit Bytes(uint64_t value = 0) : value_(value) {}
    // Factory methods for conversion from Gigabytes and Kilobytes
    static Bytes fromGigabytes(uint64_t gb) {
        return Bytes(gb * 1'000'000'000);
    }
    static Bytes fromKilobytes(uint64_t kb) {
        return Bytes(kb * 1'000);
    }
    static Bytes from(uint64_t size) {
        return Bytes(size);
    }
    // Get the raw byte value
    uint64_t to_uint64() const {
        return value_;
    }
    // Overload the stream insertion operator for display
    friend std::ostream& operator<<(std::ostream& os, const Bytes& bytes) {
        os << bytes.value_ << "B";
        return os;
    }
private:
    uint64_t value_; // Internal storage for bytes
};
//CcKinds
class CcKind {
public:
    // 枚举值
    enum Value {
        DCTCP,
        DCQCN,
        HP,
        TIMELY
    };
private:
    Value kind; // 存储当前的协议类型
public:
    // 构造函数
    explicit CcKind(Value kind) : kind(kind) {}
    // 默认构造函数 (默认值为 DCTCP)
    CcKind() : kind(DCTCP) {}
    // 获取字符串表示
    string as_str() const {
        switch (kind) {
            case DCTCP:
                return "dctcp";
            case DCQCN:
                return "dcqcn_paper_vwin";
            case HP:
                return "hp";
            case TIMELY:
                return "timely_vwin";
            default:
                return "unknown";
        }
    }
    // 获取整数值
    int get_int_value() const {
        switch (kind) {
            case DCTCP:
                return 0;
            case DCQCN:
                return 1;
            case HP:
                return 2;
            case TIMELY:
                return 3;
            default:
                throw std::runtime_error("Invalid CcKind");
        }
    }
    // 获取当前类型
    Value get_kind() const {
        return kind;
    }
    // 设置当前类型
    void set_kind(Value new_kind) {
        kind = new_kind;
    }
};
typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim) -> embedding to linear layer
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} ModelWeightsTransformer;

typedef struct { 
    Config config; // the hyperparameters of the architecture (the blueprint)
    ModelWeightsTransformer weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;



typedef struct
{
    int input_dim;        // transformer dimension
    int hidden_dim_1; // for ffn layers
    int hidden_dim_2;   // number of layers
    int output_dim;    // max sequence length
    int y_len;    // max sequence length
} ConfigMLP;
typedef struct
{
    float *x;      // activation at current time stamp (dim,)
    float *h1;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *h2;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *logits; // output logits
} RunStateMLP;








//func
std::unordered_map<int, Flow> readFlowsAsMap(const std::string &filename);
std::vector<std::vector<std::pair<int, int>>> convertToPathList(const std::unordered_map<int, Flow>& vec_flows) ;
ChannelToFlowIDMap constructChannelToFlowIDMap(const std::vector<std::vector<std::pair<int, int>>>& path_list,const std::unordered_map<int, Flow>& vec_flows) ;
long time_in_ms() ;
void get_fct_mmf(unsigned int n_flows, Flow* flows, int h, int *topo, int method_mmf, int method_routing, int type_topo, int base_lr) ;
void config_link_to_delay(int n_hosts,double *link_to_delay,float *param_list);
int compare_flow_sldn(const void *a, const void *b);
int compare_flow_size(const void *a, const void *b);
void calculate_and_save_percentiles(Flow* flows, int n_flows, double buckets[], int n_buckets,double percentiles[], int n_percentiles, float *feat_map,int pos_start,int bucket_thold, float *const_opt);
void error_usage();
void build_transformer(Transformer *t, char* ckpt_path_llama);
void read_checkpoint(char* checkpoint, Config* config, ModelWeightsTransformer* weights,
                     int* fd, float** data, ssize_t* file_size);
void malloc_run_state(RunState* s, Config* p);
void build_mlp(char* ckpt_path_mlp, ConfigMLP *config_mlp, ModelWeightsMLP *weights_mlp, RunStateMLP *state_mlp);
void memory_map_weights_mlp(ModelWeightsMLP *w, ConfigMLP *p, float *f);
void malloc_run_state_mlp(RunStateMLP *s, ConfigMLP *p);
void memory_map_weights(ModelWeightsTransformer *w, Config* p, float* ptr, int shared_weights);
void generate(Transformer *transformer, float* feat_map, int n_hosts, int n_feat_input,int n_feat_context, float* feat_concat);
void forward_MLP(float *x, ConfigMLP *p, ModelWeightsMLP *w, RunStateMLP *s);
void matmul_with_bias(float* xout, float* x, float* w, int n, int d);
void write_vecs_to_file(const char *filename, float *vec_1, int vec_size_1, float *vec_2, int vec_size_2) ;
void free_transformer(Transformer* t) ;
void free_run_state(RunState* s);
void free_run_state_mlp(RunStateMLP *s);
float* forward(Transformer* transformer, float* feat,int pos) ;
void rmsnorm(float* o, float* x, float* weight, int size) ;
void softmax(float* x, int size);
void matmul(float* xout, float* x, float* w, int n, int d);
void writeSampledPathsToFile(const std::vector<std::vector<std::pair<int, int>>>& sampled_paths, const std::string& filename);
void writeFlowsToFile(const Flow* flows, int n_flows, const string& filename);






#endif