use rayon::prelude::*;
use std::process::Command;
use clap::Parser;
use std::path::PathBuf;
use std::fs;
struct Parameters {
    shard: Vec<u32>,
    n_flows: Vec<u32>,
    n_hosts: Vec<u32>,
    shard_cc: Vec<u32>
}

#[derive(Debug, Parser)]
pub struct Main {
    #[clap(long, default_value = "/data1/lichenni/software/anaconda3/envs/py39/bin/python")]
    python_path: PathBuf,
    #[clap(long, default_value = "/data2/lichenni/test")]
    output_dir: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Main::parse();
    let python_path = args.python_path.display().to_string();
    let output_dir = args.output_dir.display().to_string();

    println!("python_path: {:?}, output_dir: {:?}", python_path,output_dir);

    let base_rtt = 14400;
    let enable_tr = 0;
    let enable_debug = 0;

    // setup the configurations
    let params = Parameters {
        shard: (0..7).collect(),
        n_flows: vec![20000],
        n_hosts: vec![3, 5, 7],
        shard_cc: (0..3).collect(),
    };

    // config for demo purpose
    // let params = Parameters {
    //     shard: (0..100).collect(),
    //     n_flows: vec![100],
    //     n_hosts: vec![3],
    //     shard_cc: (0..1).collect(),
    // };

    // no need to change
    let root_path = format!("..");
    let log_dir = "./logs";
    if let Err(err) = fs::create_dir_all(log_dir) {
        eprintln!("Error creating directory '{}': {}", log_dir, err);
    } else {
        println!("Directory '{}' created successfully.", log_dir);
    }

    let file_traffic = format!("{}/traffic_gen/traffic_gen_synthetic.py", root_path);//用于生成仿真流量
    let file_sim = format!("{}/simulation/run_m3.py", root_path);
    let file_ns3 = format!("{}/analysis/fct_to_file.py", root_path);
    let file_reference = format!("{}/analysis/main_flowsim_mmf.py", root_path);
    let type_topo = "topo-pl";


    // 生成流量
    // println!("{:?}", Parameters::field_names());
    itertools::iproduct!(&params.shard, &params.n_flows, &params.n_hosts)
        .par_bridge()
        .for_each(|combination| {
            let shard = combination.0;
            let n_flows = combination.1;
            let n_hosts = combination.2;

            // println!("{:?}", combination);
            let scenario_dir = format!(
                "shard{}_nflows{}_nhosts{}_lr10Gbps",
                shard, n_flows, n_hosts,
            );
            let output_path = format!("{}/{}", output_dir, scenario_dir);

            // gen traffic
            let command_args = format!(
                "--shard {} -f {} -n {} -b 10G -o {} --switchtohost 4",
                shard, n_flows, n_hosts, output_path,
            );
            let log_path = format!("{}/nhosts{}_traffic.log", log_dir, n_hosts);
            let py_command = format!("{} {} {}", python_path, file_traffic, command_args);
            let cmd = format!(
                "echo {} >> {}; {} >> {}; echo \"\">>{}",
                py_command, log_path, py_command, log_path, log_path
            );
            // println!("{}", cmd);
            let mut child = Command::new("sh").arg("-c").arg(cmd).spawn().unwrap();
            let mut _result = child.wait().unwrap();
        });

    // 运行网络仿真，运行m3
    // println!("{:?}", Parameters::field_names());
    itertools::iproduct!(
        &params.shard,
        &params.n_flows,
        &params.n_hosts,
        &params.shard_cc
    )
    .par_bridge()
    .for_each(|combination| {
        let shard = combination.0;
        let n_flows = combination.1;
        let n_hosts = combination.2;
        let shard_cc = combination.3;
        let shard_total = shard * params.shard_cc.len() as u32 + shard_cc;

        println!("{:?}", combination);
        let scenario_dir = format!(
            "shard{}_nflows{}_nhosts{}_lr10Gbps",
            shard, n_flows, n_hosts,
        );

        // ns3 sim
        let mut command_args = format!(
            "--trace flows --bw 10 --base_rtt {} \
            --topo {}-{}  --root {}/{} --shard_cc {} --shard_total {} --enable_tr {} --enable_debug {}",base_rtt, type_topo, n_hosts, output_dir, scenario_dir, shard_cc, shard_total, enable_tr, enable_debug,
        );
        let mut log_path = format!("{}/nhosts{}_sim.log", log_dir, n_hosts,);
        let mut py_command = format!("{} {} {}", python_path, file_sim, command_args,);
        let mut cmd = format!(
            "echo {} >> {}; {} >> {}/{}/pdrop_{}-{}_s{}.txt",
            py_command, log_path, py_command, output_dir, scenario_dir,type_topo, n_hosts, shard_cc
        );
        // let mut cmd = format!(
        //     "echo {} >> {}; {} >> {}; echo \"\">>{}",
        //     py_command, log_path, py_command, log_path, log_path
        // );
        // println!("{}", cmd);
        let mut child = Command::new("sh").arg("-c").arg(cmd).spawn().unwrap();
        let mut _result = child.wait().unwrap();

        // parse ground-truth
        command_args = format!(
            "--shard {} -b 10 -p {}-{} --output_dir {} --scenario_dir {} --shard_cc {} --enable_debug {}",
            shard, type_topo, n_hosts, output_dir, scenario_dir, shard_cc,enable_debug
        );
        log_path = format!("{}/nhosts{}_ns3.log", log_dir, n_hosts,);
        py_command = format!("{} {} {}", python_path, file_ns3, command_args,);
        cmd = format!(
            "echo {} >> {}; {} >> {}; echo \"\">>{}",
            py_command, log_path, py_command, log_path, log_path
        );
        // println!("{}", cmd);
        child = Command::new("sh").arg("-c").arg(cmd).spawn().unwrap();
        _result = child.wait().unwrap();
    });


    // println!("{:?}", Parameters::field_names());
    itertools::iproduct!(&params.shard, &params.n_flows, &params.n_hosts)
        .par_bridge()
        .for_each(|combination| {
            let shard = combination.0;
            let n_flows = combination.1;
            let n_hosts = combination.2;

            // println!("{:?}", combination);
            let scenario_dir = format!(
                "shard{}_nflows{}_nhosts{}_lr10Gbps",
                shard, n_flows, n_hosts,
            );

            // run reference sys (e.g., max-min fair sharing)
            let command_args = format!(
                "--shard {} -b 10 -p {}-{} --output_dir {} --scenario_dir {} --nhost {}",
                shard, type_topo, n_hosts, output_dir, scenario_dir, n_hosts,
            );
            let log_path = format!("{}/nhosts{}_reference.log", log_dir, n_hosts,);
            let py_command = format!("{} {} {}", python_path, file_reference, command_args,);
            let cmd = format!(
                "echo {} >> {}; {} >> {}; echo \"\">>{}",
                py_command, log_path, py_command, log_path, log_path
            );
            // println!("{}", cmd);
            let mut child = Command::new("sh").arg("-c").arg(cmd).spawn().unwrap();
            let mut _result = child.wait().unwrap();
        });

    Ok(())
}




/*
流量生成：生成流量的命令由Python脚本完成，并将输出写入日志。
NS3仿真：调用仿真脚本run_m3.py并捕获日志输出。该脚本使用生成的流量配置和主机配置来执行仿真。
解析Ground-Truth：通过fct_to_file.py解析NS3的仿真结果。
参考系统运行：运行参考系统的仿真，如最大-最小公平分配算法的实现，并将其结果记录到日志文件中。
*/


/*
第一阶段：生成网络流量
生成每种参数组合对应的网络流量：

输入：
shard: 分片编号。
n_flows: 流量数量。
n_hosts: 主机数量。
执行：
调用 traffic_gen_synthetic.py，传入参数 shard、n_flows、n_hosts。
将生成的流量数据保存到对应的输出目录中。
日志： 将生成流量的命令和结果写入 ./logs/nhosts<N>_traffic.log。


第二阶段：运行网络仿真
根据网络流量配置运行仿真任务：

输入：
增加了 shard_cc 参数，用于标识分片下的子任务编号。
执行：
调用 run_m3.py 脚本，传入参数 shard_cc、n_hosts 等。
将仿真结果保存到输出目录中。
日志： 将仿真的命令和结果写入 ./logs/nhosts<N>_sim.lo



第三阶段：分析仿真结果
对仿真结果进行解析，提取关键性能指标（如流量完成时间 FCT）：

输入：
传入之前仿真任务的输出目录作为输入。
执行：
调用 fct_to_file.py 脚本，解析仿真结果。
将解析后的数据存储到对应的文件中。
日志： 将解析任务的命令和结果写入 ./logs/nhosts<N>_ns3.log



第四阶段：运行参考仿真系统
运行一个参考仿真系统（如 Max-Min Fairness），用于与主仿真任务结果进行对比：

执行：
调用 main_flowsim_mmf.py，传入相应的参数。
保存参考仿真的输出结果到对应目录。
日志： 将参考仿真的命令和结果写入 ./logs/nhosts<N>_reference.log
*/

