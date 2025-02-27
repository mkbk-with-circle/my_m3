#![feature(path_file_prefix)]

pub mod experiment;
pub mod mlsys;
pub mod mix;
pub mod ns3;
pub mod ns3link;
pub use experiment::Experiment;

/*
这些声明表示项目定义了多个模块，这些模块的实现分别位于
 experiment.rs、mlsys.rs、mix.rs、ns3.rs 和 ns3link.rs 文件中。pub mod 语句使这些模块公开，使得它们可以在项目的其他部分或外部被访问。

pub use experiment::Experiment;：
这行代码将 Experiment 结构体或模块从 experiment 模块中公开，
使得它可以直接通过 sensitivity_analysis::Experiment（假设你的项目包名为 sensitivity_analysis）
*/