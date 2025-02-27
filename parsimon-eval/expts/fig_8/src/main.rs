use clap::Parser;
use sensitivity_analysis::Experiment;

fn main() -> anyhow::Result<()> {
    let expt = Experiment::parse();//解析命令行
    expt.run()?;
    Ok(())
}
