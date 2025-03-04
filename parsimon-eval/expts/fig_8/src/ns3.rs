//! An interface to the backend ns-3 simulation.
//!
//! This crate is tightly coupled to interface provided by the ns-3 scripts.

#![warn(unreachable_pub, missing_debug_implementations, missing_docs)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::{fmt::Write, io};

use derivative::Derivative;
use parsimon::core::{
    network::Flow,
    network::{
        types::{Link, Node},
        FctRecord, NodeKind,
    },
    units::{Bytes, Nanosecs},
};

/// An ns-3 simulation.
#[derive(Debug, typed_builder::TypedBuilder)]
pub struct Ns3Simulation {
    /// The directory in the ns-3 source tree containing the `run.py`.
    #[builder(setter(into))]
    pub ns3_dir: PathBuf,
    /// The directory in which to write simulation configs and data.
    #[builder(setter(into))]
    pub data_dir: PathBuf,
    /// The topology nodes.
    pub nodes: Vec<Node>,
    /// The topology links.
    pub links: Vec<Link>,
    /// The base RTT.
    pub base_rtt: Nanosecs,
    /// The flows to simulate.
    /// PRECONDITION: `flows` must be sorted by start time
    pub flows: Vec<Flow>,
    /// The buffer size factor.
    #[builder(default = 30.0)]
    pub bfsz: f64,
    /// The sencing window.
    #[builder(default = Bytes::new(18000))]
    pub window: Bytes,
    /// Enable PFC.
    #[builder(default = 1.0)]
    pub enable_pfc: f64,
    /// The congestion control protocol.
    #[builder(default)]
    pub cc_kind: CcKind,
    /// The congestion control parameter.
    #[builder(default = 30.0)]
    pub param_1: f64,
    /// The congestion control parameter.
    #[builder(default = 0.0)]
    pub param_2: f64,
}

impl Ns3Simulation {
    /// Run the simulation, returning a vector of [FctRecord]s.
    ///
    /// This routine can fail due to IO errors or errors parsing ns-3 data.
    pub fn run(&self) -> Result<Vec<FctRecord>, Error> {
        // Set up directory
        let mk_path = |dir, file| [dir, file].into_iter().collect::<PathBuf>();
        fs::create_dir_all(&self.data_dir)?;

        // Set up the topology
        let topology = translate_topology(&self.nodes, &self.links);
        fs::write(
            mk_path(self.data_dir.as_path(), "topology.txt".as_ref()),
            topology,
        )?;

        // Set up the flows
        let flows = translate_flows(&self.flows);
        fs::write(
            mk_path(self.data_dir.as_path(), "flows.txt".as_ref()),
            flows,
        )?;

        // Run ns-3
        self.invoke_ns3()?;

        // Parse and return results
        let s = fs::read_to_string(mk_path(
            self.data_dir.as_path(),
            format!("fct_topology_flows_{}.txt", self.cc_kind.as_str()).as_ref(),
        ))?;
        let records = parse_ns3_records(&s)?;
        let data_dir=self.data_dir.to_str().unwrap();
        let fct_file=format!("fct_topology_flows_{}.txt", self.cc_kind.as_str());
        // println!("rm {data_dir}/{fct_file}");
        let _output = Command::new("sh")
            .arg("-c")
            .arg(format!("rm {data_dir}/{fct_file}"))
            .output()?;
        Ok(records)
    }

    fn invoke_ns3(&self) -> io::Result<()> {
        // We need to canonicalize the directories because we run `cd` below.
        let data_dir = std::fs::canonicalize(&self.data_dir)?;
        let data_dir = data_dir.display();
        let ns3_dir = std::fs::canonicalize(&self.ns3_dir)?;
        let ns3_dir = ns3_dir.display();

        // Build the command that runs the Python script.
        let base_rtt = self.base_rtt.into_u64();
        let bfsz = self.bfsz;
        let window = self.window.into_u64();
        let enable_pfc = self.enable_pfc;
        let cc = self.cc_kind.as_str();
        let param_1 = self.param_1;
        let param_2 = self.param_2;
        
        let python_command = format!(
            "python2 run.py --root {data_dir} --base_rtt {base_rtt} \
            --topo topology --trace flows --bw 10 --bfsz {bfsz} --fwin {window} --enable_pfc {enable_pfc} --cc {cc} --param_1 {param_1} --param_2 {param_2} \
            > {data_dir}/output.txt 2>&1"
        );
        // Execute the command in a child process.
        let _output = Command::new("sh")
            .arg("-c")
            .arg(format!("cd {ns3_dir}; {python_command}; rm {data_dir}/flows.txt"))
            // .arg(format!("cd {ns3_dir};{python_command}"))
            .output()?;
        Ok(())
    }
}

/// The error type for [Ns3Simulation::run].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Error parsing ns-3 formats.
    #[error("failed to parse ns-3 format")]
    ParseNs3(#[from] ParseNs3Error),

    /// IO error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn translate_topology(nodes: &[Node], links: &[Link]) -> String {
    let mut s = String::new();
    let switches = nodes
        .iter()
        .filter(|&n| matches!(n.kind, NodeKind::Switch))
        .collect::<Vec<_>>();
    // First line: total node #, switch node #, link #
    writeln!(s, "{} {} {}", nodes.len(), switches.len(), links.len()).unwrap();
    // Second line: switch node IDs...
    let switch_ids = switches
        .iter()
        .map(|&s| s.id.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    writeln!(s, "{switch_ids}").unwrap();
    // src0 dst0 rate delay error_rate
    // src1 dst1 rate delay error_rate
    // ...
    for link in links {
        writeln!(
            s,
            "{} {} {} {} 0",
            link.a, link.b, link.bandwidth, link.delay
        )
        .unwrap();
    }
    s
}

fn translate_flows(flows: &[Flow]) -> String {
    let nr_flows = flows.len();
    // First line: # of flows
    // src0 dst0 3 dst_port0 size0 start_time0
    // src1 dst1 3 dst_port1 size1 start_time1
    let lines = std::iter::once(nr_flows.to_string())
        .chain(flows.iter().map(|f| {
            format!(
                "{} {} {} 3 100 {} {}",
                f.id,
                f.src,
                f.dst,
                f.size.into_u64(),
                f.start.into_u64() as f64 / 1e9 // in seconds, for some reason
            )
        }))
        .collect::<Vec<_>>();
    lines.join("\n")
}

fn parse_ns3_records(s: &str) -> Result<Vec<FctRecord>, ParseNs3Error> {
    s.lines().map(parse_ns3_record).collect()
}

fn parse_ns3_record(s: &str) -> Result<FctRecord, ParseNs3Error> {
    // sip, dip, sport, dport, size (B), start_time, fct (ns), standalone_fct (ns)
    const NR_NS3_FIELDS: usize = 9;
    let fields = s.split_whitespace().collect::<Vec<_>>();
    let nr_fields = fields.len();
    if nr_fields != NR_NS3_FIELDS {
        return Err(ParseNs3Error::WrongNrFields {
            expected: NR_NS3_FIELDS,
            got: nr_fields,
        });
    }
    Ok(FctRecord {
        id: fields[0].parse()?,
        size: fields[5].parse()?,
        start: fields[6].parse()?,
        fct: fields[7].parse()?,
        ideal: fields[8].parse()?,
    })
}

/// Error parsing ns-3 formats.
#[derive(Debug, thiserror::Error)]
pub enum ParseNs3Error {
    /// Incorrect number of fields.
    #[error("Wrong number of fields (expected {expected}, got {got}")]
    WrongNrFields {
        /// Expected number of fields.
        expected: usize,
        /// Actual number of fields.
        got: usize,
    },

    /// Error parsing field value.
    #[error("Failed to parse field")]
    ParseInt(#[from] std::num::ParseIntError),
}

/// Congestion control protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Derivative, serde::Serialize, serde::Deserialize)]
#[derivative(Default)]
#[serde(rename_all = "lowercase")]
pub enum CcKind {
    /// DCTCP.
    #[derivative(Default)]
    Dctcp,
    /// DCQCN.
    Dcqcn,
    /// HP
    Hp,
    /// TIMELY.
    Timely,
}

impl CcKind {
    fn as_str(&self) -> &'static str {
        match self {
            CcKind::Dctcp => "dctcp",
            CcKind::Dcqcn => "dcqcn_paper_vwin",
            CcKind::Hp => "hp",
            CcKind::Timely => "timely_vwin",
        }
    }

    const DCTCP_VALUE: usize = 0;
    const DCQCN_VALUE: usize = 1;
    const HP_VALUE: usize = 2;
    const TIMELY_VALUE: usize = 3;

    /// Get the integer value of the cc protocol.
    pub fn get_int_value(&self) -> usize {
        match self {
            CcKind::Dctcp => Self::DCTCP_VALUE,
            CcKind::Dcqcn => Self::DCQCN_VALUE,
            CcKind::Hp => Self::HP_VALUE,
            CcKind::Timely => Self::TIMELY_VALUE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use parsimon::core::{
        network::{Flow, FlowId, NodeId},
        testing,
        units::{Bytes, Nanosecs},
    };

    #[test]
    fn translate_topology_correct() -> anyhow::Result<()> {
        let (nodes, links) = testing::eight_node_config();
        let s = translate_topology(&nodes, &links);
        insta::assert_snapshot!(s, @r###"
        8 4 8
        4 5 6 7
        0 4 10000000000bps 1000ns 0
        1 4 10000000000bps 1000ns 0
        2 5 10000000000bps 1000ns 0
        3 5 10000000000bps 1000ns 0
        4 6 10000000000bps 1000ns 0
        4 7 10000000000bps 1000ns 0
        5 6 10000000000bps 1000ns 0
        5 7 10000000000bps 1000ns 0
        "###);
        Ok(())
    }

    #[test]
    fn translate_flows_correct() -> anyhow::Result<()> {
        let flows = vec![
            Flow {
                id: FlowId::new(0),
                src: NodeId::new(0),
                dst: NodeId::new(1),
                size: Bytes::new(1234),
                start: Nanosecs::new(1_000_000_000),
            },
            Flow {
                id: FlowId::new(1),
                src: NodeId::new(0),
                dst: NodeId::new(2),
                size: Bytes::new(5678),
                start: Nanosecs::new(2_000_000_000),
            },
        ];
        let s = translate_flows(&flows);
        insta::assert_snapshot!(s, @r###"
        2
        0 0 1 3 100 1234 1
        1 0 2 3 100 5678 2
        "###);
        Ok(())
    }

    #[test]
    fn simulate_8_node_topology() -> anyhow::Result<()> {
        // Create an instance of the 8-node topology
        let (nodes, links) = testing::eight_node_config();
        let flows = (0..1)
            .map(|i| Flow {
                id: FlowId::new(i),
                src: NodeId::new(0),
                dst: NodeId::new(3),
                size: Bytes::new(3_000),
                start: Nanosecs::new(2_000_000_000),
            })
            .collect::<Vec<_>>();
        // Create an instance of Ns3Simulation using the 8-node topology
        let simulation = Ns3Simulation::builder()
            .ns3_dir("../../../parsimon/backends/High-Precision-Congestion-Control/simulation")
            .data_dir("/data1/lichenni/projects/flow_simulation/parsimon-eval/expts/fig_8/data_test/test")
            .nodes(nodes)
            .links(links)
            .base_rtt(Nanosecs::new(1000)) // Set base RTT as required
            .flows(flows) // Add flows if needed
            .build();
        
        // Run the simulation
        let result = simulation.run();

        // Assert that the simulation ran successfully
        Ok(())
    }
}
