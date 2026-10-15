use anyhow::{Context, Result, bail, ensure};
use cairn::{Index, Mode, SearchOptions, storage};
use serde::Deserialize;
use std::{
    io::{self, BufRead, Read, Write},
    path::Path,
};

const HELP: &str = "Cairn: quantized exact vector search\n\n\
  cairn build VECTORS.fbin INDEX.cairn [--tags TAGS.u64]\n\
  cairn inspect INDEX.cairn\n\
  cairn verify INDEX.cairn [--deep]\n\
  cairn search INDEX.cairn QUERIES.fbin [--k 10] [--mode certified|exhaustive|binary|int8] [--oversample 4] [--tag N]\n\
  cairn serve INDEX.cairn\n\n\
Search writes one JSON object per query. Serve accepts JSON lines on stdin:\n\
  {\"query\":[0.1,0.2],\"k\":10,\"mode\":\"certified\",\"tag\":7}\n\
fbin is little-endian u32 rows/dimensions followed by row-major f32.\n\
Indexes are immutable: never write/truncate a mapped file. Build refuses overwrite.\n\
verify --deep validates hashes and mathematical invariants of externally supplied indexes.\n\
Certified results refer to stored normalized f32 vectors, not semantic relevance.\n";

fn main() {
    if let Err(e) = run() {
        eprintln!("cairn: {e:#}");
        std::process::exit(1);
    }
}
fn value(args: &[String], flag: &str) -> Result<Option<String>> {
    match args.iter().position(|a| a == flag) {
        Some(i) => Ok(Some(
            args.get(i + 1)
                .filter(|s| !s.starts_with("--"))
                .context(format!("missing value for {flag}"))?
                .clone(),
        )),
        None => Ok(None),
    }
}
fn validate_flags(args: &[String], names: &[&str]) -> Result<()> {
    ensure!(args.len().is_multiple_of(2), "flags require values");
    let mut seen = std::collections::HashSet::new();
    for pair in args.chunks(2) {
        ensure!(
            names.contains(&pair[0].as_str()),
            "unknown option: {}",
            pair[0]
        );
        ensure!(seen.insert(&pair[0]), "duplicate option: {}", pair[0]);
    }
    Ok(())
}
fn options(args: &[String]) -> Result<SearchOptions> {
    validate_flags(args, &["--k", "--mode", "--oversample", "--tag"])?;
    Ok(SearchOptions {
        k: value(args, "--k")?
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(10),
        mode: value(args, "--mode")?
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(Mode::Certified),
        oversample: value(args, "--oversample")?
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(4),
        tag: value(args, "--tag")?.map(|s| s.parse()).transpose()?,
    })
}
fn open(path: &str) -> Result<Index> {
    // SAFETY: CLI contract requires immutable snapshots. Cairn itself never writes
    // an existing index. External writers must use a new filename as documented.
    unsafe { Index::open(path) }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || ["--help", "-h", "help"].contains(&args[0].as_str()) {
        print!("{HELP}");
        return Ok(());
    }
    if args[0] == "--version" {
        println!(
            "cairn {} ({})",
            env!("CARGO_PKG_VERSION"),
            cairn::kernel::name()
        );
        return Ok(());
    }
    let a = |i: usize| {
        args.get(i)
            .map(|s| s.as_str())
            .context("missing argument; see --help")
    };
    match args[0].as_str() {
        "build" => {
            let (input, output) = (a(1)?, a(2)?);
            validate_flags(&args[3..], &["--tags"])?;
            let tags = value(&args[3..], "--tags")?;
            let result = storage::build(
                Path::new(input),
                Path::new(output),
                tags.as_deref().map(Path::new),
            )?;
            println!("{}", serde_json::to_string(&result)?);
        }
        "inspect" => {
            ensure!(args.len() == 2, "inspect expects one index");
            println!("{}", serde_json::to_string_pretty(&open(a(1)?)?.info())?);
        }
        "verify" => {
            ensure!(
                args.len() == 2 || (args.len() == 3 && args[2] == "--deep"),
                "verify expects INDEX [--deep]"
            );
            let index = open(a(1)?)?;
            index.verify()?;
            if args.len() == 3 {
                index.validate_bounds()?;
            }
            println!(
                "{}",
                serde_json::json!({"ok":true,"deep":args.len()==3,"rows":index.layout().rows})
            );
        }
        "search" => {
            let index = open(a(1)?)?;
            let queries = storage::FbinReader::open(Path::new(a(2)?))?;
            let opts = options(&args[3..])?;
            let stdout = io::stdout();
            let mut out = io::BufWriter::new(stdout.lock());
            for (qid, q) in queries.enumerate() {
                let result = index.search(&q?, &opts)?;
                serde_json::to_writer(
                    &mut out,
                    &serde_json::json!({"query_id":qid,"result":result}),
                )?;
                writeln!(out)?;
            }
            out.flush()?;
        }
        "serve" => {
            ensure!(args.len() == 2, "serve expects one index");
            let index = open(a(1)?)?;
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Request {
                query: Vec<f32>,
                k: Option<usize>,
                mode: Option<Mode>,
                oversample: Option<usize>,
                tag: Option<u64>,
            }
            let stdin = io::stdin();
            let mut reader = stdin.lock();
            let stdout = io::stdout();
            let mut out = stdout.lock();
            // Bound a JSON frame before allocating an unbounded line.
            loop {
                let mut line = Vec::new();
                let n = Read::by_ref(&mut reader)
                    .take(4 * 1024 * 1024 + 1)
                    .read_until(b'\n', &mut line)?;
                if n == 0 {
                    break;
                }
                ensure!(n <= 4 * 1024 * 1024, "request exceeds 4 MiB");
                let response = (|| -> Result<_> {
                    let r: Request = serde_json::from_slice(&line)?;
                    index.search(
                        &r.query,
                        &SearchOptions {
                            k: r.k.unwrap_or(10),
                            mode: r.mode.unwrap_or(Mode::Certified),
                            oversample: r.oversample.unwrap_or(4),
                            tag: r.tag,
                        },
                    )
                })();
                match response {
                    Ok(r) => serde_json::to_writer(&mut out, &r)?,
                    Err(e) => serde_json::to_writer(
                        &mut out,
                        &serde_json::json!({"error":e.to_string()}),
                    )?,
                };
                writeln!(out)?;
                out.flush()?;
            }
        }
        _ => bail!("unknown command {}; see --help", args[0]),
    }
    Ok(())
}
