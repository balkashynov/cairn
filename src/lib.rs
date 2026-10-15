//! Static vector search: quantized scans, conservative bounds, exact verification.
pub mod kernel;
pub mod quantize;
pub mod search;
pub mod storage;
pub use search::{Mode, SearchOptions, SearchResult};
pub use storage::{Index, build};
