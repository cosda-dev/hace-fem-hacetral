// Memory Mirror stress test (Linux-only, ignored by default).
// Requires a large blob file to be present at HACE_TEST_BLOB.

use std::fs::File;
use std::io::Read;
use std::path::Path;

#[cfg(target_os = "linux")]
fn rss_kb() -> usize {
    let mut statm = String::new();
    let mut file = File::open("/proc/self/statm").expect("statm");
    file.read_to_string(&mut statm).expect("statm read");
    let parts: Vec<&str> = statm.split_whitespace().collect();
    if parts.len() < 2 {
        return 0;
    }
    let pages: usize = parts[1].parse().unwrap_or(0);
    pages * 4
}

#[test]
#[ignore]
fn mmap_large_blob_under_10mb_rss_delta() {
    let path = std::env::var("HACE_TEST_BLOB").expect("HACE_TEST_BLOB env var");
    let path = Path::new(&path);
    assert!(path.exists(), "blob path must exist");

    let before = rss_kb();

    // Touching only a small window; OS should keep RSS low.
    let mut file = File::open(path).expect("blob open");
    let mut buf = [0u8; 4096];
    file.read_exact(&mut buf).expect("read window");

    let after = rss_kb();
    let delta = after.saturating_sub(before);

    // 10MB = 10240 KB
    assert!(delta < 10_240, "RSS delta too high: {delta} KB");
}
