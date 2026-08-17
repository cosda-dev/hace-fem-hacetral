
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub temperature: f32,
}

impl Sampler {
    pub const fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Greedy sampler (baseline). Ignores temperature for now.
    pub fn sample(&self, logits: &[f32]) -> Option<usize> {
        if logits.is_empty() {
            return None;
        }

        let mut max = logits[0];
        let mut idx = 0usize;

        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > max {
                max = v;
                idx = i;
            }
        }

        Some(idx)
    }
}
