
use core::marker::PhantomData;

/// Maximum supported rank (compile-time cap to avoid alloc).
pub const MAX_DIMS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    InvalidShape,
    OutOfBounds,
    Overflow,
}

/// Core tensor view (zero-copy, borrowed only).
pub struct TensorView<'a, T> {
    data: &'a [T],
    shape: [usize; MAX_DIMS],
    strides: [usize; MAX_DIMS],
    rank: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> Copy for TensorView<'a, T> {}
impl<'a, T> Clone for TensorView<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Mutable tensor view (zero-copy, borrowed only).
pub struct TensorViewMut<'a, T> {
    data: &'a mut [T],
    shape: [usize; MAX_DIMS],
    strides: [usize; MAX_DIMS],
    rank: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> TensorView<'a, T> {
    pub fn new(data: &'a [T], shape: &[usize], strides: &[usize]) -> Result<Self, TensorError> {
        let rank = shape.len();
        if rank == 0 || rank > MAX_DIMS || strides.len() != rank {
            return Err(TensorError::InvalidShape);
        }

        let mut s = [0usize; MAX_DIMS];
        let mut st = [0usize; MAX_DIMS];
        for i in 0..rank {
            if shape[i] == 0 {
                return Err(TensorError::InvalidShape);
            }
            s[i] = shape[i];
            st[i] = strides[i];
        }

        let required = required_len(rank, &s, &st)?;
        if required > data.len() {
            return Err(TensorError::OutOfBounds);
        }

        Ok(Self {
            data,
            shape: s,
            strides: st,
            rank,
            _marker: PhantomData,
        })
    }

    pub fn from_contiguous(data: &'a [T], shape: &[usize]) -> Result<Self, TensorError> {
        let rank = shape.len();
        if rank == 0 || rank > MAX_DIMS {
            return Err(TensorError::InvalidShape);
        }

        let mut strides = [0usize; MAX_DIMS];
        let mut stride = 1usize;
        for i in (0..rank).rev() {
            strides[i] = stride;
            stride = stride.checked_mul(shape[i]).ok_or(TensorError::Overflow)?;
        }

        Self::new(data, shape, &strides[..rank])
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.rank]
    }

    pub fn numel(&self) -> usize {
        let mut n = 1usize;
        for i in 0..self.rank {
            n = n.saturating_mul(self.shape[i]);
        }
        n
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        for i in (0..self.rank).rev() {
            if self.strides[i] != expected {
                return false;
            }
            match expected.checked_mul(self.shape[i]) {
                Some(next) => expected = next,
                None => return false,
            }
        }
        true
    }

    #[inline(always)]
    fn compute_offset(&self, idx: &[usize]) -> Result<usize, TensorError> {
        if idx.len() != self.rank {
            return Err(TensorError::OutOfBounds);
        }

        let mut offset = 0usize;
        for i in 0..self.rank {
            let v = idx[i];
            if v >= self.shape[i] {
                return Err(TensorError::OutOfBounds);
            }
            let step = v
                .checked_mul(self.strides[i])
                .ok_or(TensorError::Overflow)?;
            offset = offset.checked_add(step).ok_or(TensorError::Overflow)?;
        }
        Ok(offset)
    }

    pub fn get(&self, idx: &[usize]) -> Result<&T, TensorError> {
        let offset = self.compute_offset(idx)?;
        self.data.get(offset).ok_or(TensorError::OutOfBounds)
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: &[usize]) -> &T {
        let mut offset = 0usize;
        for i in 0..self.rank {
            offset += idx[i] * self.strides[i];
        }
        self.data.get_unchecked(offset)
    }

    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Self, TensorError> {
        if dim >= self.rank || start >= end || end > self.shape[dim] {
            return Err(TensorError::OutOfBounds);
        }

        let mut new_shape = self.shape;
        new_shape[dim] = end - start;

        let base = start
            .checked_mul(self.strides[dim])
            .ok_or(TensorError::Overflow)?;

        let data = self.data.get(base..).ok_or(TensorError::OutOfBounds)?;

        Ok(Self {
            data,
            shape: new_shape,
            strides: self.strides,
            rank: self.rank,
            _marker: PhantomData,
        })
    }

    pub fn narrow(&self, dim: usize, index: usize) -> Result<Self, TensorError> {
        if dim >= self.rank || index >= self.shape[dim] {
            return Err(TensorError::OutOfBounds);
        }

        let mut new_shape = self.shape;
        new_shape[dim] = 1;

        let base = index
            .checked_mul(self.strides[dim])
            .ok_or(TensorError::Overflow)?;

        let data = self.data.get(base..).ok_or(TensorError::OutOfBounds)?;

        Ok(Self {
            data,
            shape: new_shape,
            strides: self.strides,
            rank: self.rank,
            _marker: PhantomData,
        })
    }
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn new(
        data: &'a mut [T],
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, TensorError> {
        let rank = shape.len();
        if rank == 0 || rank > MAX_DIMS || strides.len() != rank {
            return Err(TensorError::InvalidShape);
        }

        let mut s = [0usize; MAX_DIMS];
        let mut st = [0usize; MAX_DIMS];
        for i in 0..rank {
            if shape[i] == 0 {
                return Err(TensorError::InvalidShape);
            }
            s[i] = shape[i];
            st[i] = strides[i];
        }

        let required = required_len(rank, &s, &st)?;
        if required > data.len() {
            return Err(TensorError::OutOfBounds);
        }

        Ok(Self {
            data,
            shape: s,
            strides: st,
            rank,
            _marker: PhantomData,
        })
    }

    pub fn from_contiguous(data: &'a mut [T], shape: &[usize]) -> Result<Self, TensorError> {
        let rank = shape.len();
        if rank == 0 || rank > MAX_DIMS {
            return Err(TensorError::InvalidShape);
        }

        let mut strides = [0usize; MAX_DIMS];
        let mut stride = 1usize;
        for i in (0..rank).rev() {
            strides[i] = stride;
            stride = stride.checked_mul(shape[i]).ok_or(TensorError::Overflow)?;
        }

        Self::new(data, shape, &strides[..rank])
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.rank]
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: &[usize]) -> &T {
        let mut offset = 0usize;
        for i in 0..self.rank {
            offset += idx[i] * self.strides[i];
        }
        self.data.get_unchecked(offset)
    }

    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, idx: &[usize]) -> &mut T {
        let mut offset = 0usize;
        for i in 0..self.rank {
            offset += idx[i] * self.strides[i];
        }
        self.data.get_unchecked_mut(offset)
    }

    pub fn set(&mut self, idx: &[usize], value: T) -> Result<(), TensorError> {
        if idx.len() != self.rank {
            return Err(TensorError::OutOfBounds);
        }

        let mut offset = 0usize;
        for i in 0..self.rank {
            let v = idx[i];
            if v >= self.shape[i] {
                return Err(TensorError::OutOfBounds);
            }
            let step = v
                .checked_mul(self.strides[i])
                .ok_or(TensorError::Overflow)?;
            offset = offset.checked_add(step).ok_or(TensorError::Overflow)?;
        }

        if offset >= self.data.len() {
            return Err(TensorError::OutOfBounds);
        }

        self.data[offset] = value;
        Ok(())
    }

    pub fn as_view(&self) -> TensorView<'_, T> {
        TensorView {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            rank: self.rank,
            _marker: PhantomData,
        }
    }
}

fn required_len(
    rank: usize,
    shape: &[usize; MAX_DIMS],
    strides: &[usize; MAX_DIMS],
) -> Result<usize, TensorError> {
    if rank == 0 {
        return Ok(0);
    }

    let mut last_index = 0usize;
    for i in 0..rank {
        let span = shape[i]
            .checked_sub(1)
            .ok_or(TensorError::Overflow)?
            .checked_mul(strides[i])
            .ok_or(TensorError::Overflow)?;
        last_index = last_index.checked_add(span).ok_or(TensorError::Overflow)?;
    }

    last_index
        .checked_add(1)
        .ok_or(TensorError::Overflow)
}
