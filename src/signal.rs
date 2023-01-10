use core::f32::consts::PI;
use core::mem::size_of;

#[derive(Debug)]
pub enum Type {
    Sin,
    Cos,
    Tan,
}

pub struct Wave {
    raw: Vec<u8>,
}

pub struct Sin;
pub struct Cos;
pub struct Tan;

#[derive(Debug, Clone, Copy)]
pub struct TrigonometricSigal {
    freq: usize,
    amp: f32,
    offset: f32,
    op: fn(f32) -> f32,
}

#[inline]
fn sin(x: f32) -> f32 {
    x.sin()
}

#[inline]
fn cos(x: f32) -> f32 {
    x.cos()
}

#[inline]
fn tan(x: f32) -> f32 {
    x.tan()
}

impl Sin {
    #[inline]
    pub fn new(freq: usize, amp: f32, offset: f32) -> TrigonometricSigal {
        TrigonometricSigal::new(Type::Sin, freq, amp, offset)
    }
}

impl Cos {
    #[inline]
    pub fn new(freq: usize, amp: f32, offset: f32) -> TrigonometricSigal {
        TrigonometricSigal::new(Type::Cos, freq, amp, offset)
    }
}

impl Tan {
    #[allow(unused)]
    #[inline]
    pub fn new(freq: usize, amp: f32, offset: f32) -> TrigonometricSigal {
        TrigonometricSigal::new(Type::Tan, freq, amp, offset)
    }
}

impl TrigonometricSigal {
    #[inline]
    pub fn new(sig: Type, freq: usize, amp: f32, offset: f32) -> Self {
        TrigonometricSigal {
            freq,
            amp,
            offset,
            op: sig.into(),
        }
    }

    // evaluate a result for time t (s)
    pub fn evaluate(&self, t: f32) -> f32 {
        self.amp * (self.op)(2.0 * PI * (self.freq as f32) * t + self.offset)
    }

    /// duration: time length (s),
    ///
    /// start: time start (s),
    ///
    /// framerate: wave sample rate (Hz), such as 11025, 44100, 48000
    pub fn make_wave(&self, duration: f32, start: f32, framerate: usize) -> Wave {
        let nums = (duration * framerate as f32) as usize;

        let mut wave = Wave::new_with_capacity(nums);

        let t = (duration as f32) / (nums as f32);

        // wave.raw as Vec<f32>, and push values in, then update the length of raw as Vec<u8>
        {
            let ptr_f32 = wave.raw.as_mut_ptr() as *mut f32;

            for i in 0..nums {
                let val = self.evaluate(start + (i as f32) * t);

                unsafe { ptr_f32.add(i).write(val) }
            }

            unsafe { wave.raw.set_len(nums * size_of::<f32>()) }
        }

        println!("Made a new wave, samples length = {} bytes", wave.raw.len());

        wave
    }
}

impl Wave {
    #[inline]
    fn new_with_capacity(capacity: usize) -> Wave {
        Wave {
            raw: Vec::with_capacity(capacity * size_of::<f32>()),
        }
    }

    #[inline]
    pub fn raw_as_ref(&self) -> &[u8] {
        &self.raw
    }
}

impl From<Type> for fn(f32) -> f32 {
    fn from(t: Type) -> Self {
        match t {
            Type::Sin => sin,
            Type::Cos => cos,
            Type::Tan => tan,
        }
    }
}

// Not 100% safe to user
// impl From<&'static Wave> for &[u8] {
//     fn from(w: &'static Wave) -> Self {
//         // unsafe {
//         //     core::slice::from_raw_parts(
//         //         w.raw.as_slice().as_ptr() as *const u8,
//         //         w.raw.len() * core::mem::size_of::<f32>(),
//         //     )
//         // }
//         &w.raw
//     }
// }

impl core::ops::Add for TrigonometricSigal {
    type Output = TrigonometricSigal;

    #[allow(unused)]
    fn add(self, rhs: Self) -> Self::Output {
        if self.op == sin {
            if rhs.op == sin {
                return signal_sin_add_sin(self, rhs);
            } else if rhs.op == cos {
                return signal_sin_add_cos(self, rhs);
            } else {
                // tan()
                unimplemented!()
            }
        } else if self.op == cos {
            if rhs.op == sin {
                return signal_cos_add_sin(self, rhs);
            } else if rhs.op == cos {
                return signal_cos_add_cos(self, rhs);
            } else {
                // tan()
                unimplemented!()
            }
        } else {
            // branch start with tan()
            unimplemented!()
        }
        // !!! 'match' can not match 'fn(f32) -> f32', but 'if else' can do it
    }
}

#[allow(unused)]
fn signal_sin_add_sin(sin1: TrigonometricSigal, sin2: TrigonometricSigal) -> TrigonometricSigal {
    todo!()
}

#[allow(unused)]
fn signal_cos_add_cos(cos1: TrigonometricSigal, cos2: TrigonometricSigal) -> TrigonometricSigal {
    todo!()
}

fn signal_cos_add_sin(cos: TrigonometricSigal, sin: TrigonometricSigal) -> TrigonometricSigal {
    let (freq, amp, offset, op);
    op = Type::Cos.into();

    if cos.freq == sin.freq {
        // freq same
        freq = cos.freq;
        amp = (cos.amp.powf(2.0) + sin.amp.powf(2.0)
            - 2.0 * cos.amp * sin.amp * (cos.offset - sin.offset))
            .sqrt();
        offset = ((cos.amp * cos.offset.sin() - sin.amp * sin.offset.cos())
            / (cos.amp * cos.offset.cos() + sin.amp * sin.offset.sin()))
        .atan();
    } else {
        // not the same, and i don't know how to 'calc it
        todo!()
    }

    TrigonometricSigal {
        freq,
        amp,
        offset,
        op,
    }
}

#[inline]
fn signal_sin_add_cos(sig1: TrigonometricSigal, sig2: TrigonometricSigal) -> TrigonometricSigal {
    signal_cos_add_sin(sig2, sig1)
}
