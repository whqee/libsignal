use core::f32;
use core::f32::consts::PI;
use core::mem::size_of;

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Sin,
    Cos,
    Tan,
}

pub struct Wave {
    // chunk_id: u32, // ASCII码“0x52494646”对应字母“RIFF”
    // chunk_size: u32, // 块大小是指除去ChunkID与ChunkSize的剩余部分有多少字节数据。注意：小尾字节序数。
    // format: u32, // ASCII码“0x57415645”对应字母“WAVE”。该块由两个子块组成，一个“fmt”块用于详细说明数据格式，一个“data”块包含实际的样本数据。
    // subchunk1_id: u32, // ASCII码“0x666d7420”对应字母“fmt ”。
    // subchunk1_size: u32, // 如果文件采用PCM编码，则该子块剩余字节数为16。
    // audio_format: u16, // 如果文件采用PCM编码(线性量化)，则AudioFormat=1。AudioFormat代表不同的压缩方式，表二说明了相应的压缩方式。
    // num_channels: u16, // 声道数，单声道（Mono）为1,双声道（Stereo）为2。
    // sample_rate: u32,
    // byterate: u32, // 传输速率，单位：Byte/s。
    // blockalign: u16, // 一个样点（包含所有声道）的字节数。
    // bits_per_sample: u16, // 每个样点对应的位数。
    // extra .. (略)
    // subchunk2_id: u32, // ASCII码“0x64617461”对应字母 “data”。
    // subchunk2_size: u32, // 实际样本数据的大小（单位：字节）。
    raw: Vec<u8>,
}

#[derive(Debug, Clone)]
struct _TrigSig {
    sig_type: Type,
    freq: f32,
    amp: f32,
    offset: f32,
    op: fn(f32) -> f32,
    k: f32,
}

#[derive(Debug, Clone)]
pub struct TrigSig {
    // sig_type: Type,
    // freq: f32,
    // amp: f32,
    // offset: f32,
    /// sig_type, freq, amp, offset
    signals: Vec<_TrigSig>,
}

/// Trigonometric Sigal
impl TrigSig {
    #[inline]
    pub fn new(mut sig_type: Type, freq: f32, amp: f32, mut offset: f32, k: f32) -> Self {
        match sig_type {
            Type::Sin => {}
            Type::Cos => {
                offset += PI / 2.0;
                sig_type = Type::Sin;
            }
            Type::Tan => {}
        };
        TrigSig {
            signals: vec![_TrigSig {
                sig_type,
                freq,
                amp,
                offset,
                op: sig_type.into(),
                k,
            }],
        }
    }

    // evaluate a result for time t (s)
    pub fn evaluate(&self, t: f32) -> f32 {
        self.signals.iter().fold(0.0, |acc, x| {
            acc + x.k + x.amp * (x.op)(2.0 * PI * x.freq * t + x.offset)
        })
    }

    /// duration: time length (s),
    ///
    /// start: time start (s),
    ///
    /// framerate: wave sample rate (Hz), such as 11025, 44100, 48000
    pub fn make_wave(&self, duration: f32, start: f32, framerate: usize) -> Wave {
        let nums = (duration * framerate as f32) as usize;

        let t = (duration as f32) / (nums as f32);

        let mut wave = Wave {
            raw: vec![0; size_of::<f32>() * nums], // Vec<u8>
        };

        // wave.raw as Vec<f32>, and push values in
        for i in 0..nums {
            unsafe {
                (wave.raw.as_mut_ptr() as *mut f32)
                    .add(i)
                    .write(self.evaluate(start + (i as f32) * t));
            }
        }

        println!(
            "Made a new wave, samples length = {} bytes, f = {}, T = {}",
            wave.raw.len(),
            self.freq_sum(),
            self.cycle()
        );

        wave
    }

    #[inline]
    pub fn freq_sum(&self) -> f32 {
        if self.signals.len() > 1 {
            self.signals
                .iter()
                .fold(self.signals.first().unwrap().freq, |acc, i| {
                    (acc + i.freq) / 2.0
                })
        } else {
            self.signals.first().unwrap().freq
        }
    }

    #[inline]
    pub fn cycle(&self) -> f32 {
        self.signals.len() as f32 / self.freq_sum()
    }
}

impl Wave {
    #[inline]
    pub fn raw_as_ref(&self) -> &[u8] {
        &self.raw
    }

    #[inline]
    #[allow(unused)]
    pub fn raw_as_mut(&mut self) -> &mut [u8] {
        &mut self.raw
    }
}

impl From<Type> for fn(f32) -> f32 {
    fn from(t: Type) -> Self {
        match t {
            Type::Sin => f32::sin,
            Type::Cos => f32::cos,
            Type::Tan => f32::tan,
        }
    }
}

impl core::ops::Add for TrigSig {
    type Output = TrigSig;

    // #[allow(unused)]
    fn add(self, rhs: TrigSig) -> Self::Output {
        let mut sum;
        let sigs;

        if self.signals.len() < rhs.signals.len() {
            sum = rhs;
            sigs = self;
        } else {
            sum = self;
            sigs = rhs;
        }

        // // ! 'match' can not match 'fn(f32) -> f32', but 'if else' can do it

        // for i in sigs.signals {
        //     if let Some(s) = sum.signals.iter_mut().find(|x| x.freq.eq(&i.freq)) {
        //         s.same_freq_add(&i);
        //     } else {
        //         sum.signals.push(i)
        //     }
        // }

        for s in sigs.signals {
            for i in 0..sum.signals.len() {
                if sum.signals[i].freq > s.freq {
                    sum.signals.insert(i, s);
                    break;
                }

                if sum.signals[i].freq == s.freq {
                    sum.signals[i].same_freq_add(&s);
                    break;
                }

                if i + 1 == sum.signals.len() {
                    sum.signals.push(s);
                    break;
                }
            }
        }

        sum
    }
}

impl core::ops::AddAssign for TrigSig {
    fn add_assign(&mut self, rhs: Self) {
        // for i in rhs.signals {
        //     if let Some(s) = self.signals.iter_mut().find(|x| x.freq.eq(&i.freq)) {
        //         s.same_freq_add(&i);
        //     } else {
        //         self.signals.push(i)
        //     }
        // }
        for s in rhs.signals {
            for i in 0..self.signals.len() {
                if self.signals[i].freq > s.freq {
                    self.signals.insert(i, s);
                    break;
                }

                if self.signals[i].freq == s.freq {
                    self.signals[i].same_freq_add(&s);
                    break;
                }

                if i + 1 == self.signals.len() {
                    self.signals.push(s);
                    break;
                }
            }
        }
    }
}

impl _TrigSig {
    fn same_freq_add(&mut self, sig: &_TrigSig) {
        assert_eq!(self.freq, sig.freq);

        let mut sig = sig.clone();

        match self.sig_type {
            Type::Sin => {}
            Type::Cos => {
                self.offset += PI / 2.0;
                self.sig_type = Type::Sin;
            }
            Type::Tan => unimplemented!(),
        };

        match sig.sig_type {
            Type::Sin => {}
            Type::Cos => {
                sig.offset += PI / 2.0;
                sig.sig_type = Type::Sin;
            }
            Type::Tan => unimplemented!(),
        };

        let (amp1, amp2, of1, of2) = (self.amp, sig.amp, self.offset, sig.offset);

        if amp1 == amp2 {
            if of1 == of2 {
                self.amp = 2.0 * amp1;
                self.offset = of1;
            } else {
                self.amp = 2.0 * amp1 * (((of1 - of2) / 2.0).cos());
                self.offset = (of1 + of2) / 2.0;
            }
        } else {
            if of1 == of2 {
                self.offset = of1;
                self.amp = (amp1.powf(2.0) + amp2.powf(2.0) + 2.0 * amp1 * amp2).sqrt();
            } else {
                self.offset = ((amp1 * of1.sin() + amp2 * of2.sin())
                    / (amp1 * of1.cos() + amp2 * of2.cos()))
                .atan();
                self.amp =
                    (amp1.powf(2.0) + amp2.powf(2.0) + 2.0 * amp1 * amp2 * (of1 - of2).cos())
                        .sqrt();
            }
        }

        self.k += sig.k
    }
}

// http://spiff.rit.edu/classes/phys207/lectures/beats/add_beats.html
