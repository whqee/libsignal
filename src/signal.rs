use core::f32;
use core::f32::consts::PI;
use core::mem::size_of;

// trait Signal {
//     fn evaluate(&self, t: f32, op: fn(f32) -> f32) -> f32;
// }

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Sin,
    Cos,
    Tan,
    Mix,
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
pub struct TrigSig {
    sig_type: Type,
    freq: f32,
    /// case常数: amp, _, 0, _
    /// case函数: a1, a2, freq, offset
    amp: (f32, f32, f32, f32),

    /// case常数: offset, _, 0, _
    /// case函数: a1, a2, freq, offset
    offset: (f32, f32, f32, f32),
}

/// Trigonometric Sigal
impl TrigSig {
    #[inline]
    pub fn new(sig_type: Type, freq: f32, amp: f32, offset: f32) -> Self {
        TrigSig {
            sig_type,
            freq,
            amp: (amp, 0.0, 0.0, 0.0),
            offset: (offset, 0.0, 0.0, 0.0),
        }
    }

    #[allow(unused)]
    #[inline]
    pub fn new_mix(freq: f32, amp: (f32, f32, f32, f32), offset: (f32, f32, f32, f32)) -> Self {
        TrigSig {
            sig_type: Type::Mix,
            freq,
            amp,
            offset,
        }
    }

    // evaluate a result for time t (s)
    pub fn evaluate(&self, t: f32, op: fn(f32) -> f32) -> f32 {
        // self.amp * (op)(2.0 * PI * self.freq * t + self.offset)
        let amp = if self.amp.2 == 0.0 {
            self.amp.0
        } else {
            let (a1, a2, freq, of) = self.amp;

            (a1 + a2 * (freq * t + of).cos()).sqrt()
        };

        let offset = if self.offset.2 == 0.0 {
            self.offset.0
        } else {
            let (a1, a2, freq, of) = self.offset;

            // 法一，测试不对。
            // a1 + a2 * (freq * t + of).tan()

            // 二
            a1 + (a2 * (freq * t + of).tan()).atan()
        };

        amp * (op)(2.0 * PI * self.freq * t + offset)
    }

    /// duration: time length (s),
    ///
    /// start: time start (s),
    ///
    /// framerate: wave sample rate (Hz), such as 11025, 44100, 48000
    pub fn make_wave(&self, duration: f32, start: f32, framerate: usize) -> Wave {
        let nums = (duration * framerate as f32) as usize;

        let mut wave = Wave {
            raw: vec![0; nums * size_of::<f32>()],
        };

        let t = (duration as f32) / (nums as f32);

        // let op = self.sig_type.into();
        let op = match self.sig_type {
            Type::Sin => f32::sin,
            Type::Cos => f32::cos,
            Type::Tan => f32::tan,
            Type::Mix => f32::sin,
        };

        // wave.raw as Vec<f32>, and push values in, then update the length of raw as Vec<u8>
        {
            let ptr_f32 = wave.raw.as_mut_ptr() as *mut f32;

            for i in 0..nums {
                let val = self.evaluate(start + (i as f32) * t, op);

                unsafe { ptr_f32.add(i).write(val) }
            }
        }

        println!("Made a new wave, samples length = {} bytes", wave.raw.len());

        wave
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

impl core::ops::Add for TrigSig {
    type Output = TrigSig;

    #[allow(unused)]
    fn add(self, rhs: Self) -> Self::Output {
        // ! 'match' can not match 'fn(f32) -> f32', but 'if else' can do it
        // match sig_type
        match self.sig_type {
            Type::Sin => match rhs.sig_type {
                Type::Sin => signal_sin_add_sin(self, rhs),
                Type::Cos => signal_sin_add_cos(self, rhs),
                Type::Tan => todo!(),
                Type::Mix => todo!(),
            },
            Type::Cos => match rhs.sig_type {
                Type::Sin => signal_cos_add_sin(self, rhs),
                Type::Cos => signal_cos_add_cos(self, rhs),
                Type::Tan => todo!(),
                Type::Mix => todo!(),
            },
            Type::Tan => match rhs.sig_type {
                Type::Sin => todo!(),
                Type::Cos => todo!(),
                Type::Tan => todo!(),
                Type::Mix => todo!(),
            },
            Type::Mix => todo!(),
        }
    }
}

#[allow(unused)]
fn signal_sin_add_sin(sin1: TrigSig, sin2: TrigSig) -> TrigSig {
    assert_eq!(sin1.sig_type, Type::Sin);
    assert_eq!(sin2.sig_type, Type::Sin);

    let s1;
    let s2;
    if sin1.freq < sin2.freq {
        s1 = sin2;
        s2 = sin1;
    } else {
        s1 = sin1;
        s2 = sin2;
    }

    let (freq1, freq2, amp1, amp2, of1, of2) = (
        s1.freq,
        s2.freq,
        s1.amp.0,
        s2.amp.0,
        s1.offset.0,
        s2.offset.0,
    );

    if freq1.eq(&freq2) {
        let (freq, amp, offset);

        freq = freq1;
        if amp1.eq(&amp2) {
            if of1.eq(&of2) {
                amp = 2.0 * amp1;
                offset = of1;
            } else {
                amp = 2.0 * amp1 * (((of1 - of2) / 2.0).cos());
                offset = (of1 + of2) / 2.0;
            }
        } else {
            if of1.eq(&of2) {
                offset = of1;
                amp = (amp1.powf(2.0) + amp2.powf(2.0) + 2.0 * amp1 * amp2).sqrt();
            } else {
                offset = ((amp1 * of1.sin() + amp2 * of2.sin())
                    / (amp1 * of1.cos() + amp2 * of2.cos()))
                .atan();
                amp = (amp1.powf(2.0) + amp2.powf(2.0) + 2.0 * amp1 * amp2 * (of1 - of2).cos())
                    .sqrt();
            }
        }
        TrigSig {
            sig_type: Type::Sin,
            freq,
            amp: (amp, 0.0, 0.0, 0.0),
            offset: (offset, 0.0, 0.0, 0.0),
        }
    } else {
        let (freq, amp, offset);
        freq = (freq1 + freq2) / 2.0;

        amp = (
            amp1.powf(2.0) + amp2.powf(2.0),
            2.0 * amp1 * amp2,
            // 法一，测试不对
            // (freq1 + freq2) / 2.0,
            // of1 + of2,

            // 二
            (freq1 - freq2) / 2.0,
            of1 - of2,
        );

        offset = (
            (of1 + of2) / 2.0,
            // 一，不对
            // amp1 - amp2,
            // 二
            (amp1 - amp2) / (amp1 + amp2),
            (freq1 - freq2) / 2.0,
            (of1 - of2) / 2.0,
        );

        TrigSig {
            sig_type: Type::Mix,
            freq,
            amp,
            offset,
        }
    }
}

#[allow(unused)]
fn signal_cos_add_cos(mut cos1: TrigSig, mut cos2: TrigSig) -> TrigSig {
    assert_eq!(cos1.sig_type, Type::Cos);
    assert_eq!(cos2.sig_type, Type::Cos);

    cos1.offset.0 += PI / 2.0;
    cos2.offset.0 += PI / 2.0;

    cos1.sig_type = Type::Sin;
    cos2.sig_type = Type::Sin;

    signal_sin_add_sin(cos1, cos2)
}

fn signal_cos_add_sin(mut cos: TrigSig, sin: TrigSig) -> TrigSig {
    assert_eq!(sin.sig_type, Type::Sin);
    assert_eq!(cos.sig_type, Type::Cos);

    cos.offset.0 += PI / 2.0;

    cos.sig_type = Type::Sin;

    signal_sin_add_sin(cos, sin)
}

#[inline]
fn signal_sin_add_cos(sig1: TrigSig, sig2: TrigSig) -> TrigSig {
    signal_cos_add_sin(sig2, sig1)
}

// http://spiff.rit.edu/classes/phys207/lectures/beats/add_beats.html
