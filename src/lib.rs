#![feature(portable_simd)]
use core::simd::Simd;
use core::{mem::size_of, slice};

// type Float = f32;
// use core::f32::consts::PI;

#[cfg(target_pointer_width = "64")]
type Float = f64;
#[cfg(target_pointer_width = "64")]
use core::f64::consts::PI;
#[cfg(target_pointer_width = "64")]
type C = complex_nums::C64;

#[cfg(target_pointer_width = "32")]
type Float = f32;
#[cfg(target_pointer_width = "32")]
use core::f32::consts::PI;
#[cfg(target_pointer_width = "32")]
type C = complex_nums::C32;

pub mod complex_nums;
// use complex_nums::ComplexNums;

const PI_2: Float = PI * 2.0;
const PI_4: Float = PI * 4.0;

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Sin,
    Cos,
    Tan,
    /// y = k + A * e^(j * 2PI * f * t)
    ///   = k + [ A * e^(2PI f t) ] * (cos1 + j * sin1)
    ///   = {k + [ A * e^(2PI f t) * cos1 ]}  +  {j * sin1 * [A * e^(2PI f t)]}
    ComplexExp,
}

#[repr(C)]
/// ref https://blog.csdn.net/hjx5200/article/details/107025477
pub struct Wave {
    /// The RIFF Chunk
    chunk_id: u32, // ASCII码“0x52494646”对应字母“RIFF”
    chunk_size: u32, // 块大小是指除去ChunkID与ChunkSize的剩余部分有多少字节数据。注意：小尾字节序数。
    format: u32, // ASCII码“0x57415645”对应字母“WAVE”。该块由两个子块组成，一个“fmt”块用于详细说明数据格式，一个“data”块包含实际的样本数据。

    /// The 'fmt' sub-chunk
    subchunk1_id: u32, // ASCII码“0x666d7420”对应字母“fmt ”。
    subchunk1_size: u32, // 如果文件采用PCM编码，则该子块剩余字节数为16。
    audio_format: u16, // 如果文件采用PCM编码(线性量化)，则AudioFormat=1。AudioFormat代表不同的压缩方式，表二说明了相应的压缩方式。
    num_channels: u16, // 声道数，单声道（Mono）为1,双声道（Stereo）为2。
    sample_rate: u32,
    byterate: u32,        // 传输速率，单位：Byte/s。
    blockalign: u16,      // 一个样点（包含所有声道）的字节数。
    bits_per_sample: u16, // 每个样点对应的位数。
    /// extra .. (略)

    /// The 'data' sub-chunk
    subchunk2_id: u32, // ASCII码“0x64617461”对应字母 “data”。
    subchunk2_size: u32,  // 实际样本数据的大小（单位：字节）。
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone)]
struct _TrigSig {
    sig_type: Type,
    freq: Float,
    amp: Float,
    offset: Float,
    op: fn(Float) -> Float,
    k: Float,
}

#[derive(Debug, Clone)]
pub struct TrigSig {
    // sig_type: Type,
    // freq: Float,
    // amp: Float,
    // offset: Float,
    /// sig_type, freq, amp, offset
    signals: Vec<_TrigSig>,
}

/// Trigonometric Sigal
impl TrigSig {
    #[inline]
    pub fn new(mut sig_type: Type, freq: Float, amp: Float, mut offset: Float, k: Float) -> Self {
        match sig_type {
            Type::Sin => {}
            Type::Cos => {
                offset += PI / 2.0;
                sig_type = Type::Sin;
            }
            Type::Tan => {}
            Type::ComplexExp => {}
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

    // pub fn to_complex_exp(mut self) -> Self {
    //     for s in self.signals {
    //         if s.sig_type != Type::ComplexExp {

    //         }
    //     }
    //     self
    // }

    // evaluate a result for time t (s)
    fn evaluate(&self, t: Float) -> Float {
        let two_pi_t = PI_2 * t;
        self.signals.iter().fold(0.0, |acc, x| {
            acc + x.k + x.amp * (x.op)(two_pi_t * x.freq + x.offset)
        })
    }

    // evaluate a result for time t (s)
    /// let x(t) = 2PI * f * t + offset
    /// y = k + A * e^(j * x(t) )
    ///   = k +  A * e^x(t)  * e^j
    ///   // = k + [ A * e^x(t) ] * (cos1 + j * sin1)
    ///   // = {k +  A * e^x(t) * cos1 }  +  { j * sin1 * A * e^x(t) }
    ///   = k + A * e^x(t) * ComplexNums::<Float>::exp_of_j()
    fn evaluate_complex(&self, t: Float) -> C {
        let two_pi_t = PI_2 * t;
        self.signals.iter().fold(C::default(), |acc, x| {
            acc + C::complex_exp(two_pi_t * x.freq + x.offset)
                .mul(x.amp)
                .add(x.k)
        })
    }

    /// duration: time length (s),
    ///
    /// start: time start (s),
    ///
    /// framerate: wave sample rate (Hz), such as 11025, 44100, 48000
    pub fn make_wave(&self, duration: Float, start: Float, framerate: usize) -> Wave {
        let nums = (duration * framerate as Float) as usize;

        // let t = (duration as Float) / (nums as Float);
        let t = (1 as Float) / (framerate as Float);

        let mut wave = Wave::new_default(vec![0; size_of::<Float>() * nums]);
        wave.set_num_channels(1);
        wave.set_samplerate(framerate as u32);
        wave.set_bits_per_sample(size_of::<Float>() as u16);
        wave.subchunk2_size = (size_of::<Float>() * nums) as u32;

        // let ys = wave.raw.as_mut_ptr() as *mut Float;

        for i in 0..wave.raw.len() / size_of::<Float>() {
            // unsafe { ys.add(i).write(self.evaluate(start + (i as Float) * t)) }
            wave.raw_as_mut_float()[i] = self.evaluate(start + (i as Float) * t);
        }

        println!(
            "Made a new wave, samples length = ({}*{}) {} bytes, f = {}, T = {}",
            size_of::<Float>(),
            wave.raw.len() / size_of::<Float>(),
            wave.raw.len(),
            self.freq(),
            self.cycle()
        );
        wave
    }

    /// duration: time length (s),
    ///
    /// start: time start (s),
    ///
    /// framerate: wave sample rate (Hz), such as 11025, 44100, 48000
    pub fn make_wave_complex_exp(&self, duration: Float, start: Float, framerate: usize) -> Wave {
        let nums = (duration * framerate as Float) as usize;

        println!("dbg: nums {}  duration {}", nums, duration);

        // let t = (duration as Float) / (nums as Float);
        let t = (1 as Float) / (framerate as Float);

        let mut wave = Wave::new_default(vec![0; size_of::<Float>() * nums]);
        wave.set_num_channels(1);
        wave.set_samplerate(framerate as u32);
        wave.set_bits_per_sample(size_of::<Float>() as u16);
        wave.subchunk2_size = (size_of::<Float>() * nums) as u32;

        // let ys = wave.raw.as_mut_ptr() as *mut Float;

        for i in 0..wave.raw.len() / size_of::<Float>() {
            // unsafe { ys.add(i).write(self.evaluate(start + (i as Float) * t)) }
            wave.raw_as_mut_float()[i] = self.evaluate_complex(start + (i as Float) * t).real;
        }

        println!(
            "Made a new wave, samples length = ({}*{}) {} bytes, f = {}, T = {}",
            size_of::<Float>(),
            wave.raw.len() / size_of::<Float>(),
            wave.raw.len(),
            self.freq(),
            self.cycle()
        );
        wave
    }

    #[inline]
    pub fn freq(&self) -> Float {
        if self.signals.len() > 1 {
            self.signals
                .iter()
                .fold(self.signals.first().unwrap().freq, |acc, i| {
                    (acc + i.freq) / 2.0
                })
        } else if self.signals.len() == 0 {
            0.0
        } else {
            self.signals.first().unwrap().freq
        }
    }

    #[inline]
    pub fn cycle(&self) -> Float {
        self.signals.len() as Float / self.freq()
    }
}

impl Wave {
    #[inline]
    pub fn raw_as_ref(&self) -> &[u8] {
        &self.raw
    }

    #[inline]
    pub fn raw_as_float(&self) -> &[Float] {
        unsafe {
            slice::from_raw_parts(
                self.raw.as_ptr() as *mut Float,
                self.raw.len() / size_of::<Float>(),
            )
        }
    }

    #[inline]
    pub fn raw_as_mut_float(&mut self) -> &mut [Float] {
        unsafe {
            slice::from_raw_parts_mut(
                self.raw.as_mut_ptr() as *mut Float,
                self.raw.len() / size_of::<Float>(),
            )
        }
    }

    #[inline]
    #[allow(unused)]
    pub fn raw_as_mut(&mut self) -> &mut [u8] {
        &mut self.raw
    }

    #[inline]
    /// 44100双声道 RIFF WAVE PCM/Uncompressed
    pub fn new_default(raw: Vec<u8>) -> Self {
        Self {
            chunk_id: u32::from_le_bytes(*b"RIFF"),     // b"RIFF"
            chunk_size: 9 * 4 + raw.len() as u32,       // 9 * 4B + sizeof raw
            format: u32::from_le_bytes(*b"WAVE"),       // 'WAVE' - 0x57415645
            subchunk1_id: u32::from_le_bytes(*b"fmt "), // b"fmt "
            subchunk1_size: 16,                         // pcm->16
            audio_format: 1,                            // pcm/uncompressed
            num_channels: 2,
            sample_rate: 44100,
            byterate: 44100 * 2 * 2, // sample_rate * num_channels * 2
            blockalign: 2 * size_of::<Float>() as u16, // num_channels * bits_persamples
            bits_per_sample: size_of::<Float>() as u16, // bits
            subchunk2_id: u32::from_le_bytes(*b"data"), // b"data"
            subchunk2_size: raw.len() as u32, // size of raw
            raw,
        }
    }

    #[inline]
    pub fn set_samplerate(&mut self, samplerate: u32) {
        self.sample_rate = samplerate;
        self.byterate = samplerate * self.num_channels as u32 * 2;
    }

    #[inline]
    pub fn set_bits_per_sample(&mut self, bits: u16) {
        self.bits_per_sample = bits;
        self.blockalign = bits * self.num_channels;
    }

    #[inline]
    pub fn set_num_channels(&mut self, nums: u16) {
        self.num_channels = nums;
        self.byterate = self.sample_rate * self.num_channels as u32 * 2;
        self.blockalign = self.num_channels * self.num_channels;
    }

    pub fn into_vec_u8(mut self) -> Vec<u8> {
        self.chunk_size += self.subchunk2_size;
        let mut v = vec![];
        v.extend_from_slice(&self.chunk_id.to_le_bytes());
        v.extend_from_slice(&self.chunk_id.to_le_bytes());
        v.extend_from_slice(&self.chunk_size.to_le_bytes());
        v.extend_from_slice(&self.format.to_le_bytes());
        v.extend_from_slice(&self.subchunk1_id.to_le_bytes());
        v.extend_from_slice(&self.subchunk1_size.to_le_bytes());
        v.extend_from_slice(&self.audio_format.to_le_bytes());
        v.extend_from_slice(&self.num_channels.to_le_bytes());
        v.extend_from_slice(&self.sample_rate.to_le_bytes());
        v.extend_from_slice(&self.byterate.to_le_bytes());
        v.extend_from_slice(&self.blockalign.to_le_bytes());
        v.extend_from_slice(&self.bits_per_sample.to_le_bytes());
        v.extend_from_slice(&self.subchunk2_id.to_le_bytes());
        v.extend_from_slice(&self.subchunk2_size.to_le_bytes());
        println!("len {}", v.len());
        v.append(&mut self.raw);
        println!("len {}, raw len {}", v.len(), self.subchunk2_size);
        v
    }

    pub fn dct_to_sig(&self) -> TrigSig {
        let ys = self.raw_as_float();

        // let mut amps: Vec<Float> = ys.to_vec();
        // rustdct::DctPlanner::new()
        //     .plan_dct8(ys.len())
        //     .process_dct8(&mut amps);

        // let mut amps = dct_naive_transform(&ys[..]);

        let amps = dct_iv_cpu_multi_threads(ys);

        println!(
            "\n----- len({})/SampleRate {}",
            ys.len(),
            (ys.len() as Float) / self.sample_rate as Float
        );

        let mut sig = TrigSig { signals: vec![] };
        for (i, amp) in amps.iter().enumerate() {
            // if i > 20 && i < 20000 {
            if *amp > 8.00 || *amp < -8.00 {
                sig.signals.push(_TrigSig {
                    sig_type: Type::Cos,
                    freq: (i as Float + 0.5)
                        * (self.sample_rate as Float / ys.len() as Float / 2.0),
                    // freq: i as Float * (SampleRate as Float / ys.len() as Float / 2.0),
                    amp: *amp * 2.0 / amps.len() as Float,
                    offset: 0.0,
                    op: Type::Cos.into(),
                    k: 0.0,
                });
            }
            // println!("---> {:?}", sig.signals.last());
            // }
        }
        println!("--->sig nums: {:?}", sig.signals.len());
        sig
    }
}

impl From<Type> for fn(Float) -> Float {
    fn from(t: Type) -> Self {
        match t {
            Type::Sin => Float::sin,
            Type::Cos => Float::cos,
            Type::Tan => Float::tan,
            // 。。。。。。等改版吧
            Type::ComplexExp => Float::sin,
            // Type::ComplexExp => complex_exp,
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

        sum += sigs;
        sum
    }
}

impl core::ops::AddAssign for TrigSig {
    fn add_assign(&mut self, rhs: Self) {
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
    // http://spiff.rit.edu/classes/phys207/lectures/beats/add_beats.html
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
            Type::ComplexExp => todo!(),
        };

        match sig.sig_type {
            Type::Sin => {}
            Type::Cos => {
                sig.offset += PI / 2.0;
                sig.sig_type = Type::Sin;
            }
            Type::Tan => unimplemented!(),
            Type::ComplexExp => todo!(),
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

pub fn inverse_dct(amps: &[Float]) -> Vec<Float> {
    // dct_iv(amps)
    dct_iv(amps)
        .into_iter()
        .map(|x| x * 2.0 / amps.len() as Float)
        .collect()
}

pub fn inverse_dct_multi_threads(amps: &[Float]) -> Vec<Float> {
    // dct_iv(amps)
    dct_iv_cpu_multi_threads(amps)
        .into_iter()
        .map(|x| x * 2.0 / amps.len() as Float)
        .collect()
}

pub fn inverse_dct_naive(amps: &[Float]) -> Vec<Float> {
    // dct_iv(amps).into_iter().map(|x| x * 2.0).collect()
    dct_naive_transform(amps)
        .into_iter()
        .map(|x| x * 2.0 / amps.len() as Float)
        .collect()
}

pub fn dct_naive_transform(vector: &[Float]) -> Vec<Float> {
    let mut result = Vec::<Float>::with_capacity(vector.len());
    let factor: Float = PI / (vector.len() as Float);
    for i in 0..vector.len() {
        let mut sum = 0.0;
        for j in 0..vector.len() {
            sum += vector[j] * (((j as Float) + 0.5) * (i as Float) * factor).cos();
        }
        result.push(sum);
    }
    result
}

#[inline]
/// info: assume len = n, the real frequents = samplerate * (0..n)+0.5 / n
pub fn dct_iv(v: &[Float]) -> Vec<Float> {
    #[cfg(any(windos, wasm, unix))]
    return dct_iv_cpu_multi_threads(v);

    #[cfg(not(any(windos, wasm, unix)))]
    return dct_iv_cpu(v);
}

/// info: assume len = n, the real frequents = samplerate * (0..n)+0.5 / n
pub fn dct_iv_cpu_multi_threads(v: &[Float]) -> Vec<Float> {
    let ys = v;
    let n = ys.len();

    println!("ys.len = {}, n = {}", ys.len(), n);

    let cpus = 12;
    let divs = n / 12;

    let mut ranges: Vec<_> = (0..cpus - 1).map(|i| i * divs..(i + 1) * divs).collect();
    ranges.push((cpus - 1) * divs..n);

    // prepare vectors
    let ts: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) / n as Float).collect();
    // info: assume len = n, the real frequents = samplerate * (0..n) / n
    let fs_2_pi: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) * PI).collect();
    let mut amps = vec![0.0; n];

    let ts_ptr = ts.as_ptr() as usize;
    let ts_len = ts.len();

    let fs_2_pi_ptr = fs_2_pi.as_ptr() as usize;
    let fs_2_pi_len = fs_2_pi.len();

    let ys_ptr = ys.as_ptr() as usize;
    let ys_len = ys.len();

    let amps_ptr = amps.as_mut_ptr() as usize;
    let amps_len = amps.len();

    let mut handles = vec![];
    for range in ranges {
        handles.push(std::thread::spawn(move || {
            // safe
            let ts = unsafe { slice::from_raw_parts(ts_ptr as *const Float, ts_len) };
            let fs_2_pi =
                unsafe { slice::from_raw_parts(fs_2_pi_ptr as *const Float, fs_2_pi_len) };
            let ys = unsafe { slice::from_raw_parts(ys_ptr as *const Float, ys_len) };

            let amps = unsafe { slice::from_raw_parts_mut(amps_ptr as *mut Float, amps_len) };

            for j in range {
                let mut amp = 0.0;
                for i in 0..n {
                    amp += (fs_2_pi[j] * ts[i]).cos() * ys[i]
                }
                amps[j] = amp;
            }
        }));
    }
    handles
        .into_iter()
        .for_each(|handle| handle.join().unwrap());

    amps
}

/// info: assume len = n, the real frequents = samplerate * (0..n)+0.5 / n
pub fn dct_iv_cpu(v: &[Float]) -> Vec<Float> {
    let ys = v;
    let n = ys.len();

    println!("ys.len = {}, n = {}", ys.len(), n);

    let ts: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) / n as Float).collect();
    // let fs: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) / 2.0).collect();
    // info: assume len = n, the real frequents = samplerate * (0..n) / n
    let fs_2_pi: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) * PI).collect();

    let mut amps = Vec::with_capacity(n);
    for f_2_pi in fs_2_pi.iter() {
        let mut amp = 0.0;
        for (i, t) in ts.iter().enumerate() {
            amp += (f_2_pi * t).cos() * ys[i]
        }
        amps.push(amp)
    }
    amps
}

// portable-simd没有提供cos方法，还是慢
/// info: assume len = n, the real frequents = samplerate * (0..n)+0.5 / n
pub fn dct_iv_simd(v: &[Float]) -> Vec<Float> {
    let ys = v;
    let n = ys.len();

    println!("ys.len = {}, n = {}", ys.len(), n);

    let mut amps = Vec::with_capacity(n);

    let ts: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) / n as Float).collect();
    // info: assume len = n, the real frequents = samplerate * (0..n) / n
    let fs_2_pi: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) * PI).collect();

    let (prefix, middle, suffix) = fs_2_pi.as_simd();

    println!("len: prefix {:?}, suffix {:?}", prefix.len(), suffix.len());

    if prefix.len() > 0 {
        for f_2_pi in prefix {
            let mut amp = 0.0;
            for i in 0..n {
                amp += (*f_2_pi * ts[i]).cos() * ys[i]
            }
            amps.push(amp)
        }
    }

    for f_2_pi in middle {
        let mut amp = Simd::from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for i in 0..n {
            let mut xs = Simd::from_array([ts[i], ts[i], ts[i], ts[i], ts[i], ts[i], ts[i], ts[i]])
                * *f_2_pi;

            xs.as_mut_array().iter_mut().for_each(|x| *x = x.cos());

            amp += xs * Simd::from_array([ys[i], ys[i], ys[i], ys[i], ys[i], ys[i], ys[i], ys[i]]);
        }
        amps.extend_from_slice(amp.as_array());
    }

    if suffix.len() > 0 {
        for f_2_pi in suffix {
            let mut amp = 0.0;
            for i in 0..n {
                amp += (*f_2_pi * ts[i]).cos() * ys[i]
            }
            amps.push(amp)
        }
    }
    amps
}

/// info: assume len = n, the real frequents = samplerate * (0..n) / n
pub fn dft(v: &[C]) -> Vec<C> {
    let ys = v;
    let n = ys.len();

    // println!("ys.len = {}, n = {}", ys.len(), n);

    let ts: Vec<Float> = (0..n).map(|x| (x as Float) / n as Float).collect();
    // let fs: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) / 2.0).collect();
    // info: assume len = n, the real frequents = samplerate * (0..n) / n
    let fs_2_pi: Vec<Float> = (0..n).map(|f| ((f) as Float) * PI_2).collect();

    let mut amps = Vec::with_capacity(n);
    for f_2_pi in fs_2_pi.iter() {
        let mut amp = C::default();
        for (i, t) in ts.iter().enumerate() {
            amp += C::complex_exp(-f_2_pi * t) * ys[i]
        }
        amps.push(amp)
    }
    amps
}

pub fn idft(v: &[C]) -> Vec<C> {
    let ys = v;
    let n = ys.len();

    // println!("ys.len = {}, n = {}", ys.len(), n);

    let ts: Vec<Float> = (0..n).map(|x| (x as Float) / n as Float).collect();
    // let fs: Vec<Float> = (0..n).map(|x| (x as Float + 0.5) / 2.0).collect();
    // info: assume len = n, the real frequents = samplerate * (0..n) / n
    let fs_2_pi: Vec<Float> = (0..n).map(|f| ((f) as Float) * PI_2).collect();

    let mut amps = Vec::with_capacity(n);
    for f_2_pi in fs_2_pi.iter() {
        let mut amp = C::default();
        for (i, t) in ts.iter().enumerate() {
            amp += C::complex_exp(f_2_pi * t) * ys[i]
        }
        amps.push(amp.div(n as Float))
    }
    amps
}

fn even_odds_from_n_complex(n: usize) -> Vec<C> {
    even_odds(&vec![(0..n).collect::<Vec<usize>>()])
        .into_iter()
        .map(|v| C::new(v[0] as Float, 0.0))
        .collect()
}

fn even_odds_from_n(n: usize) -> Vec<usize> {
    even_odds(&vec![(0..n).collect::<Vec<usize>>()])
        .into_iter()
        .map(|v| v[0])
        .collect()
}

// recursive, 可能会爆栈
fn even_odds(v: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut new = vec![];
    for i in v {
        if i.len() > 1 {
            for j in even_odds(&even_odd(i)) {
                new.push(j)
            }
        } else {
            new.push(i.clone())
        }
    }
    new
}

fn even_odd(x: &[usize]) -> Vec<Vec<usize>> {
    vec![
        x.iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, &x)| x)
            .collect(),
        x.iter()
            .enumerate()
            .filter(|(i, _)| i % 2 != 0)
            .map(|(_, &x)| x)
            .collect(),
    ]
}

/// assume len = n, the real frequents = samplerate * (0..n) / n
/// 不是recursive但内部有recursive，太大可能会爆栈
fn __fft_common(inverse: bool, v: &[C]) -> Vec<C> {
    if v.len() == 1 {
        return v.to_vec();
    }

    if v.len() == 2 {
        let mut amps = vec![C::default(); 2];
        amps[0] = v[0] + v[1];
        amps[1] = v[0] - v[1];
        return amps;
    }

    assert!(v.len().is_power_of_two());

    let sqrt_usize = |mut x: usize| {
        let mut n = 1_usize;
        // let mut x = v.len();
        loop {
            x /= 2;
            if x == 1 {
                break;
            }
            n += 1;
        }
        n
    };
    let n_sqrt_2 = sqrt_usize(v.len());

    let ys = v;
    let n = ys.len();

    let mut neg_pi = -(PI);
    if inverse {
        neg_pi = -neg_pi;
    }

    // even_odds_from_n_complex： recursive !
    let mut amps = even_odds_from_n_complex(n);
    let mut i_amps = 0;
    loop {
        let j_amps = i_amps + 1;
        let i_ys = amps[i_amps].real as usize;
        let j_ys = amps[j_amps].real as usize;

        amps[i_amps] = ys[i_ys] + ys[j_ys];
        amps[j_amps] = ys[i_ys] - ys[j_ys];

        i_amps += 2;
        if i_amps >= n {
            break;
        }
    }

    for i in 1..n_sqrt_2 {
        let nums_a_time = 2usize.pow(i as u32);

        // when f = 0
        let mut j = 0;
        loop {
            let tmp = amps[j];
            amps[j] = tmp + amps[j + nums_a_time];
            amps[j + nums_a_time] = tmp - amps[j + nums_a_time];
            j += nums_a_time + nums_a_time;
            if j < n {
                continue;
            }
            break;
        }

        let neg_pi_div_nums_a_time = neg_pi / nums_a_time as Float;

        for f in 1..nums_a_time {
            let w = C::complex_exp(neg_pi_div_nums_a_time * f as Float);
            let mut j = f;
            loop {
                let tmp = amps[j];
                amps[j] = tmp + w * amps[j + nums_a_time];
                amps[j + nums_a_time] = tmp - w * amps[j + nums_a_time];
                j += nums_a_time + nums_a_time;
                if j < n {
                    continue;
                }
                break;
            }
        }
    }

    amps
}

pub fn fft(v: &[C]) -> Vec<C> {
    if v.len().is_power_of_two() {
        __fft_common(false, v)
    } else {
        todo!()
    }
}

pub fn ifft(v: &[C]) -> Vec<C> {
    if v.len().is_power_of_two() {
        __fft_common(true, v)
    } else {
        todo!()
    }
    .into_iter()
    .map(|x| x.div(v.len() as Float))
    .collect()
}