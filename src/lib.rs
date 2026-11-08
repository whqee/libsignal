#![feature(portable_simd)]
use core::mem::size_of;

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

const PI_2: Float = PI * 2.0;
// const PI_4: Float = PI * 4.0;

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
    pub chunk_id: u32, // ASCII码“0x52494646”对应字母“RIFF”
    pub chunk_size: u32, // 块大小是指除去ChunkID与ChunkSize的剩余部分有多少字节数据。注意：小尾字节序数。
    pub format: u32, // ASCII码“0x57415645”对应字母“WAVE”。该块由两个子块组成，一个“fmt”块用于详细说明数据格式，一个“data”块包含实际的样本数据。

    /// The 'fmt' sub-chunk
    pub subchunk1_id: u32, // ASCII码“0x666d7420”对应字母“fmt ”。
    pub subchunk1_size: u32, // 如果文件采用PCM编码，则该子块剩余字节数为16。
    pub audio_format: u16, // 如果文件采用PCM编码(线性量化)，则AudioFormat=1。AudioFormat代表不同的压缩方式，表二说明了相应的压缩方式。
    pub num_channels: u16, // 声道数，单声道（Mono）为1,双声道（Stereo）为2。
    pub sample_rate: u32,
    pub byterate: u32,        // 传输速率，单位：Byte/s。
    pub blockalign: u16,      // 一个样点（包含所有声道）的字节数。
    pub bits_per_sample: u16, // 每个样点对应的位数。
    /// extra .. (略)

    /// The 'data' sub-chunk
    pub subchunk2_id: u32, // ASCII码“0x64617461”对应字母 “data”。
    pub subchunk2_size: u32,  // 实际样本数据的大小（单位：字节）。
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
    pub fn evaluate(&self, t: Float) -> Float {
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
    pub fn evaluate_complex(&self, t: Float) -> C {
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

        let mut wave = Wave::new_default(Vec::with_capacity(size_of::<Float>() * nums));
        wave.set_num_channels(1);
        wave.set_samplerate(framerate as u32);
        wave.set_bits_per_sample((size_of::<Float>() * 8) as u16);
        for i in 0..nums {
            let sample = self.evaluate(start + (i as Float) * t);
            wave.raw.extend_from_slice(&sample.to_le_bytes());
        }
        wave.subchunk2_size = wave.raw.len() as u32;

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
    ///
    /// output: Vec<(t, y)>
    pub fn make_wave_with_time(
        &self,
        duration: Float,
        start: Float,
        framerate: usize,
    ) -> Vec<(Float, Float)> {
        let nums = (duration * framerate as Float) as usize;

        // let t = (duration as Float) / (nums as Float);
        let t = (1 as Float) / (framerate as Float);
        let mut wave = vec![(0.0, 0.0); nums];

        for i in 0..wave.len() {
            let t_now = (i as Float) * t;
            wave[i] = (t_now, self.evaluate(start + t_now));
        }
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

        let mut wave = Wave::new_default(Vec::with_capacity(size_of::<Float>() * nums));
        wave.set_num_channels(1);
        wave.set_samplerate(framerate as u32);
        wave.set_bits_per_sample((size_of::<Float>() * 8) as u16);
        for i in 0..nums {
            let sample = self.evaluate_complex(start + (i as Float) * t).real;
            wave.raw.extend_from_slice(&sample.to_le_bytes());
        }
        wave.subchunk2_size = wave.raw.len() as u32;

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
    /// Decode little-endian IEEE-754 samples into aligned values.
    /// Any trailing bytes that do not form a complete sample are ignored.
    #[inline]
    pub fn raw_as_float(&self) -> Vec<Float> {
        self.raw.chunks_exact(size_of::<Float>()).map(|chunk| {
            #[cfg(target_pointer_width = "64")]
            { f64::from_le_bytes(chunk.try_into().expect("exact sample width")) }
            #[cfg(target_pointer_width = "32")]
            { f32::from_le_bytes(chunk.try_into().expect("exact sample width")) }
        }).collect()
    }

    #[inline]
    #[allow(unused)]
    pub fn raw_as_mut(&mut self) -> &mut [u8] {
        &mut self.raw
    }

    #[inline]
    /// Construct a mono IEEE-float RIFF/WAVE file using the platform's Float type.
    pub fn new_default(raw: Vec<u8>) -> Self {
        Self {
            chunk_id: u32::from_le_bytes(*b"RIFF"),     // b"RIFF"
            chunk_size: 36 + raw.len() as u32,       // 9 * 4B + sizeof raw
            format: u32::from_le_bytes(*b"WAVE"),       // 'WAVE' - 0x57415645
            subchunk1_id: u32::from_le_bytes(*b"fmt "), // b"fmt "
            subchunk1_size: 16,                         // pcm->16
            audio_format: 3, // WAVE_FORMAT_IEEE_FLOAT
            num_channels: 1,
            sample_rate: 44100,
            byterate: 44100 * size_of::<Float>() as u32,
            blockalign: size_of::<Float>() as u16,
            bits_per_sample: (size_of::<Float>() * 8) as u16
            subchunk2_id: u32::from_le_bytes(*b"data"), // b"data"
            subchunk2_size: raw.len() as u32, // size of raw
            raw,
        }
    }

    #[inline]
    pub fn set_samplerate(&mut self, samplerate: u32) {
        self.sample_rate = samplerate;
        self.byterate = samplerate * self.blockalign as u32;
    }

    #[inline]
    pub fn set_bits_per_sample(&mut self, bits: u16) {
        self.bits_per_sample = bits;
        self.blockalign = (bits / 8) * self.num_channels;
        self.byterate = self.sample_rate * self.blockalign as u32;
    }

    #[inline]
    pub fn set_num_channels(&mut self, nums: u16) {
        self.num_channels = nums;
        self.blockalign = (self.bits_per_sample / 8) * nums;
        self.byterate = self.sample_rate * self.blockalign as u32;
    }

    pub fn into_vec_u8(mut self) -> Vec<u8> {
        self.subchunk2_size = u32::try_from(self.raw.len()).expect("WAV data chunk exceeds RIFF limit");
        self.chunk_size = 36u32.checked_add(self.subchunk2_size).expect("WAV file exceeds RIFF limit");
        let mut v = Vec::with_capacity(44 + self.raw.len());
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
        v.append(&mut self.raw);
        v
    }

    pub fn dct_to_sig(&self) -> TrigSig {
        let ys = self.raw_as_float();

        // let mut amps: Vec<Float> = ys.to_vec();
        // rustdct::DctPlanner::new()
        //     .plan_dct8(ys.len())
        //     .process_dct8(&mut amps);

        // let mut amps = dct_naive_transform(&ys[..]);

        let amps = dct_iv(&ys);
        // let amps = dct_iv_cpu_multi_threads(ys);

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
            Type::ComplexExp => {
                if sig.sig_type == Type::ComplexExp {
                    self.amp += sig.amp;
                    self.k += sig.k;
                    return;
                }
                unimplemented!()
            }
        };

        match sig.sig_type {
            Type::Sin => {}
            Type::Cos => {
                sig.offset += PI / 2.0;
                sig.sig_type = Type::Sin;
            }
            Type::Tan => unimplemented!(),
            Type::ComplexExp => unreachable!(),
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
    dct_iv(amps)
        // dct_iv_cpu_multi_threads(amps)
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
    #[cfg(any(windows, unix))]
    return dct_iv_cpu_multi_threads(v);

    #[cfg(not(any(windows, unix)))]
    return dct_iv_cpu(v);
}

#[cfg(any(windows, unix))]
/// info: assume len = n, the real frequents = samplerate * (0..n)+0.5 / n
pub fn dct_iv_cpu_multi_threads(v: &[Float]) -> Vec<Float> {
    let n = v.len();
    if n == 0 { return Vec::new(); }
    let workers = std::thread::available_parallelism().map(usize::from).unwrap_or(1).min(n);
    let chunk_size = n.div_ceil(workers);
    let mut output = vec![0.0; n];
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in output.chunks_mut(chunk_size).enumerate() {
            let start = chunk_index * chunk_size;
            scope.spawn(move || {
                for (offset, value) in chunk.iter_mut().enumerate() {
                    let frequency = (start + offset) as Float + 0.5;
                    let mut sum = 0.0;
                    for (sample_index, sample) in v.iter().enumerate() {
                        let time = (sample_index as Float + 0.5) / n as Float;
                        sum += (frequency * PI * time).cos() * sample;
                    }
                    *value = sum;
                }
            });
        }
    });
    output
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
    use core::simd::Simd;
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

/// assume len = n, the real frequents = samplerate * (0..n) / n
fn __fft_common(inverse: bool, v: &[C]) -> Vec<C> {
    let n = v.len();
    if n <= 1 { return v.to_vec(); }
    assert!(n.is_power_of_two(), "radix-2 FFT requires a power-of-two length");
    let shift = usize::BITS - n.trailing_zeros();
    let mut output = v.to_vec();
    for i in 0..n {
        let reversed = i.reverse_bits() >> shift;
        if reversed > i { output.swap(i, reversed); }
    }
    let direction = if inverse { 1.0 } else { -1.0 };
    let mut width = 2;
    while width <= n {
        let half = width / 2;
        for base in (0..n).step_by(width) {
            for k in 0..half {
                let angle = direction * PI_2 * k as Float / width as Float;
                let twiddle = C::complex_exp(angle);
                let even = output[base + k];
                let odd = twiddle * output[base + k + half];
                output[base + k] = even + odd;
                output[base + k + half] = even - odd;
            }
        }
        width *= 2;
    }
    output
}

/// Forward DFT. Uses radix-2 FFT for powers of two and direct DFT otherwise.
pub fn fft(v: &[C]) -> Vec<C> {
    if v.len().is_power_of_two() {
        __fft_common(false, v)
    } else {
        dft(v)
    }
}

/// Inverse DFT. Uses radix-2 FFT for powers of two and direct IDFT otherwise.
pub fn ifft(v: &[C]) -> Vec<C> {
    if v.is_empty() {
        return Vec::new();
    }
    if v.len().is_power_of_two() {
        let mut output = __fft_common(true, v);
        let scale = v.len() as Float;
        for value in &mut output {
            *value = value.div(scale);
        }
        output
    } else {
        idft(v)
    }
}

/// Forward transform of real-valued input, represented as a full complex spectrum.
pub fn rfft(v: &[Float]) -> Vec<C> {
    fft(&(v.iter().map(|&x| C::new(x, 0.0)).collect::<Vec<C>>()))
}

/// Inverse of a full complex spectrum produced by rfft.
pub fn irfft(v: &[C]) -> Vec<Float> {
    ifft(v).into_iter().map(|value| value.real).collect()
}

// slower
pub fn correlate_padded(ys: &[Float], window: &[Float]) -> Vec<Float> {
    correlate(ys, window)
}

pub fn correlate(ys: &[Float], window: &[Float]) -> Vec<Float> {
    let mut output = vec![0.0; ys.len()];
    if window.is_empty() { return output; }
    for n in 0..ys.len() {
        for m in 0..window.len().min(n + 1) {
            output[n] += ys[n - m] * window[window.len() - 1 - m];
        }
    }
    output
}

/// Circular Version
pub fn convolve2(ys: &[Float], window: &[Float]) -> Vec<Float> {
    let window: Vec<Float> = window.iter().rev().map(|x| *x).collect();
    correlate(ys, &window)
}

pub fn convolve(ys: &[Float], window: &[Float]) -> Vec<Float> {
    let mut out = vec![0.0; ys.len()];
    for n in 0..ys.len() {
        for m in 0..window.len().min(n + 1) {
            out[n] += ys[n - m] * window[m];
        }
    }
    out
}

/// FFT Version
pub fn fft_correlate(ys: &[Float], window: &[Float]) -> Vec<Float> {
    let pad_zero = |v: &mut Vec<Float>, n: usize| v.extend(vec![0.0; n]);

    let mut ys_padded = ys.to_owned();
    let mut padded = window.to_owned();
    // pad 0
    pad_zero(&mut ys_padded, ys.len());
    pad_zero(&mut padded, ys_padded.len() - window.len());

    let rfft_padded = rfft(&padded);
    let mut rfft_ys = rfft(&ys_padded);

    for i in 0..rfft_ys.len() {
        rfft_ys[i] *= rfft_padded[i];
    }
    irfft(&rfft_ys)[..ys.len()].to_owned()
}

#[inline]
pub fn fft_convolve(ys: &[Float], window: &[Float]) -> Vec<Float> {
    let window: Vec<Float> = window.iter().rev().map(|x| *x).collect();
    fft_correlate(ys, &window)
}

/// Parameters: n:
///                 Number of points in the output window.
///                 If zero, an empty array is returned.
///                 An exception is thrown when it is negative.
///             std:
///                 The standard deviation, sigma.
///
pub fn gaussian_window(n: usize, std: Float) -> Vec<Float> {
    let t = (n as Float - 1.0) / 2.0;
    let sig2 = 2.0 * std * std;

    let mut out = vec![0.0; n];

    for i in 0..n {
        out[i] = (-(i as Float - t).powi(2) / sig2).exp()
    }
    out
}

pub fn derivative() {}

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn close(left: Float, right: Float, tolerance: Float) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    #[test]
    fn fft_matches_direct_dft_for_radix_two_lengths() {
        for n in [1usize, 2, 4, 8, 16, 32] {
            let input: Vec<C> = (0..n)
                .map(|i| C::new(i as Float * 0.25 - 1.0, (i as Float * 0.3).sin()))
                .collect();
            let expected = dft(&input);
            let actual = fft(&input);
            for (left, right) in actual.iter().zip(expected.iter()) {
                close(left.real, right.real, if size_of::<Float>() == 4 { 1e-4 } else { 1e-8 });
                close(left.image, right.image, if size_of::<Float>() == 4 { 1e-4 } else { 1e-8 });
            }
        }
    }

    #[test]
    fn fft_and_ifft_round_trip_for_arbitrary_lengths() {
        for n in [0usize, 1, 2, 3, 5, 6, 8, 10, 17] {
            let input: Vec<C> = (0..n)
                .map(|i| C::new(i as Float * 0.125, (i as Float * 0.2).cos()))
                .collect();
            let output = ifft(&fft(&input));
            assert_eq!(output.len(), input.len());
            for (left, right) in output.iter().zip(input.iter()) {
                close(left.real, right.real, 1e-8);
                close(left.image, right.image, 1e-8);
            }
        }
    }

    #[test]
    fn dct_parallel_matches_single_threaded_reference() {
        for n in [0usize, 1, 2, 7, 12, 31] {
            let input: Vec<Float> = (0..n).map(|i| (i as Float * 0.17).sin()).collect();
            let expected = dct_iv_cpu(&input);
            let actual = dct_iv_cpu_multi_threads(&input);
            for (left, right) in actual.iter().zip(expected.iter()) {
                close(*left, *right, if size_of::<Float>() == 4 { 1e-4 } else { 1e-8 });
            }
        }
    }

    #[test]
    fn wav_float_header_is_consistent() {
        let mut wave = Wave::new_default(Vec::new());
        wave.set_num_channels(1);
        wave.set_samplerate(44_100);
        wave.set_bits_per_sample(64);
        wave.raw.extend_from_slice(&((1.25 as Float).to_le_bytes()));
        let bytes = wave.into_vec_u8();

        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 3);
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1);
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 44_100 * size_of::<Float>() as u32);
        assert_eq!(u16::from_le_bytes([bytes[32], bytes[33]]), size_of::<Float>() as u16);
        assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), (size_of::<Float>() * 8) as u16);
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), size_of::<Float>() as u32);
        assert_eq!(bytes.len(), 44 + size_of::<Float>());
    }

    #[test]
    fn wav_float_samples_decode_without_alignment_assumptions() {
        let samples: [Float; 3] = [0.5, -2.0, 3.25];
        let raw: Vec<u8> = samples.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        let wave = Wave::new_default(raw);
        assert_eq!(wave.raw_as_float(), samples.to_vec());
    }

    #[test]
    fn convolution_handles_empty_and_oversized_windows() {
        assert_eq!(convolve(&[1.0, 2.0], &[]), vec![0.0, 0.0]);
        assert_eq!(convolve(&[1.0, 2.0], &[3.0, 4.0, 5.0]), vec![3.0, 10.0]);
    }
}

