#![feature(portable_simd)]

use std::{fs::File, io::Write};

use signal::{dct_iv, inverse_dct, Wave};

use crate::signal::{dct_iv_cpu, dct_iv_cpu_multi_threads, dct_iv_simd, inverse_dct_multi_threads};

mod signal;

type Float = f64;

fn main() {
    // let (freq, amp, offset, k) = (880.0, 0.5, 0.0, 0.0);
    // let sin_sig = signal::TrigSig::new(signal::Type::Sin, freq, amp, offset, k);

    // let (freq, amp, offset, k) = (440.0, 1.0, 0.0, 0.0);
    // let cos_sig = signal::TrigSig::new(signal::Type::Cos, freq, amp, offset, k);

    // let tan_sig = signal::Tan::new(freq, amp, offset, k);
    // let mut mix_sig = sin_sig.clone() + cos_sig.clone();

    // let (freq, amp, offset, k) = (440.0, 0.5, 0.2, 0.0);
    // let mut mix_sig = signal::TrigSig::new(signal::Type::Sin, freq, amp, offset, k);
    // let (freq, amp, offset, k) = (440.0, 0.5, 0.0, 0.0);

    // // 方波
    // for i in [
    //     3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
    // ] {
    //     mix_sig += signal::TrigSig::new(signal::Type::Sin, freq * i, amp / i, offset);
    // }

    // // 方波
    // for i in (2..2000).filter(|x| x % 2 != 0) {
    //     let i = i as Float;
    //     mix_sig += signal::TrigSig::new(signal::Type::Sin, freq * i, amp / i, offset, 0.0);
    // }

    // // 三角波
    // for i in (2..=2000).map(|x| x as Float) {
    //     mix_sig += signal::TrigSig::new(signal::Type::Sin, freq * i, amp / i, offset, k / i);
    // }

    // let (duration, start, framerate) = (0.1, 0.0, 44100);
    // let sin_wave = sin_sig.make_wave(duration, start, framerate);
    // let cos_wave = cos_sig.make_wave(duration, start, framerate);
    // // let tan_wave = tan_sig.make_wave(duration, start, framerate);
    // let mix_wave = mix_sig.make_wave(duration, start, framerate);

    // let dct_wave = mix_wave.dct_to_sig().make_wave(duration, start, framerate);

    // File::create("test.wav").unwrap().write(&mix_wave.into_vec_u8()).unwrap();
    // File::create("dct.wav").unwrap().write(&dct_wave.into_vec_u8()).unwrap();

    let read = std::fs::read("小提琴.raw").unwrap();
    // let read = std::fs::read("东风破.raw").unwrap();

    let mut wav = Wave::new_default(read[20000..(read.len() / 4)].to_vec());
    // let mut wav = Wave::new_default(read);
    wav.set_num_channels(1);

    let sig = wav.dct_to_sig();

    let (_duration, start, framerate) = (0.1, 0.0, 44100);
    let sig_wav = sig.make_wave(
        wav.raw_as_float().len() as Float / framerate as Float,
        start,
        framerate,
    );
    let music_dct_wave = sig_wav.raw_as_ref();
    File::create("小提琴 dct-to-sig-remake-wave.raw")
        .unwrap()
        .write(&music_dct_wave)
        .unwrap();

    let ys = wav.raw_as_mut_float();

    let mut count = 0;

    // let now = std::time::SystemTime::now();
    // let mut amps = dct_iv_simd(&ys[..]);
    // println!("simd dct time spent {}", now.elapsed().unwrap().as_millis());

    let now = std::time::SystemTime::now();
    let mut amps = dct_iv_cpu_multi_threads(&ys[..]);
    println!("cpu dct time spent {}", now.elapsed().unwrap().as_millis());

    for i in 0..amps.len() {
        if amps[i] < 8.50 && amps[i] > -8.50 {
            amps[i] = 0.0;
            count += 1;
        } else {
            // print!("amps[{}]:{} ", i, amps[i]);
        }
    }
    println!("left count: {}", amps.len() - count);

    let music_dct_wave = inverse_dct_multi_threads(&amps);

    let music_dct_wave = unsafe {
        core::slice::from_raw_parts(
            music_dct_wave.as_ptr() as *mut u8,
            music_dct_wave.len() * 8 as usize,
        )
    };

    File::create("小提琴 dct.raw")
        .unwrap()
        .write(&music_dct_wave)
        .unwrap();

    // file.write(sin_wave.raw_as_ref()).unwrap();
    // file.write(cos_wave.raw_as_ref()).unwrap();
    // file.write(tan_wave.raw_as_ref()).unwrap();
}
