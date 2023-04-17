#![feature(portable_simd)]
use core::simd;
use std::f64::consts::PI;
use std::ops::Deref;
use std::ptr::slice_from_raw_parts_mut;
use std::{fs::File, io::Write};

use signal::complex_nums::{ComplexNums, C32};
use signal::{dct_iv, dft, inverse_dct, irfft, rfft, Wave};

use signal::{dct_iv_cpu, dct_iv_cpu_multi_threads, dct_iv_simd, inverse_dct_multi_threads};

type Float = f64;

fn main() {
    {
        let (freq, amp, offset, k) = (880.0, 0.5, 0.0, 0.0);
        // let sin_sig = signal::TrigSig::new(signal::Type::Sin, freq, amp, offset, k);
        let cos_sig = signal::TrigSig::new(signal::Type::Cos, freq, amp, offset, k);

        let complex_cos =
            signal::TrigSig::new(signal::Type::ComplexExp, freq, amp / 2.0, offset, k)
                + signal::TrigSig::new(signal::Type::ComplexExp, -freq, amp / 2.0, offset, k);

        // let (freq, amp, offset, k) = (440.0, 1.0, 0.0, 0.0);

        // let tan_sig = signal::Tan::new(freq, amp, offset, k);
        // let mut mix_sig = sin_sig.clone() + cos_sig.clone();

        // let (freq, amp, offset, k) = (440.0, 0.5, 0.2, 0.0);
        // let mut mix_sig = signal::TrigSig::new(signal::Type::Sin, freq, amp, offset, k);
        // let (freq, amp, offset, k) = (440.0, 0.5, 0.0, 0.0);

        // // 方波
        // for i in [
        //     3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
        // ] {
        //     mix_sig += signal::TrigSig::new(signal::Type::Sin, freq * i, amp / i, offset, k / i);
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

        let (duration, start, framerate) = (0.1, 0.0, 44100);
        let cos_wave = cos_sig.make_wave(duration, start, framerate);
        let complex_cos_wave = complex_cos.make_wave_complex_exp(duration, start, framerate);
        // let cos_wave = cos_sig.make_wave(duration, start, framerate);
        // // let tan_wave = tan_sig.make_wave(duration, start, framerate);
        // let mix_wave = mix_sig.make_wave(duration, start, framerate);

        File::create("cos.raw")
            .unwrap()
            .write(&cos_wave.raw_as_ref())
            .unwrap();

        File::create("complex_cos.raw")
            .unwrap()
            .write(&complex_cos_wave.raw_as_ref())
            .unwrap();

        // let dct_wave = mix_wave.dct_to_sig().make_wave(duration, start, framerate);
        // File::create("test.wav")
        //     .unwrap()
        //     .write(&mix_wave.into_vec_u8())
        //     .unwrap();
        // File::create("dct.wav")
        //     .unwrap()
        //     .write(&dct_wave.into_vec_u8())
        //     .unwrap();

        // let read = std::fs::read("小提琴.raw").unwrap();
        // // let read = std::fs::read("东风破.raw").unwrap();

        // let mut wav = Wave::new_default(read[20000..(read.len() / 4)].to_vec());
        // // let mut wav = Wave::new_default(read);
        // wav.set_num_channels(1);

        // let sig = wav.dct_to_sig();

        // let (_duration, start, framerate) = (0.1, 0.0, 44100);
        // let sig_wav = sig.make_wave(
        //     wav.raw_as_float().len() as Float / framerate as Float,
        //     start,
        //     framerate,
        // );
        // let music_dct_wave = sig_wav.raw_as_ref();
        // File::create("小提琴 dct-to-sig-remake-wave.raw")
        //     .unwrap()
        //     .write(&music_dct_wave)
        //     .unwrap();

        // let ys = wav.raw_as_mut_float();

        // let mut count = 0;

        // // let now = std::time::SystemTime::now();
        // // let mut amps = dct_iv_simd(&ys[..]);
        // // println!("simd dct time spent {}", now.elapsed().unwrap().as_millis());

        // let now = std::time::SystemTime::now();
        // let mut amps = dct_iv_cpu_multi_threads(&ys[..]);
        // println!("cpu dct time spent {}", now.elapsed().unwrap().as_millis());

        // for i in 0..amps.len() {
        //     if amps[i] < 8.50 && amps[i] > -8.50 {
        //         amps[i] = 0.0;
        //         count += 1;
        //     } else {
        //         // print!("amps[{}]:{} ", i, amps[i]);
        //     }
        // }
        // println!("left count: {}", amps.len() - count);

        // let music_dct_wave = inverse_dct_multi_threads(&amps);

        // let music_dct_wave = unsafe {
        //     core::slice::from_raw_parts(
        //         music_dct_wave.as_ptr() as *mut u8,
        //         music_dct_wave.len() * 8 as usize,
        //     )
        // };

        // File::create("小提琴 dct.raw")
        //     .unwrap()
        //     .write(&music_dct_wave)
        //     .unwrap();
    }

    // fft test
    {
        use signal::complex_nums::C64;
        let v = vec![
            C64::new(2.4, 0.1),
            C64::new(1.0, 0.1),
            C64::new(0.4, 0.1),
            C64::new(0.2, 0.1),
            C64::new(0.2, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(2.4, 0.1),
            C64::new(1.0, 0.1),
            C64::new(0.4, 0.1),
            C64::new(0.2, 0.1),
            C64::new(0.2, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(2.4, 0.1),
            C64::new(1.0, 0.1),
            C64::new(0.4, 0.1),
            C64::new(0.2, 0.1),
            C64::new(0.2, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(2.4, 0.1),
            C64::new(1.0, 0.1),
            C64::new(0.4, 0.1),
            C64::new(0.2, 0.1),
            C64::new(0.2, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            // C64::new(0.0, 0.0),
        ];
        // println!(
        //     "v.conj {:#?}",
        //     v.iter()
        //         .map(|x| x.conj())
        //         .collect::<Vec<ComplexNums<Float>>>()
        // );

        let dft_out = signal::dft(&v);
        println!("dft {:#?}", dft_out);

        let fft_out = signal::fft(&v);
        println!("fft {:#?}", fft_out);

        let idft_out = signal::idft(&dft_out);
        // idft_out.iter_mut()
        // .for_each(|x| (*x) = (*x).div(v1.len() as Float));
        println!("idft {:#?}", idft_out);

        let ifft_out = signal::ifft(&fft_out);
        // ifft_out.iter_mut()
        // .for_each(|x| (*x) = (*x).div(v1.len() as Float));
        println!("ifft {:#?}", ifft_out);
    }

    // dct, rfft comparing
    {
        let mut v = (0..4096 * 128).map(|x| x as Float).collect::<Vec<Float>>();
        println!("\nv.len = {}\n", v.len());

        // let now = std::time::SystemTime::now();

        // let v_dct = dct_iv(&v);

        // println!(
        //     "12_thread dct_iv time spent {} ms",
        //     now.elapsed().unwrap().as_millis()
        // );
        // let now = std::time::SystemTime::now();

        // let v_idct = inverse_dct(&v_dct);

        // println!(
        //     "12_thread idct_iv time spent {} ms\n",
        //     now.elapsed().unwrap().as_millis()
        // );

        let now = std::time::SystemTime::now();

        let v_rfft = rfft(&v);

        println!(
            "1 thread rfft time spent {} ms",
            now.elapsed().unwrap().as_millis()
        );
        let now = std::time::SystemTime::now();

        let v_irfft = irfft(&v_rfft);

        println!(
            "1 thread irfft time spent {} ms",
            now.elapsed().unwrap().as_millis()
        );

        // println!("dct {:?}\nrfft {:?}\n", v_dct, v_rfft);
        // println!("idct {:?}\nirfft {:?}", v_idct, v_irfft);

        use rustfft::num_complex::Complex;
        let mut planner = rustfft::FftPlanner::<f64>::new();
        let rust_fft = planner.plan_fft_forward(4096 * 2);

        let now = std::time::SystemTime::now();

        let mut buffer = v
            .iter()
            .map(|&re| Complex { re, im: 0.0 })
            .collect::<Vec<Complex<f64>>>();
        planner.plan_fft_forward(buffer.len()).process(&mut buffer);

        println!(
            "RustFFT v6.1.0 (SIMD-accelerated) rfft time spent {} ms",
            now.elapsed().unwrap().as_millis()
        );
        let now = std::time::SystemTime::now();

        planner.plan_fft_inverse(buffer.len()).process(&mut buffer);
        
        let out = buffer.into_iter().map(|x| x.re).collect::<Vec<Float>>();
    }

    // {
    //     let now = std::time::SystemTime::now();
    //     for n in (0..16) {
    //         println!("{:?}", (1..n).product::<usize>())
    //         // println!("{:?}", FACTORIALS[n])
    //         // println!("{}", (1..n).fold(1.0, |acc, x| acc * x as Float))
    //     }
    //     println!(
    //         "n in 0..16 product time spent {} us",
    //         now.elapsed().unwrap().as_micros()
    //     );

    //     let now = std::time::SystemTime::now();
    //     println!("{:?}", (1..1000000).product::<usize>());
    //     println!(
    //         "0..1000000 product time spent {} us",
    //         now.elapsed().unwrap().as_micros()
    //     );
    // }

    // {
    //     println!("{}", (PI / 2.0f64).cos());
    //     println!("{}", (3.14f64).cos());

    //     println!("{}", signal::cos(PI / 2.0f64));
    //     let now = std::time::SystemTime::now();
    //     // signal::cos(3.14f64);
    //     // let mut a = [0.0;8];
    //     let mut m = [0.0;8];
    //     for i in 0..8 {
    //         m[i] = (i as f64);
    //     }
    //     let a = signal::cos_simd_x8(simd::Simd::from_array(m)).to_array();

    //     // println!("{}", signal::cos(3.14f64));
    //     println!(
    //         " time spent {} ns",
    //         now.elapsed().unwrap().as_nanos()
    //     );
    // }
}
