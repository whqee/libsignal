use std::{fs::File, io::Write};

mod signal;

fn main() {
    let (freq, amp, offset) = (880.0, 0.5, 0.0);
    let sin_sig = signal::TrigSig::new(signal::Type::Sin, freq, amp, offset);

    let (freq, amp, offset) = (440.0, 1.0, 0.0);
    let cos_sig = signal::TrigSig::new(signal::Type::Cos, freq, amp, offset);

    // let tan_sig = signal::Tan::new(freq, amp, offset);
    // let mut mix_sig = sin_sig.clone() + cos_sig.clone();

    let (freq, amp, offset) = (1000.0, 0.5, 0.0);
    let mut mix_sig = signal::TrigSig::new(signal::Type::Sin, freq, amp, offset);

    // // 方波
    // for i in [
    //     3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
    // ] {
    //     mix_sig += signal::TrigSig::new(signal::Type::Sin, freq * i, amp / i, offset);
    // }

    // // 方波
    // for i in (2..2000).filter(|x| x % 2 != 0) {
    //     let i = i as f32;
    //     mix_sig += signal::TrigSig::new(signal::Type::Sin, freq * i, amp / i, offset);
    // }

    // 三角波
    for i in (2..200).map(|x| x as f32) {
        mix_sig += signal::TrigSig::new(signal::Type::Sin, freq * i, amp / i, offset);
    }

    let (duration, start, framerate) = (2.0, 0.0, 44100);
    let sin_wave = sin_sig.make_wave(duration, start, framerate);
    let cos_wave = cos_sig.make_wave(duration, start, framerate);
    // let tan_wave = tan_sig.make_wave(duration, start, framerate);
    let mix_wave = mix_sig.make_wave(duration, start, framerate);

    let mut file = File::create("test.wav").unwrap();
    // let wave_head = [0u8; 44];
    // file.write(&wave_head).unwrap();

    // file.write(sin_wave.raw_as_ref()).unwrap();
    // file.write(cos_wave.raw_as_ref()).unwrap();
    // file.write(tan_wave.raw_as_ref()).unwrap();
    file.write(mix_wave.raw_as_ref()).unwrap();
}
