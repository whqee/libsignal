use std::{fs::File, io::Write};

mod signal;

fn main() {
    let (freq, amp, offset) = (880, 0.5, 0.0);
    let sin_sig = signal::Sin::new(freq, amp, offset);
    
    let (freq, amp, offset) = (440, 1.0, 0.0);
    let cos_sig = signal::Cos::new(freq, amp, offset);
    
    // let tan_sig = signal::Tan::new(freq, amp, offset);
    // let mix_sig = sin_sig + cos_sig;
    
    let (duration, start, framerate) = (1.5, 0.0, 44100);
    let sin_wave = sin_sig.make_wave(duration, start, framerate);
    let cos_wave = cos_sig.make_wave(duration, start, framerate);
    // let tan_wave = tan_sig.make_wave(duration, start, framerate);
    // let mix_wave = mix_sig.make_wave(duration, start, framerate);

    let mut file = File::create("test.wav").unwrap();
    // let wave_head = [0u8; 44];
    // file.write(&wave_head).unwrap();

    file.write(sin_wave.raw_as_ref()).unwrap();
    file.write(cos_wave.raw_as_ref()).unwrap();
    // file.write(tan_wave.raw_as_ref()).unwrap();
    // file.write(mix_wave.raw_as_ref()).unwrap();
}
