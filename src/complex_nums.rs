// const EXP_OF_J_F64: C64 = ComplexNums {
//     // real: (1.0 as Float).cos(),
//     // image: (1.0 as Float).sin(),
//     real: 0.5403023058681398 as f64,
//     image: 0.8414709848078965 as f64,
// };

// const EXP_OF_J_F32: C32 = ComplexNums {
//     // real: (1.0 as Float).cos(),
//     // image: (1.0 as Float).sin(),
//     real: 0.5403023058681398 as f32,
//     image: 0.8414709848078965 as f32,
// };
pub type C64 = ComplexNums<f64>;
pub type C32 = ComplexNums<f32>;

#[derive(Clone, Copy, PartialEq)]
pub struct ComplexNums<T> {
    pub real: T,
    pub image: T,
}

impl C64 {
    #[inline]
    pub fn new(r: f64, i: f64) -> Self {
        Self { real: r, image: i }
    }

    #[inline]
    // real = real; image = -image;
    pub fn conj(mut self) -> Self {
        self.image = -self.image;
        self
    }

    #[inline]
    pub fn add(mut self, rhs: f64) -> Self {
        self.real += rhs;
        self
    }

    #[inline]
    pub fn sub(mut self, rhs: f64) -> Self {
        self.real -= rhs;
        self
    }

    #[inline]
    pub fn mul(mut self, rhs: f64) -> Self {
        self.real *= rhs;
        self.image *= rhs;
        self
    }

    #[inline]
    pub fn div(mut self, rhs: f64) -> Self {
        self.real /= rhs;
        self.image /= rhs;
        self
    }

    // #[inline]
    // pub fn exp_of_j() -> Self {
    //     EXP_OF_J_F64
    // }

    pub fn abs(&self) -> f64 {
        (self.real.powi(2) + self.image.powi(2)).sqrt()
    }

    // #[inline]
    pub fn complex_exp(val: f64) -> ComplexNums<f64> {
        ComplexNums {
            real: val.cos(),
            image: val.sin(),
        }
    }
}

impl C32 {
    #[inline]
    pub fn new(r: f32, i: f32) -> Self {
        Self { real: r, image: i }
    }

    #[inline]
    pub fn conj(mut self) -> Self {
        self.image = -self.image;
        self
    }

    #[inline]
    pub fn add(mut self, rhs: f32) -> Self {
        self.real += rhs;
        self
    }

    #[inline]
    pub fn sub(mut self, rhs: f32) -> Self {
        self.real -= rhs;
        self
    }

    #[inline]
    pub fn mul(mut self, rhs: f32) -> Self {
        self.real *= rhs;
        self.image *= rhs;
        self
    }

    #[inline]
    pub fn div(mut self, rhs: f32) -> Self {
        self.real /= rhs;
        self.image /= rhs;
        self
    }

    // #[inline]
    // pub fn exp_of_j() -> Self {
    //     EXP_OF_J_F32
    // }

    pub fn abs(&self) -> f32 {
        (self.real.powi(2) + self.image.powi(2)).sqrt()
    }

    // #[inline]
    pub fn complex_exp(val: f32) -> ComplexNums<f32> {
        ComplexNums {
            real: val.cos(),
            image: val.sin(),
        }
    }
}

impl Default for C64 {
    fn default() -> Self {
        Self {
            real: Default::default(),
            image: Default::default(),
        }
    }
}

impl Default for C32 {
    fn default() -> Self {
        Self {
            real: Default::default(),
            image: Default::default(),
        }
    }
}

impl From<f64> for C32 {
    fn from(value: f64) -> Self {
        Self {
            real: value as f32,
            image: 0.0 as f32,
        }
    }
}

impl From<f32> for C32 {
    fn from(value: f32) -> Self {
        Self {
            real: value as f32,
            image: 0.0 as f32,
        }
    }
}

impl From<f32> for C64 {
    fn from(value: f32) -> Self {
        Self {
            real: value as f64,
            image: 0.0 as f64,
        }
    }
}

impl From<f64> for C64 {
    fn from(value: f64) -> Self {
        Self {
            real: value as f64,
            image: 0.0 as f64,
        }
    }
}

impl From<C64> for C32 {
    fn from(value: C64) -> Self {
        Self {
            real: value.real as f32,
            image: value.image as f32,
        }
    }
}

impl From<C32> for C64 {
    fn from(value: C32) -> Self {
        Self {
            real: value.real as f64,
            image: value.image as f64,
        }
    }
}

// F64
//
impl core::ops::Add for C64 {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl core::ops::Sub for C64 {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl core::ops::Mul for C64 {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl core::ops::Div for C64 {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl core::ops::AddAssign for C64 {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.image += rhs.image;
    }
}

impl core::ops::SubAssign for C64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.real -= rhs.real;
        self.image -= rhs.image;
    }
}

impl core::ops::MulAssign for C64 {
    fn mul_assign(&mut self, rhs: Self) {
        let tmp = self.clone();
        self.real = tmp.real * rhs.real - tmp.image * rhs.image;
        self.image = tmp.real * rhs.image + tmp.image * rhs.real;
    }
}

impl core::ops::DivAssign for C64 {
    fn div_assign(&mut self, rhs: Self) {
        let tmp = self.clone();
        self.real =
            (tmp.real * rhs.real + tmp.image * rhs.image) / (rhs.real.powi(2) + rhs.image.powi(2));
        self.image =
            (tmp.image * rhs.real - tmp.real * rhs.image) / (rhs.real.powi(2) + rhs.image.powi(2));
    }
}

impl core::ops::Neg for C64 {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.real = -self.real;
        self.image = -self.image;
        self
    }
}

// F32
//
impl core::ops::Add for C32 {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl core::ops::Sub for C32 {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl core::ops::Mul for C32 {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl core::ops::Div for C32 {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl core::ops::AddAssign for C32 {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.image += rhs.image;
    }
}

impl core::ops::SubAssign for C32 {
    fn sub_assign(&mut self, rhs: Self) {
        self.real -= rhs.real;
        self.image -= rhs.image;
    }
}

impl core::ops::MulAssign for C32 {
    fn mul_assign(&mut self, rhs: Self) {
        let tmp = self.clone();
        self.real = tmp.real * rhs.real - tmp.image * rhs.image;
        self.image = tmp.real * rhs.image + tmp.image * rhs.real;
    }
}

impl core::ops::DivAssign for C32 {
    fn div_assign(&mut self, rhs: Self) {
        let tmp = self.clone();
        self.real =
            (tmp.real * rhs.real + tmp.image * rhs.image) / (rhs.real.powi(2) + rhs.image.powi(2));
        self.image =
            (tmp.image * rhs.real - tmp.real * rhs.image) / (rhs.real.powi(2) + rhs.image.powi(2));
    }
}

impl core::ops::Neg for C32 {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.real = -self.real;
        self.image = -self.image;
        self
    }
}

impl core::fmt::Debug for ComplexNums<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = if self.image < 0.0 {
            self.real.to_string() + &" - ".to_string() + &(-self.image).to_string()
        } else {
            self.real.to_string() + &" + ".to_string() + &self.image.to_string()
        };
        f.write_fmt(format_args!("[{}j]", str))
    }
}

impl core::fmt::Debug for ComplexNums<f32> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = if self.image < 0.0 {
            self.real.to_string() + &" - ".to_string() + &(-self.image).to_string()
        } else {
            self.real.to_string() + &" + ".to_string() + &self.image.to_string()
        };
        f.write_fmt(format_args!("[{}j]", str))
    }
}

#[cfg(test)]
mod tests {
    use crate::complex_nums::*;

    #[test]
    fn test_add() {
        assert_eq!(
            C64::new(1.2, 2.1) + C64::new(4.1, -3.6),
            C64::new(1.2 + 4.1, 2.1 + -3.6)
        );
        assert_eq!(
            C32::new(1.2, 2.1) + C32::new(4.1, -3.6),
            C32::new(1.2 + 4.1, 2.1 + -3.6)
        );
    }

    #[test]
    fn test_sub() {
        assert_eq!(
            C64::new(1.2, 2.1) - C64::new(4.1, -3.6),
            C64::new(1.2 - 4.1, 2.1 - -3.6)
        );
        assert_eq!(
            C32::new(1.2, 2.1) - C32::new(4.1, -3.6),
            C32::new(1.2 - 4.1, 2.1 - -3.6)
        );
    }

    #[test]
    fn test_mul() {
        assert_eq!(
            C64::new(1.2, 2.1) * C64::new(4.1, -3.6),
            C64::new(1.2 * 4.1 - 2.1 * -3.6, 1.2 * -3.6 + 2.1 * 4.1)
        );
        assert_eq!(
            C32::new(1.2, 2.1) * C32::new(4.1, -3.6),
            C32::new(1.2 * 4.1 - 2.1 * -3.6, 1.2 * -3.6 + 2.1 * 4.1)
        );
    }

    #[test]
    fn test_div() {
        assert_eq!(
            C64::new(1.2, 2.1) / C64::new(4.1, -3.6),
            C64::new(
                (1.2 * 4.1 + 2.1 * -3.6) / (4.1_f64.powi(2) + (-3.6_f64).powi(2)),
                (2.1 * 4.1 - 1.2 * -3.6) / (4.1_f64.powi(2) + (-3.6_f64).powi(2))
            )
        );
        assert_eq!(
            C32::new(1.2, 2.1) / C32::new(4.1, -3.6),
            C32::new(
                (1.2 * 4.1 + 2.1 * -3.6) / (4.1_f32.powi(2) + (-3.6_f32).powi(2)),
                (2.1 * 4.1 - 1.2 * -3.6) / (4.1_f32.powi(2) + (-3.6_f32).powi(2))
            )
        );
    }
}
