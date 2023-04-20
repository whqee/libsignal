pub type C64 = ComplexNums<f64>;
pub type C32 = ComplexNums<f32>;

#[derive(Clone, Copy, PartialEq)]
pub struct ComplexNums<T> {
    pub real: T,
    pub image: T,
}

macro_rules! impl_complexnums {
    ($float: ty, $float2: ty) => {
        impl ComplexNums<$float> {
            #[inline]
            pub fn new(r: $float, i: $float) -> Self {
                Self { real: r, image: i }
            }

            #[inline]
            // real = real; image = -image;
            pub fn conj(mut self) -> Self {
                self.image = -self.image;
                self
            }

            #[inline]
            pub fn add(mut self, rhs: $float) -> Self {
                self.real += rhs;
                self
            }

            #[inline]
            pub fn sub(mut self, rhs: $float) -> Self {
                self.real -= rhs;
                self
            }

            #[inline]
            pub fn mul(mut self, rhs: $float) -> Self {
                self.real *= rhs;
                self.image *= rhs;
                self
            }

            #[inline]
            pub fn div(mut self, rhs: $float) -> Self {
                self.real /= rhs;
                self.image /= rhs;
                self
            }

            pub fn abs(&self) -> $float {
                (self.real.powi(2) + self.image.powi(2)).sqrt()
            }

            // #[inline]
            pub fn complex_exp(val: $float) -> ComplexNums<$float> {
                ComplexNums {
                    real: val.cos(),
                    image: val.sin(),
                }
            }
        }

        impl Default for ComplexNums<$float> {
            fn default() -> Self {
                Self {
                    real: Default::default(),
                    image: Default::default(),
                }
            }
        }

        impl From<$float2> for ComplexNums<$float> {
            fn from(value: $float2) -> Self {
                Self {
                    real: value as $float,
                    image: 0.0 as $float,
                }
            }
        }

        impl From<$float> for ComplexNums<$float> {
            fn from(value: $float) -> Self {
                Self {
                    real: value as $float,
                    image: 0.0 as $float,
                }
            }
        }

        impl From<ComplexNums<$float2>> for ComplexNums<$float> {
            fn from(value: ComplexNums<$float2>) -> Self {
                Self {
                    real: value.real as $float,
                    image: value.image as $float,
                }
            }
        }

        impl core::ops::Add for ComplexNums<$float> {
            type Output = Self;

            fn add(mut self, rhs: Self) -> Self::Output {
                self += rhs;
                self
            }
        }

        impl core::ops::Sub for ComplexNums<$float> {
            type Output = Self;

            fn sub(mut self, rhs: Self) -> Self::Output {
                self -= rhs;
                self
            }
        }

        impl core::ops::Mul for ComplexNums<$float> {
            type Output = Self;

            fn mul(mut self, rhs: Self) -> Self::Output {
                self *= rhs;
                self
            }
        }

        impl core::ops::Div for ComplexNums<$float> {
            type Output = Self;

            fn div(mut self, rhs: Self) -> Self::Output {
                self /= rhs;
                self
            }
        }

        impl core::ops::AddAssign for ComplexNums<$float> {
            fn add_assign(&mut self, rhs: Self) {
                self.real += rhs.real;
                self.image += rhs.image;
            }
        }

        impl core::ops::SubAssign for ComplexNums<$float> {
            fn sub_assign(&mut self, rhs: Self) {
                self.real -= rhs.real;
                self.image -= rhs.image;
            }
        }

        impl core::ops::MulAssign for ComplexNums<$float> {
            fn mul_assign(&mut self, rhs: Self) {
                let tmp = self.clone();
                self.real = tmp.real * rhs.real - tmp.image * rhs.image;
                self.image = tmp.real * rhs.image + tmp.image * rhs.real;
            }
        }

        impl core::ops::DivAssign for ComplexNums<$float> {
            fn div_assign(&mut self, rhs: Self) {
                let tmp = self.clone();
                self.real = (tmp.real * rhs.real + tmp.image * rhs.image)
                    / (rhs.real.powi(2) + rhs.image.powi(2));
                self.image = (tmp.image * rhs.real - tmp.real * rhs.image)
                    / (rhs.real.powi(2) + rhs.image.powi(2));
            }
        }

        impl core::ops::Neg for ComplexNums<$float> {
            type Output = Self;

            fn neg(mut self) -> Self::Output {
                self.real = -self.real;
                self.image = -self.image;
                self
            }
        }

        impl core::fmt::Debug for ComplexNums<$float> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let str = if self.image < 0.0 {
                    self.real.to_string() + &" - ".to_string() + &(-self.image).to_string()
                } else {
                    self.real.to_string() + &" + ".to_string() + &self.image.to_string()
                };
                f.write_fmt(format_args!("[{}j]", str))
            }
        }
    };
}

impl_complexnums!(f64, f32);
impl_complexnums!(f32, f64);

macro_rules! tests {
    ($name: ident, $float: ty) => {
        #[cfg(test)]
        mod $name {
            use crate::complex_nums::*;

            #[test]
            fn test_add() {
                assert_eq!(
                    ComplexNums::<$float>::new(1.2, 2.1) + ComplexNums::<$float>::new(4.1, -3.6),
                    ComplexNums::<$float>::new(1.2 + 4.1, 2.1 + -3.6)
                );
            }

            #[test]
            fn test_sub() {
                assert_eq!(
                    ComplexNums::<$float>::new(1.2, 2.1) - ComplexNums::<$float>::new(4.1, -3.6),
                    ComplexNums::<$float>::new(1.2 - 4.1, 2.1 - -3.6)
                );
            }

            #[test]
            fn test_mul() {
                assert_eq!(
                    ComplexNums::<$float>::new(1.2, 2.1) * ComplexNums::<$float>::new(4.1, -3.6),
                    ComplexNums::<$float>::new(1.2 * 4.1 - 2.1 * -3.6, 1.2 * -3.6 + 2.1 * 4.1)
                );
            }

            #[test]
            fn test_div() {
                assert_eq!(
                    ComplexNums::<$float>::new(1.2, 2.1) / ComplexNums::<$float>::new(4.1, -3.6),
                    ComplexNums::<$float>::new(
                        (1.2 * 4.1 + 2.1 * -3.6)
                            / ((4.1 as $float).powi(2) + (-3.6 as $float).powi(2)),
                        (2.1 * 4.1 - 1.2 * -3.6)
                            / ((4.1 as $float).powi(2) + (-3.6 as $float).powi(2))
                    )
                );
            }
        }
    };
}

tests!(c64, f64);
tests!(c32, f32);
