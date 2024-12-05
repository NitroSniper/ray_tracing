use cgmath::{BaseFloat, InnerSpace, Vector3};
use rand::Rng;

pub struct Ray<S> {
    pub orig: Vector3<S>,
    pub dir: Vector3<S>,
}

impl<S: BaseFloat> Ray<S> {
    pub fn new(orig: Vector3<S>, dir: Vector3<S>) -> Self {
        Self { orig, dir }
    }

    pub fn at(&self, lambda: S) -> Vector3<S> {
        self.orig + self.dir * lambda
    }
}

pub trait VectorRayExt<S> {
    fn random_unit_vector() -> Vector3<S>;
    fn invert_if_dot(&mut self, rhs: &Vector3<S>, is_negative: bool) -> &mut Self;
}

impl<S: BaseFloat> VectorRayExt<S> for Vector3<S> {
    fn random_unit_vector() -> Vector3<S> {
        let mut rng = rand::thread_rng();
        loop {
            let p = Vector3::<S>::new(
                S::from(rng.gen_range(-1.0..=1.0))
                    .expect("-1.0..1.0 should be viable as BaseFloat"),
                S::from(rng.gen_range(-1.0..=1.0))
                    .expect("-1.0..1.0 should be viable as BaseFloat"),
                S::from(rng.gen_range(-1.0..=1.0))
                    .expect("-1.0..1.0 should be viable as BaseFloat"),
            );
            // check if it is within circle; this filter removes any bias
            if p.dot(p) <= S::one() {
                return p.normalize();
            }
        }
    }

    fn invert_if_dot(&mut self, rhs: &Vector3<S>, is_negative: bool) -> &mut Vector3<S> {
        let val = self.dot(*rhs);
        let minus_one = -S::one();
        match (val.is_sign_negative(), is_negative) {
            (true, true) => *self *= minus_one,
            (false, false) => *self *= minus_one,
            _ => (),
        };
        self
    }
}
