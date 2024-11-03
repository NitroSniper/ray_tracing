use cgmath::Vector3;


pub struct Ray<S> {
    pub orig: Vector3<S>,
    pub dir: Vector3<S>
}

impl<S: cgmath::BaseFloat> Ray<S> {
    pub fn new(orig: Vector3<S>, dir: Vector3<S>) -> Self {
        Self {
            orig,
            dir
        }
    }

    pub fn at(&self, lambda: S) -> Vector3<S> {
        self.orig + self.dir*lambda
    }
}