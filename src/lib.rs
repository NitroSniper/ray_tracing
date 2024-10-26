

#[derive(Copy, Clone, Default)]
pub struct GenericVec3<T> {
    e: [T; 3],
}
 
impl<T: Clone + Copy> GenericVec3<T> {
    pub fn new(x: T, y: T, z: T) -> GenericVec3<T> {
        GenericVec3 { e: [x, y, z] }
    }
 
    pub fn x(&self) -> T {
        self.e[0]
    }
 
    pub fn y(&self) -> T {
        self.e[1]
    }
 
    pub fn z(&self) -> T {
        self.e[2]
    }
}
 

pub type Vec3 = GenericVec3<f64>;
