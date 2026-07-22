use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::Error;

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ResourceQuantity(i64);
impl ResourceQuantity {
    pub const SCALE: i64 = 1_000;
    pub fn from_units(value: f64) -> Result<Self, Error> {
        if !value.is_finite() || value < 0.0 || value > i64::MAX as f64 / Self::SCALE as f64 {
            return Err(Error::InvalidResource(
                "quantity must be finite, non-negative, and in range".into(),
            ));
        }
        let scaled = (value * Self::SCALE as f64).round() as i64;
        if value > 0.0 && scaled == 0 {
            return Err(Error::InvalidResource(
                "positive quantity is below 0.001 unit".into(),
            ));
        }
        Ok(Self(scaled))
    }
    pub const fn milli(self) -> i64 {
        self.0
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceSet(BTreeMap<String, ResourceQuantity>);
impl ResourceSet {
    pub fn cpu_gpu(cpu: f64, gpu: f64) -> Result<Self, Error> {
        let mut result = Self::default();
        for (name, value) in [("cpu", cpu), ("gpu", gpu)] {
            let q = ResourceQuantity::from_units(value)?;
            if q.milli() != 0 {
                result.0.insert(name.into(), q);
            }
        }
        Ok(result)
    }
    pub fn get(&self, name: &str) -> ResourceQuantity {
        self.0.get(name).copied().unwrap_or_default()
    }
    pub fn can_fit(&self, needed: &Self) -> bool {
        needed
            .0
            .iter()
            .all(|(name, value)| self.get(name) >= *value)
    }
    pub fn subtract(&mut self, needed: &Self) -> Result<(), Error> {
        if !self.can_fit(needed) {
            return Err(Error::InvalidResource("insufficient resources".into()));
        }
        for (name, value) in &needed.0 {
            let remaining = self.get(name).milli() - value.milli();
            if remaining == 0 {
                self.0.remove(name);
            } else {
                self.0.insert(name.clone(), ResourceQuantity(remaining));
            }
        }
        Ok(())
    }
    pub fn add_capped(&mut self, returned: &Self, total: &Self) -> Result<(), Error> {
        let mut next = self.clone();
        for (name, value) in &returned.0 {
            let quantity = next
                .get(name)
                .milli()
                .checked_add(value.milli())
                .ok_or_else(|| Error::InvalidResource("resource overflow".into()))?;
            if quantity > total.get(name).milli() {
                return Err(Error::InvalidResource(
                    "resource release exceeds worker total".into(),
                ));
            }
            if quantity == 0 {
                next.0.remove(name);
            } else {
                next.0.insert(name.clone(), ResourceQuantity(quantity));
            }
        }
        *self = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_is_atomic_and_capped() {
        let total = ResourceSet::cpu_gpu(1.0, 1.0).unwrap();
        let mut available = ResourceSet::cpu_gpu(0.5, 0.5).unwrap();
        let before = available.clone();
        assert!(available
            .add_capped(&ResourceSet::cpu_gpu(0.5, 1.0).unwrap(), &total)
            .is_err());
        assert_eq!(available, before);
    }

    #[test]
    fn rejects_non_representable_quantities() {
        assert!(ResourceQuantity::from_units(f64::NAN).is_err());
        assert!(ResourceQuantity::from_units(-1.0).is_err());
        assert!(ResourceQuantity::from_units(0.0001).is_err());
    }
}
