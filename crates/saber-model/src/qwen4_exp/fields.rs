//! A JSON object reader that refuses what it was not told about.
//!
//! Every `config.json` object the `qwen4_exp` config reads goes through [`Fields`]: each key is
//! either TAKEN (parsed into a typed field), ACKNOWLEDGED (named here as present-but-unused by the
//! forward, e.g. `initializer_range`), or it is an error. [`Fields::finish`] refuses the first key
//! nobody took, by its full dotted path. A missing required key is refused by name too. There are
//! no silent defaults: a value the checkpoint omits and upstream fills in is a named
//! [`super::config::DeclaredDefault`], never an `unwrap_or`.

use serde_json::{Map, Value};

use crate::ModelError;

/// One JSON object under a dotted path, with the set of keys consumed so far.
pub(crate) struct Fields<'a> {
    path: String,
    obj: &'a Map<String, Value>,
    taken: Vec<&'a str>,
}

fn err(msg: String) -> ModelError {
    ModelError::Config(msg)
}

impl<'a> Fields<'a> {
    /// Wrap `value` (which must be an object) found at `path`.
    pub(crate) fn new(path: &str, value: &'a Value) -> Result<Self, ModelError> {
        let obj = value
            .as_object()
            .ok_or_else(|| err(format!("{path}: expected an object")))?;
        Ok(Self {
            path: path.to_string(),
            obj,
            taken: Vec::new(),
        })
    }

    fn key_path(&self, key: &str) -> String {
        if self.path.is_empty() {
            key.to_string()
        } else {
            format!("{}.{key}", self.path)
        }
    }

    fn mark(&mut self, key: &str) {
        if let Some((k, _)) = self.obj.get_key_value(key) {
            self.taken.push(k.as_str());
        }
    }

    /// The raw value; `Err` when absent.
    pub(crate) fn value(&mut self, key: &str) -> Result<&'a Value, ModelError> {
        self.mark(key);
        self.obj
            .get(key)
            .ok_or_else(|| err(format!("{}: missing required field", self.key_path(key))))
    }

    /// The raw value, or `None` when the key is absent (the caller decides what absence means and
    /// must name it — see `DeclaredDefault`).
    pub(crate) fn optional(&mut self, key: &str) -> Option<&'a Value> {
        self.mark(key);
        self.obj.get(key)
    }

    /// Present in the file, not used by the forward. Absence is fine; any value is fine.
    pub(crate) fn ack(&mut self, key: &str) {
        self.mark(key);
    }

    /// Present in the file and required to hold exactly `want`.
    pub(crate) fn expect(&mut self, key: &str, want: &Value) -> Result<(), ModelError> {
        let v = self.value(key)?;
        if v != want {
            return Err(err(format!(
                "{}: expected {want}, the file says {v}",
                self.key_path(key)
            )));
        }
        Ok(())
    }

    /// Absent, or present and exactly `want` (a key some revisions of a checkpoint state and
    /// others omit, with one meaning either way).
    pub(crate) fn expect_if_present(&mut self, key: &str, want: &Value) -> Result<(), ModelError> {
        match self.optional(key) {
            None => Ok(()),
            Some(v) if v == want => Ok(()),
            Some(v) => Err(err(format!(
                "{}: expected {want} (or absent), the file says {v}",
                self.key_path(key)
            ))),
        }
    }

    pub(crate) fn u32(&mut self, key: &str) -> Result<u32, ModelError> {
        let path = self.key_path(key);
        let v = self.value(key)?;
        v.as_u64()
            .and_then(|x| u32::try_from(x).ok())
            .ok_or_else(|| {
                err(format!(
                    "{path}: expected an unsigned 32-bit integer, got {v}"
                ))
            })
    }

    pub(crate) fn u64(&mut self, key: &str) -> Result<u64, ModelError> {
        let path = self.key_path(key);
        let v = self.value(key)?;
        v.as_u64()
            .ok_or_else(|| err(format!("{path}: expected an unsigned integer, got {v}")))
    }

    pub(crate) fn f64(&mut self, key: &str) -> Result<f64, ModelError> {
        let path = self.key_path(key);
        let v = self.value(key)?;
        v.as_f64()
            .ok_or_else(|| err(format!("{path}: expected a number, got {v}")))
    }

    pub(crate) fn bool(&mut self, key: &str) -> Result<bool, ModelError> {
        let path = self.key_path(key);
        let v = self.value(key)?;
        v.as_bool()
            .ok_or_else(|| err(format!("{path}: expected a boolean, got {v}")))
    }

    pub(crate) fn str(&mut self, key: &str) -> Result<&'a str, ModelError> {
        let path = self.key_path(key);
        let v = self.value(key)?;
        v.as_str()
            .ok_or_else(|| err(format!("{path}: expected a string, got {v}")))
    }

    pub(crate) fn array(&mut self, key: &str) -> Result<&'a Vec<Value>, ModelError> {
        let path = self.key_path(key);
        let v = self.value(key)?;
        v.as_array()
            .ok_or_else(|| err(format!("{path}: expected an array, got {v}")))
    }

    pub(crate) fn u32_array(&mut self, key: &str) -> Result<Vec<u32>, ModelError> {
        let path = self.key_path(key);
        self.array(key)?
            .iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|y| u32::try_from(y).ok())
                    .ok_or_else(|| err(format!("{path}: expected unsigned integers, got {x}")))
            })
            .collect()
    }

    /// A nested object, as its own `Fields`.
    pub(crate) fn object(&mut self, key: &str) -> Result<Fields<'a>, ModelError> {
        let path = self.key_path(key);
        let v = self.value(key)?;
        Fields::new(&path, v)
    }

    /// Refuse the first key nobody took or acknowledged.
    pub(crate) fn finish(self) -> Result<(), ModelError> {
        for k in self.obj.keys() {
            if !self.taken.contains(&k.as_str()) {
                return Err(err(format!(
                    "{}: unknown field (not read by the qwen4_exp config; name it or refuse the \
                     checkpoint)",
                    self.key_path(k)
                )));
            }
        }
        Ok(())
    }
}
