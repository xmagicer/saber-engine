//! Registered shared host pages for explicit, non-P2P GPU communication.
use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;

use crate::handles::post_destructor;
use crate::thread::Msg;
use crate::{DeviceAddr, Error, HipThread, Result};

/// An owned shared mapping, registered on its owning HIP thread. Not a device allocation.
pub struct SharedHost {
    base: usize,
    address: DeviceAddr,
    bytes: usize,
    map_len: usize,
    path: Option<PathBuf>,
    tx: Sender<Msg>,
    not_send: PhantomData<*mut ()>,
}

impl SharedHost {
    /// The address returned by HIP for these registered pages.
    pub fn addr(&self) -> DeviceAddr {
        self.address
    }
    /// Requested byte length (the mapping itself is page rounded).
    pub fn len(&self) -> u64 {
        self.bytes as u64
    }
    /// The private file name to exchange before unlinking it.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
    /// Remove the owner's file name after both processes have mapped it.
    pub fn unlink(&mut self) -> Result<()> {
        if let Some(p) = self.path.take() {
            std::fs::remove_file(p)
                .map_err(|e| Error::Invalid(format!("unlink shared host pages: {e}")))?;
        }
        Ok(())
    }
}

impl HipThread {
    /// Create private host pages for a peer to open, without enabling peer-device access.
    pub fn shared_host_create(&self, bytes: u64) -> Result<SharedHost> {
        let root = if Path::new("/dev/shm").is_dir() {
            PathBuf::from("/dev/shm")
        } else {
            std::env::temp_dir()
        };
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| Error::Invalid(e.to_string()))?
            .as_nanos();
        let path = root.join(format!("saber-host-ar-{}-{nonce}", std::process::id()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .map_err(|e| Error::Invalid(format!("create shared host pages: {e}")))?;
        let map_len = rounded_len(bytes)?;
        if let Err(e) = file.set_len(map_len as u64) {
            let _ = std::fs::remove_file(&path);
            return Err(Error::Invalid(format!("size shared host pages: {e}")));
        }
        match self.shared_host_map(file, bytes) {
            Ok(mut mapping) => {
                mapping.path = Some(path);
                Ok(mapping)
            }
            Err(e) => {
                let _ = std::fs::remove_file(path);
                Err(e)
            }
        }
    }

    /// Open the peer's private shared pages. A symlink, wrong owner/mode/size is refused.
    pub fn shared_host_open(&self, path: &Path, bytes: u64) -> Result<SharedHost> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .map_err(|e| Error::Invalid(format!("open shared host pages: {e}")))?;
        let meta = file.metadata().map_err(|e| Error::Invalid(e.to_string()))?;
        // SAFETY: geteuid has no preconditions.
        let uid = unsafe { libc::geteuid() };
        if !meta.is_file()
            || meta.uid() != uid
            || meta.mode() & 0o777 != 0o600
            || meta.len() != rounded_len(bytes)? as u64
        {
            return Err(Error::Invalid(
                "shared host file owner, mode or size mismatch".into(),
            ));
        }
        self.shared_host_map(file, bytes)
    }

    fn shared_host_map(&self, file: File, bytes: u64) -> Result<SharedHost> {
        let map_len = rounded_len(bytes)?;
        // SAFETY: file is live, read/write, and validated/sized by the caller.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(Error::Invalid(format!(
                "mmap shared host pages: {}",
                std::io::Error::last_os_error()
            )));
        }
        let base = raw as usize;
        let registered = self.run(move |lib, ctx| {
            // SAFETY: the live mapping remains owned until unregister has returned.
            if let Err(e) =
                unsafe { lib.host_register(ctx, base as *mut _, map_len as u64, 0x1 | 0x2) }
            {
                return Ok((Err(e), false));
            }
            // SAFETY: registered immediately above, no coarse-grained registration flag.
            match unsafe { lib.host_get_device_pointer(ctx, base as *mut _) } {
                Ok(p) => Ok((Ok(DeviceAddr::from_raw(p)), true)),
                Err(e) => {
                    // SAFETY: registration succeeded above; no GPU operation has used it.
                    let still_registered =
                        unsafe { lib.host_unregister(ctx, base as *mut _) }.is_err();
                    Ok((Err(e), still_registered))
                }
            }
        });
        match registered {
            Ok((Ok(address), _)) => Ok(SharedHost {
                base,
                address,
                bytes: bytes as usize,
                map_len,
                path: None,
                tx: self.sender(),
                not_send: PhantomData,
            }),
            Ok((Err(e), false)) => {
                // SAFETY: registration never succeeded or unregister completed successfully.
                unsafe { libc::munmap(base as *mut _, map_len) };
                Err(e)
            }
            // A failed job/cleanup may leave driver references: retain the pages.
            Ok((Err(e), true)) | Err(e) => Err(e),
        }
    }
}

fn rounded_len(bytes: u64) -> Result<usize> {
    let n = usize::try_from(bytes)
        .map_err(|_| Error::Invalid("shared host byte size overflow".into()))?;
    if n == 0 {
        return Err(Error::Invalid("zero shared host byte size".into()));
    }
    n.checked_add(4095)
        .map(|x| x & !4095)
        .ok_or_else(|| Error::Invalid("shared host mapping size overflow".into()))
}

impl Drop for SharedHost {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_file(path);
        }
        let (base, len) = (self.base, self.map_len);
        post_destructor(&self.tx, "shared host pages", move |lib, ctx| {
            // SAFETY: this handle uniquely owns the registration, after queued graph teardown.
            unsafe { lib.host_unregister(ctx, base as *mut _) }?;
            // SAFETY: unregister succeeded; no driver references remain in this process.
            unsafe { libc::munmap(base as *mut _, len) };
            Ok(())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_are_page_rounded_without_overflow() {
        assert_eq!(rounded_len(1).unwrap(), 4096);
        assert_eq!(rounded_len(4096).unwrap(), 4096);
        assert_eq!(rounded_len(4097).unwrap(), 8192);
        assert!(rounded_len(0).is_err());
        assert!(rounded_len(u64::MAX).is_err());
    }
}
